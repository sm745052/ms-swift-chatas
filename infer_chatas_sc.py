# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal
import sys

from tqdm import tqdm

from chatas.code.utils.dataset import (
    Dialog,
    MMDDData,
    create_image_path_by_url_mmdd,
    ImageChatData,
    create_image_path_by_url_image_chat,
)
import pandas as pd
from swift.llm import (
    InferEngine,
    InferRequest,
    PtEngine,
    RequestConfig,
)
from argparse import ArgumentParser

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

THRESHOLD = 0.0
OBS = 32
BS = 8
OUTPUT_FILE = "logprobs.minicpm" #f"out.all.mmdd.minicpm.minp.{THRESHOLD}"
OUTPUT_DIR = "output/"
NO_IMG = False
ADAPTER= "output/MiniCPM-V-2_6/v27-20250320-103302/checkpoint-3000"

def transform_dialog_data_to_message(dialog: Dialog, suffix: str) -> dict[str, any]:
    query = ""
    images = []
    for utterance in dialog.utterances[:-1]:
        if len(utterance.images) > 0:
            if not NO_IMG:
                images.extend(utterance.images)
            query += f"{utterance.speaker}:<|IMAGE|>\n"
        else:
            query += f"{utterance.speaker}:{utterance.text}\n"
    images.extend(dialog.utterances[-1].images)
    query += f"{dialog.utterances[-1].speaker}:{dialog.utterances[-1].text}"
    if NO_IMG:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
            ]
        }
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": images[0],
            },
            {"type": "text", "text": query},
        ],
    }


def get_data(dataset_name: str = "mmdd") -> List[dict[str, any]]:
    if dataset_name == "mmdd":
        test_data = MMDDData(
            path="/home/anubhab-pg/CHAT-AS-MULTIMODAL/data/MMDD/test.csv",
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url_mmdd(
                "/home/anubhab-pg/CHAT-AS-MULTIMODAL/data/MMDD/images"
            ),
            to_unroll=True,
            min_images_per_dialog=1,
            # n_samples=2,
            to_split=True,
        )
    elif dataset_name == "image_chat":
        test_data = ImageChatData(
            path="../../mnt/anubhab/ParlAI/data/image_chat/test.csv",
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url_image_chat("../../mnt/anubhab/ParlAI/data/yfcc_images"),
            to_unroll=True,
            min_images_per_dialog=1,
            # n_samples=2,
            to_split=True,
        )
    data = []
    for dialog, suffix in test_data:
        if len([i for i in dialog.utterances[:-1] if len(i.images) > 0]) == 0:
            continue
        data.append((dialog.idx, transform_dialog_data_to_message(dialog, suffix)))
    return data


def infer_batch(engine: "InferEngine", infer_requests: List["InferRequest"]):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer(infer_requests, request_config, use_stopping_criteria=3, threshold=THRESHOLD)
    # print(f"resp_list: {resp_list}")
    return resp_list


def make_outer_batches(data: List[dict[str, any]]) -> List[List[dict[str, any]]]:
    outer_batches = []
    for i in range(0, len(data), OBS):
        batch = data[i : i + OBS]
        outer_batches.append(batch)
    return outer_batches


# def process_outer_batch(
#     outer_batch: List[dict[str, any]], engine: "InferEngine"
# ) -> None:
#     infer_requests = [InferRequest(messages=[message]) for _, message in outer_batch]
#     infer_idxs = [idx for idx, _ in outer_batch]
#     resp = infer_batch(engine, infer_requests)
#     print(resp)
#     pd.DataFrame(
#         {
#             "idx": infer_idxs,
#             "pred": [r.choices[0].message.content for r in resp],
#             # "logprobs": [r.choices[0].logprobs for r in resp],
#             "nll": [
#                 torch.tensor(r.choices[0].logprobs).sum().item() if r.choices[0].message.logprobs else None
#                 for r in resp
#             ],
#         }
#     ).to_csv(
#         os.path.join(OUTPUT_DIR, OUTPUT_FILE),
#         index=False,
#         sep=";",
#         mode="a",
#         header=False,
#     )

def process_outer_batch(
    outer_batch: List[dict[str, any]], engine: "InferEngine"
) -> None:
    infer_requests = [InferRequest(messages=[message]) for _, message in outer_batch]
    infer_idxs = [idx for idx, _ in outer_batch]
    resp = infer_batch(engine, infer_requests)
    
    # For debug
    print(resp)

    def compute_nll(logprobs_dict):
        if not logprobs_dict or "content" not in logprobs_dict:
            return None
        try:
            logprobs = [entry["logprob"] for entry in logprobs_dict["content"]]
            first_token_logprob = logprobs[0] if logprobs else 0.0
            return -sum(logprobs), first_token_logprob
        except Exception as e:
            print(f"Error computing NLL: {e}")
            return None

    output_logprobs = [compute_nll(r.choices[0].logprobs) for r in resp]
    df = pd.DataFrame(
        {
            "idx": infer_idxs,
            "pred": [r.choices[0].message.content for r in resp],
            "nll": [logprob[0] if logprob else None for logprob in output_logprobs],
            "first_token_logprob": [logprob[1] if logprob else None for logprob in output_logprobs],
        }
    )

    df.to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILE),
        index=False,
        sep=";",
        mode="a",
        header=False,
    )
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    args = parser.parse_args()
    model = "openbmb/MiniCPM-V-2_6"
    adapter = ADAPTER
    engine = PtEngine(model, max_batch_size=BS, adapters=[adapter])
    
    
    dataset = get_data()
    if args.resume:
        # Read existing file to get the last idx
        print(f"reading {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
        df = pd.read_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILE), sep=";")
        print(f"read {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
        already_present = set(df["idx"].values)
        dataset = [d for d in dataset if d[0] not in already_present]
        print(f"skipping {len(df)} samples")
    # Create outer batches
    outer_batches = make_outer_batches(dataset)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not args.resume:
        # Write header
        pd.DataFrame(columns=["idx", "pred", "nll", "first_token_lp"]).to_csv(
            os.path.join(OUTPUT_DIR, OUTPUT_FILE), index=False, sep=";"
        )
    for outer_batch in tqdm(outer_batches):
        process_outer_batch(outer_batch, engine)
