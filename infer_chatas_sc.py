# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal

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

from transformers import StoppingCriteria, StoppingCriteriaList
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

OBS = 4
BS = 1
OUTPUT_FILE = "out.all.mmdd.minicpm.no_img.0.1"
OUTPUT_DIR = "output/"
NO_IMG = True
ADAPTER= "/home/anubhab-pg/sm745052/swift/output/MiniCPM-V-2_6/v5-20250320-065856/checkpoint-2500"

mae = []

# Custom stopping criteria based on token-level confidence
class ConfidenceStoppingCriteria(StoppingCriteria):
    def __init__(self, threshold: float, batch_size: int=1):
        """
        Args:
            threshold (float): Stop generation if the average maximum probability across beams falls below this threshold.
        """
        self.threshold = threshold
        self.batch_size = batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        print("Called")
        if isinstance(scores, tuple):
            scores = scores[0]  # logits are first element
        # Extract the tensor from the tuple
        scores_tensor = scores[0]  # shape: [batch_size*num_beams, vocab_size]
        
        # Calculate number of beams based on the provided batch_size
        num_beams = scores_tensor.shape[0] // self.batch_size
        
        # Reshape scores_tensor to [batch_size, num_beams, vocab_size]
        scores_tensor = scores_tensor.view(self.batch_size, num_beams, -1)
        
        # Compute softmax probabilities along the vocabulary dimension
        probs = scores_tensor.softmax(dim=-1)
        
        # Compute the entropy for each beam:
        # entropy = -sum(p * log(p)) over the vocab dimension.
        # We add a small epsilon to avoid log(0) issues.
        eps = 1e-12
        entropy = -(probs * (probs + eps).log()).sum(dim=-1)  # shape: [batch_size, num_beams]
        avg_entropy = entropy.mean(dim=-1)  # shape: [batch_size]
        decisions = avg_entropy > self.threshold  # boolean tensor of shape [batch_size]
        mae.append(avg_entropy)
        return decisions[0]


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
            n_samples=2,
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
            n_samples=1100,
            to_split=True,
        )
    data = []
    for dialog, suffix in test_data:
        if len([i for i in dialog.utterances[:-1] if len(i.images) > 0]) == 0:
            continue
        data.append((dialog.idx, transform_dialog_data_to_message(dialog, suffix)))
    return data


def infer_batch(engine: "InferEngine", infer_requests: List["InferRequest"]):
    stopper = ConfidenceStoppingCriteria(threshold=0.1, batch_size=BS)
    stopping_criteria = StoppingCriteriaList([stopper])
    request_config = RequestConfig(max_tokens=512, temperature=0, stopping_criteria=stopping_criteria)
    resp_list = engine.infer(infer_requests, request_config)
    # print(f"resp_list: {resp_list}")
    return resp_list


def make_outer_batches(data: List[dict[str, any]]) -> List[List[dict[str, any]]]:
    outer_batches = []
    for i in range(0, len(data), OBS):
        batch = data[i : i + OBS]
        outer_batches.append(batch)
    return outer_batches


def process_outer_batch(
    outer_batch: List[dict[str, any]], engine: "InferEngine"
) -> None:
    infer_requests = [InferRequest(messages=[message]) for _, message in outer_batch]
    infer_idxs = [idx for idx, _ in outer_batch]
    resp = infer_batch(engine, infer_requests)
    pd.DataFrame(
        {
            "idx": infer_idxs,
            "pred": [r.choices[0].message.content for r in resp],
        }
    ).to_csv(
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
        pd.DataFrame(columns=["idx", "pred"]).to_csv(
            os.path.join(OUTPUT_DIR, OUTPUT_FILE), index=False, sep=";"
        )
    for outer_batch in tqdm(outer_batches):
        process_outer_batch(outer_batch, engine)
