#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_eval.sh - Simple wrapper to run infer_chatas.py with different models
# ------------------------------------------------------------------------------

set -e

# Show usage if no arguments or help flag
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 <model_name> [batch_size] [outer_batch_size]"
  echo ""
  echo "Models:"
  echo "  paligemma    - PaliGemma 2 3B PT-224"
  echo "  qwen         - Qwen2-VL-2B-Instruct"
  echo ""
  echo "Examples:"
  echo "  $0 paligemma         # Run PaliGemma with default batch sizes"
  echo "  $0 qwen 16 4         # Run Qwen with batch_size=16, outer_batch_size=4"
  exit 1
fi

MODEL_NAME="$1"
BATCH_SIZE="${2:-16}"  # Default to 16 if not provided
OUTER_BATCH="${3:-32}" # Default to 32 if not provided

case "$MODEL_NAME" in
  paligemma)
    OUTPUT_FILE="out.all.imagechat.paligemma2.3b.pt224"
    ADAPTER="ckpts/exp_output_paligemma_imgchat/v3-20250529-040720/checkpoint-105000"
    MODEL="google/paligemma2-3b-pt-224"
    ;;
  qwen)
    OUTPUT_FILE="out.all.imagechat.qwen2.vl.2b.instruct"
    ADAPTER="ckpts/exp_output_qwen2_vl_imagechat/v1-20250529-055538/checkpoint-103000"
    MODEL="qwen/Qwen2-VL-2B-Instruct"
    ;;
  *)
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: paligemma, qwen"
    exit 1
    ;;
esac

echo "Running $MODEL_NAME evaluation:"
echo "  Model:           $MODEL"
echo "  Adapter:         $ADAPTER"
echo "  Output file:     $OUTPUT_FILE"
echo "  Batch size:      $BATCH_SIZE"
echo "  Outer batch:     $OUTER_BATCH"
echo ""

python infer_chatas.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --output_file "$OUTPUT_FILE" \
  --batch_size "$BATCH_SIZE" \
  --outer_batch_size "$OUTER_BATCH" \
  "$@" 
