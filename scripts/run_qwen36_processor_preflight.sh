#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
DATA=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e5_crop_aux20_json_v1
TRAIN="$DATA/train.jsonl"
OUT=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/qwen36_27b/processor_preflight_v1

test -x "$ENV/bin/python"
test -r "$MODEL/config.json"
test -r "$TRAIN"
test -r "$REPO/scripts/check_qwen36_processor_preflight.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=1024
export TOKENIZERS_PARALLELISM=false

"$ENV/bin/python" "$REPO/scripts/check_qwen36_processor_preflight.py" \
    --model "$MODEL" \
    --train "$TRAIN" \
    --output "$OUT/summary.json" \
    --max-length 3072 \
    2>&1 | tee "$OUT/preflight.log"
