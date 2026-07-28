#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: inference PoC failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
TRAIN="$ROOT/ms_swift/e5_crop_aux20_json_v1/train.jsonl"
OUT="$ROOT/evaluations/qwen36_27b/inference_poc20_v1"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
GPU=4

require_executable() {
    [[ -x "$1" ]] || { echo "ERROR: missing executable: $1" >&2; exit 1; }
}

require_readable() {
    [[ -r "$1" ]] || { echo "ERROR: missing or unreadable file: $1" >&2; exit 1; }
}

require_executable "$ENV/bin/python"
require_readable "$MODEL/config.json"
require_readable "$DEV"
require_readable "$TRAIN"
require_readable "$REPO/scripts/run_qwen36_inference_poc.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

echo "=== GPU PREFLIGHT ==="
for CHECK_GPU in 4 5 6 7; do
    FREE=$(nvidia-smi -i "$CHECK_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    echo "GPU $CHECK_GPU free: ${FREE} MiB"
    if [[ "$FREE" -lt 70000 ]]; then
        echo "ERROR: GPU $CHECK_GPU has less than 70000 MiB free" >&2
        exit 1
    fi
done

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=1024
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV/bin/python" "$REPO/scripts/run_qwen36_inference_poc.py" \
    --model "$MODEL" \
    --dev "$DEV" \
    --expected-dev-sha256 "$EXPECTED_DEV_SHA256" \
    --train "$TRAIN" \
    --output-dir "$OUT/artifacts" \
    2>&1 | tee "$OUT/inference.log"

echo "=== GPU AFTER PROCESS EXIT ==="
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.free --format=csv,noheader
