#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
TRAIN="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"
OUT_DIR="$ROOT/evaluations/e4_crop_aux_v1/token_preflight_v1"
POC_DIR="$OUT_DIR/poc"

command -v python >/dev/null
test -r "$MODEL/config.json"
test -r "$TRAIN"
test -r "$REPO/scripts/build_e4_token_poc.py"
test -r "$REPO/scripts/check_e4_token_lengths.py"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: output directory already exists: $OUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
python "$REPO/scripts/build_e4_token_poc.py" \
    --train "$TRAIN" \
    --output-dir "$POC_DIR" \
    --expected-rows 8026 \
    2>&1 | tee "$OUT_DIR/build.log"

export CUDA_VISIBLE_DEVICES=""
export IMAGE_MAX_TOKEN_NUM=1024
python "$REPO/scripts/check_e4_token_lengths.py" \
    --model "$MODEL" \
    --poc "$POC_DIR/poc.jsonl" \
    --manifest "$POC_DIR/manifest.json" \
    --output "$OUT_DIR/token-summary.json" \
    2>&1 | tee "$OUT_DIR/token-check.log"
