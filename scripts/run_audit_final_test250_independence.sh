#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
WORKBOOK=/home/data/h30082292/DATA_71/h30082292/data/pose/label/pose_label/pose_260428_20260724-v1.xlsx
IMAGE_ROOT=/home/data/h30082292/DATA_71/h30082292/data/pose/pose/260428
TRAIN="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"

test -r "$WORKBOOK"
test -d "$IMAGE_ROOT"
test -r "$TRAIN"
test -r "$DEV"
test -r "$REPO/scripts/audit_final_test250_independence.py"

export CUDA_VISIBLE_DEVICES=""

python "$REPO/scripts/audit_final_test250_independence.py" \
    --workbook "$WORKBOOK" \
    --image-root "$IMAGE_ROOT" \
    --train "$TRAIN" \
    --dev "$DEV"
