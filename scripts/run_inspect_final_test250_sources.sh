#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
WORKBOOK=/home/data/h30082292/DATA_71/h30082292/data/pose/label/pose_label/pose_260428_20260724-v1.xlsx
IMAGE_ROOT=/home/data/h30082292/DATA_71/h30082292/data/pose/pose/260428

test -r "$WORKBOOK"
test -d "$IMAGE_ROOT"
test -r "$REPO/scripts/inspect_final_test250_sources.py"

export CUDA_VISIBLE_DEVICES=""

python "$REPO/scripts/inspect_final_test250_sources.py" \
    --workbook "$WORKBOOK" \
    --image-root "$IMAGE_ROOT"
