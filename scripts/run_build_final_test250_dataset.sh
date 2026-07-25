#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
WORKBOOK=/home/data/h30082292/DATA_71/h30082292/data/pose/label/pose_label/pose_260428_20260724-v1.xlsx
IMAGE_ROOT=/home/data/h30082292/DATA_71/h30082292/data/pose/pose/260428
OUTPUT="$ROOT/evaluations/final_test250_v1/dataset"
WORKBOOK_SHA256=884601f97be529420529c87798bcfedca63c90d0b516da525c42806bd03e38b6

test -r "$WORKBOOK"
test -d "$IMAGE_ROOT"
test -r "$REPO/scripts/build_final_test250_dataset.py"
if [[ -e "$OUTPUT" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""

python "$REPO/scripts/build_final_test250_dataset.py" \
    --workbook "$WORKBOOK" \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$OUTPUT" \
    --expected-workbook-sha256 "$WORKBOOK_SHA256"
