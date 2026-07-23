#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
LABELS="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"
T1="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
OUTPUT="$ROOT/ms_swift/e4_crop_aux_json_v1"

test -r "$LABELS"
test -r "$T1"
test -r "$DEV"
test -r "$REPO/scripts/build_e4_crop_aux_dataset.py"

if [[ -e "$OUTPUT" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT" >&2
    exit 1
fi

python "$REPO/scripts/build_e4_crop_aux_dataset.py" \
    --labels "$LABELS" \
    --t1-train "$T1" \
    --dev "$DEV" \
    --output-dir "$OUTPUT" \
    --expected-label-rows 8026 \
    --expected-t1-rows 9978 \
    --expected-dev-rows 200 \
    --expected-label-good 6074 \
    --expected-label-bad 1952 \
    --expected-t1-good 6074 \
    --expected-t1-bad 3904 \
    --expected-dev-good 142 \
    --expected-dev-bad 58 \
    --expected-dev-sha256 cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
