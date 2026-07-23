#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
SOURCE="$ROOT/ms_swift/e4_crop_aux_json_v1"
OUTPUT="$ROOT/ms_swift/e5_crop_aux20_json_v1"
REPO=/home/data/h30082292/code/sft

test -r "$SOURCE/train.jsonl"
test -r "$SOURCE/dev.jsonl"
test -r "$SOURCE/local_manifest.jsonl"
test -r "$SOURCE/build_summary.json"

if [[ -e "$OUTPUT" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT" >&2
    exit 1
fi

python "$REPO/scripts/build_e5_reduced_local_dataset.py" \
    --source-data "$SOURCE" \
    --output-dir "$OUTPUT" \
    --local-pairs 1247
