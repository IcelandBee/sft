#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
TRAIN="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"
OUT_DIR="$ROOT/evaluations/e4_crop_aux_v1/bbox_audit_v1"
OUT="$OUT_DIR/summary.json"

command -v python >/dev/null
test -r "$TRAIN"
test -r "$REPO/scripts/audit_e4_bbox_coverage.py"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: output directory already exists: $OUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
python "$REPO/scripts/audit_e4_bbox_coverage.py" \
    --train "$TRAIN" \
    --output "$OUT" \
    --expected-rows 8026 \
    --expected-good 6074 \
    --expected-bad 1952 \
    --small-area-threshold 0.01 \
    --max-crops-per-bad-image 2 \
    --t1-rows 9978 \
    --local-share 0.40 \
    2>&1 | tee "$OUT_DIR/audit.log"
