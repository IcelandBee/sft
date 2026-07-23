#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
PREFLIGHT="$ROOT/evaluations/e4_crop_aux_v1/token_preflight_v1"
MANIFEST="$PREFLIGHT/poc/manifest.json"
OUTPUT="$PREFLIGHT/crop-review-v1"

test -r "$MANIFEST"
test -r "$REPO/scripts/e4_crop_review_web.py"
test -r "$REPO/web/e4-crop-review/index.html"

exec python "$REPO/scripts/e4_crop_review_web.py" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT" \
    "$@"
