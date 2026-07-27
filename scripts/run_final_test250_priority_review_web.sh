#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
REPO=/home/data/h30082292/code/sft
SOURCE="$ROOT/evaluations/final_test250_v1/four_model_consensus_v1/review_priority.jsonl"
DATA_DIR="$ROOT/evaluations/final_test250_v1/priority_review73_v1"

test -r "$SOURCE" || {
    echo "ERROR: consensus review source is not readable: $SOURCE" >&2
    exit 1
}

if [[ ! -r "$DATA_DIR/review.jsonl" ]]; then
    python "$REPO/scripts/build_final_test250_priority_review.py" \
        --source "$SOURCE" \
        --output-dir "$DATA_DIR" \
        --expected-count 73 \
        --seed 20260727
fi

exec python "$REPO/scripts/final_test250_review_web.py" \
    --data-dir "$DATA_DIR" \
    --expected-count 73 \
    "$@"
