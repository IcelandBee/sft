#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
REPO=/home/data/h30082292/code/sft
TEST="$ROOT/evaluations/final_test250_v1/dataset/test.jsonl"
RESULTS="$ROOT/evaluations/final_test250_v1/lora_checkpoints_v1"
OUTPUT="$ROOT/evaluations/final_test250_v1/four_model_consensus_v1"
EXPECTED_TEST_SHA256=860e62ee2326b3f96b524e3e982e912d9f41f35042735b39744fd7a08f85649f

declare -A PARSED=(
    [e1-1248]="$RESULTS/e1-1248/evaluation/parsed.jsonl"
    [e2-1248]="$RESULTS/e2-1248/evaluation/parsed.jsonl"
    [e5-780-recall]="$RESULTS/e5-780-recall/evaluation/parsed.jsonl"
    [e5-975-balanced]="$RESULTS/e5-975-balanced/evaluation/parsed.jsonl"
)

test -r "$TEST" || {
    echo "ERROR: Test250 dataset is not readable: $TEST" >&2
    exit 1
}
for NAME in e1-1248 e2-1248 e5-780-recall e5-975-balanced; do
    test -r "${PARSED[$NAME]}" || {
        echo "ERROR: parsed result is not readable: ${PARSED[$NAME]}" >&2
        exit 1
    }
done

ACTUAL_TEST_SHA256=$(sha256sum "$TEST" | awk '{print $1}')
if [[ "$ACTUAL_TEST_SHA256" != "$EXPECTED_TEST_SHA256" ]]; then
    echo "ERROR: Test250 sha256 mismatch: $ACTUAL_TEST_SHA256" >&2
    exit 1
fi

python "$REPO/scripts/analyze_final_test250_consensus.py" \
    --test "$TEST" \
    --prediction "e1-1248=${PARSED[e1-1248]}" \
    --prediction "e2-1248=${PARSED[e2-1248]}" \
    --prediction "e5-780-recall=${PARSED[e5-780-recall]}" \
    --prediction "e5-975-balanced=${PARSED[e5-975-balanced]}" \
    --output-dir "$OUTPUT" \
    --expected-count 250
