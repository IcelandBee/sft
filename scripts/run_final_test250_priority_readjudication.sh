#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
REPO=/home/data/h30082292/code/sft
BASE="$ROOT/evaluations/final_test250_v1"
TEST="$BASE/dataset/test.jsonl"
REVIEW="$BASE/priority_review73_v1/review.jsonl"
ANNOTATIONS="$BASE/priority_review73_v1/annotations.json"
CONSENSUS="$BASE/four_model_consensus_v1/review_priority.jsonl"
RESULTS="$BASE/lora_checkpoints_v1"
OUTPUT="$BASE/priority_review73_readjudication_v1"
EXPECTED_TEST_SHA256=860e62ee2326b3f96b524e3e982e912d9f41f35042735b39744fd7a08f85649f

declare -A PARSED=(
    [e1-1248]="$RESULTS/e1-1248/evaluation/parsed.jsonl"
    [e2-1248]="$RESULTS/e2-1248/evaluation/parsed.jsonl"
    [e5-780-recall]="$RESULTS/e5-780-recall/evaluation/parsed.jsonl"
    [e5-975-balanced]="$RESULTS/e5-975-balanced/evaluation/parsed.jsonl"
)

for path in "$TEST" "$REVIEW" "$ANNOTATIONS" "$CONSENSUS"; do
    test -r "$path" || { echo "ERROR: required input is not readable: $path" >&2; exit 1; }
done
for NAME in e1-1248 e2-1248 e5-780-recall e5-975-balanced; do
    test -r "${PARSED[$NAME]}" || { echo "ERROR: parsed result is not readable: ${PARSED[$NAME]}" >&2; exit 1; }
done

ACTUAL_TEST_SHA256=$(sha256sum "$TEST" | awk '{print $1}')
if [[ "$ACTUAL_TEST_SHA256" != "$EXPECTED_TEST_SHA256" ]]; then
    echo "ERROR: Test250 sha256 mismatch: $ACTUAL_TEST_SHA256" >&2
    exit 1
fi

python "$REPO/scripts/analyze_final_test250_priority_readjudication.py" \
    --test "$TEST" \
    --review "$REVIEW" \
    --consensus "$CONSENSUS" \
    --annotations "$ANNOTATIONS" \
    --prediction "e1-1248=${PARSED[e1-1248]}" \
    --prediction "e2-1248=${PARSED[e2-1248]}" \
    --prediction "e5-780-recall=${PARSED[e5-780-recall]}" \
    --prediction "e5-975-balanced=${PARSED[e5-975-balanced]}" \
    --expected-confusion "e1-1248=33,31,27,159" \
    --expected-confusion "e2-1248=28,36,28,158" \
    --expected-confusion "e5-780-recall=38,26,52,134" \
    --expected-confusion "e5-975-balanced=32,32,34,152" \
    --output-dir "$OUTPUT"
