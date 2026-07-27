#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
REPO=/home/data/h30082292/code/sft
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
TEST="$ROOT/evaluations/final_test250_v1/priority_review73_readjudication_v1/test_conditionally_readjudicated.jsonl"
OUT="$ROOT/evaluations/final_test250_v1/dev_selected_ensemble_v1"
DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
TEST_SHA256=c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f

DEV_E1="$ROOT/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
DEV_E2="$ROOT/evaluations/e2_dev_v1/e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
DEV_E5R="$ROOT/evaluations/e5_dev_v1/e5_crop_aux20_aligner_8ckpt_v1/checkpoint-780/evaluation/parsed.jsonl"
DEV_E5B="$ROOT/evaluations/e5_dev_v1/e5_crop_aux20_aligner_8ckpt_v1/checkpoint-975/evaluation/parsed.jsonl"

TEST_ROOT="$ROOT/evaluations/final_test250_v1/lora_checkpoints_v1"
TEST_E1="$TEST_ROOT/e1-1248/evaluation/parsed.jsonl"
TEST_E2="$TEST_ROOT/e2-1248/evaluation/parsed.jsonl"
TEST_E5R="$TEST_ROOT/e5-780-recall/evaluation/parsed.jsonl"
TEST_E5B="$TEST_ROOT/e5-975-balanced/evaluation/parsed.jsonl"

for path in "$DEV" "$TEST" "$DEV_E1" "$DEV_E2" "$DEV_E5R" "$DEV_E5B" \
    "$TEST_E1" "$TEST_E2" "$TEST_E5R" "$TEST_E5B"; do
    test -r "$path" || { echo "ERROR: required input is not readable: $path" >&2; exit 1; }
done

python "$REPO/scripts/analyze_dev_selected_ensemble.py" \
    --dev "$DEV" \
    --expected-dev-sha256 "$DEV_SHA256" \
    --dev-prediction "e1-1248=$DEV_E1" \
    --dev-prediction "e2-1248=$DEV_E2" \
    --dev-prediction "e5-780-recall=$DEV_E5R" \
    --dev-prediction "e5-975-balanced=$DEV_E5B" \
    --test "$TEST" \
    --expected-test-sha256 "$TEST_SHA256" \
    --test-prediction "e1-1248=$TEST_E1" \
    --test-prediction "e2-1248=$TEST_E2" \
    --test-prediction "e5-780-recall=$TEST_E5R" \
    --test-prediction "e5-975-balanced=$TEST_E5B" \
    --output-dir "$OUT"
