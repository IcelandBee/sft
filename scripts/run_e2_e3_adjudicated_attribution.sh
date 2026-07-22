#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
REPO=/home/data/h30082292/code/sft
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
ORIGINAL_DEV="$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl"
E2="$ROOT/evaluations/e2_dev_v1/e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
E3="$ROOT/evaluations/e3_dev_v1/e3_vit_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
ANNOTATIONS="$ROOT/evaluations/e1_e2_dev_boundary_audit_v1/annotations.json"
OUTPUT="$ROOT/evaluations/e3_dev_v1/e2_e3_cp1248_attribution_v1"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb

for path in "$DEV" "$ORIGINAL_DEV" "$E2" "$E3" "$ANNOTATIONS"; do
    test -r "$path" || {
        echo "ERROR: required input is not readable: $path" >&2
        exit 1
    }
done

ACTUAL_DEV_SHA256=$(sha256sum "$DEV" | awk '{print $1}')
if [[ "$ACTUAL_DEV_SHA256" != "$EXPECTED_DEV_SHA256" ]]; then
    echo "ERROR: corrected Dev sha256 mismatch: $ACTUAL_DEV_SHA256" >&2
    exit 1
fi

python "$REPO/scripts/analyze_e2_e3_adjudicated.py" \
    --dev "$DEV" \
    --original-dev "$ORIGINAL_DEV" \
    --e2-parsed "$E2" \
    --e3-parsed "$E3" \
    --annotations "$ANNOTATIONS" \
    --output-dir "$OUTPUT"
