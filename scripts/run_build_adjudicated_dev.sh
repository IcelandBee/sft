#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
AUDIT="$ROOT/evaluations/e1_e2_dev_boundary_audit_v1"
OUT="$ROOT/ms_swift/dev_adjudicated_v1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/build_adjudicated_dev.py" \
    --dev "$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl" \
    --review "$AUDIT/review.jsonl" \
    --annotations "$AUDIT/annotations.json" \
    --output-dir "$OUT" \
    --expected-count 200 \
    --expected-review 57 \
    --expected-changes 27
