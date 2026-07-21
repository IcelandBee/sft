#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
AUDIT="$ROOT/evaluations/e1_e2_dev_boundary_audit_v1"
OUT="$AUDIT/adjudication-analysis-v1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/analyze_e1_e2_adjudication.py" \
    --dev "$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl" \
    --e1-parsed "$ROOT/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl" \
    --e2-parsed "$ROOT/evaluations/e2_dev_v1/e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl" \
    --review "$AUDIT/review.jsonl" \
    --annotations "$AUDIT/annotations.json" \
    --output-dir "$OUT"
