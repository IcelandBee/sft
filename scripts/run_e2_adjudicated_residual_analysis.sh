#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
AUDIT="$ROOT/evaluations/e1_e2_dev_boundary_audit_v1"
OUT="$ROOT/evaluations/e2_dev_v1/e2_adjudicated_residual_analysis_v1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/analyze_e2_adjudicated_residuals.py" \
    --dev "$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl" \
    --original-dev "$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl" \
    --e2-parsed "$ROOT/evaluations/e2_dev_v1/e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl" \
    --annotations "$AUDIT/annotations.json" \
    --train "$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl" \
    --output-dir "$OUT"
