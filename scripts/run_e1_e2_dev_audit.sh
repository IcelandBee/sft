#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
DEV="$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl"
E1="$ROOT/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
E2="$ROOT/evaluations/e2_dev_v1/e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248/evaluation/parsed.jsonl"
OUT="$ROOT/evaluations/e1_e2_dev_boundary_audit_v1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

test -r "$DEV"
test -r "$E1"
test -r "$E2"

python "$SCRIPT_DIR/build_e1_e2_dev_audit.py" \
    --dev "$DEV" \
    --e1-parsed "$E1" \
    --e2-parsed "$E2" \
    --output-dir "$OUT"

python - "$OUT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "both_correct": 143,
    "both_wrong": 37,
    "e1_only_correct": 10,
    "e2_only_correct": 10,
    "decision_disagreements": 20,
    "review_total": 57,
}
actual = {key: summary.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"AUDIT_COUNT_CHECK: FAILED expected={expected} actual={actual}")
print("AUDIT_COUNT_CHECK: PASS")
PY

echo "Review CSV: $OUT/review.csv"
