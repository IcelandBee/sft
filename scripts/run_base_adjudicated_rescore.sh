#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
OUT="$ROOT/evaluations/base_dev_v1/qwen35_27b_adjudicated_rescore_v1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/rescore_base_adjudicated_dev.py" \
    --original-dev "$ROOT/ms_swift/e1_broad_clean_json_v1/dev.jsonl" \
    --corrected-dev "$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl" \
    --result "$ROOT/evaluations/base_dev_v1/qwen35_27b_fixed_dev_v1/raw-result.jsonl" \
    --output-dir "$OUT"
