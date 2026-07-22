#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
OUT="$ROOT/evaluations/base_dev_v2/qwen35_27b_transformers_binary_trie_v1"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
GPU=4

command -v python >/dev/null
test -r "$MODEL/config.json"
test -r "$DEV"
test -r "$REPO/scripts/run_constrained_binary_dev.py"
test -r "$REPO/scripts/evaluate_e1_dev.py"

if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

python - <<'PY'
import swift
import transformers
print(f"swift_version={swift.__version__}")
print(f"transformers_version={transformers.__version__}")
PY

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "$OUT")"
python "$REPO/scripts/run_constrained_binary_dev.py" \
    --model "$MODEL" \
    --dev "$DEV" \
    --expected-dev-sha256 "$EXPECTED_DEV_SHA256" \
    --output-dir "$OUT" \
    2>&1 | tee "${OUT}.infer.log"

python "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/raw-result.jsonl" \
    --output-dir "$OUT/evaluation" \
    --expected-count 200 \
    --checkpoint-step 0 \
    --expected-dev "$DEV" \
    | tee "$OUT/evaluate.log"

python - "$OUT/evaluation/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("=== BASE CONSTRAINED BINARY RESULT ===")
print(f"TP={metrics['tp']} FN={metrics['fn']} FP={metrics['fp']} TN={metrics['tn']}")
print(
    f"Recall={metrics['recall']:.2%} FPR={metrics['fpr']:.2%} "
    f"Accuracy={metrics['accuracy']:.2%} F1={metrics['f1']:.2%}"
)
print(
    f"envelope_valid={metrics['envelope_valid_rate']:.2%} "
    f"payload_json_valid={metrics['payload_json_valid_rate']:.2%} "
    f"schema_valid={metrics['schema_valid_rate']:.2%}"
)
print("NOTE: schema validity is guaranteed by the token constraint, not model capability.")
print("BASE_CONSTRAINED_BINARY_DEV: PASS")
PY
