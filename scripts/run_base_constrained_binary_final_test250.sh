#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
BASE="$ROOT/evaluations/final_test250_v1"
DATA="$BASE/priority_review73_readjudication_v1/test_conditionally_readjudicated.jsonl"
READJUDICATION="$BASE/priority_review73_readjudication_v1/summary.json"
OUT="$BASE/base_constrained_binary_n241_v1"
EXPECTED_SHA256=c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f
GPU=4

command -v python >/dev/null
test -r "$MODEL/config.json"
test -r "$DATA"
test -r "$READJUDICATION"
test -r "$REPO/scripts/run_constrained_binary_dev.py"
test -r "$REPO/scripts/evaluate_e1_dev.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

ACTUAL_SHA256=$(sha256sum "$DATA" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: conditionally readjudicated Test sha256 mismatch: $ACTUAL_SHA256" >&2
    exit 1
fi

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
    --dev "$DATA" \
    --expected-dev-sha256 "$EXPECTED_SHA256" \
    --expected-count 241 \
    --expected-good 175 \
    --expected-bad 66 \
    --dataset-status conditionally_readjudicated_priority73_v1 \
    --no-test-untouched \
    --output-dir "$OUT" \
    2>&1 | tee "${OUT}.infer.log"

python "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/raw-result.jsonl" \
    --output-dir "$OUT/evaluation" \
    --expected-count 241 \
    --checkpoint-step 0 \
    --expected-dev "$DATA" \
    | tee "$OUT/evaluate.log"

python - "$OUT/evaluation/metrics.json" "$READJUDICATION" "$OUT/final-model-comparison.json" <<'PY'
import json
import sys
from pathlib import Path

base_path, readjudication_path, output_path = map(Path, sys.argv[1:])
base = json.loads(base_path.read_text(encoding="utf-8"))
readjudication = json.loads(readjudication_path.read_text(encoding="utf-8"))
rows = [
    {
        "candidate": "base-constrained-binary",
        "protocol": "transformers_binary_token_trie_v1",
        **{key: base[key] for key in (
            "total", "tp", "fn", "fp", "tn", "recall", "fpr",
            "accuracy", "precision", "f1", "schema_valid_rate"
        )},
    }
]
for name, result in readjudication["model_results"].items():
    metrics = result["adjusted_excluding_unsure"]
    rows.append({
        "candidate": name,
        "protocol": "free_generation_strict_schema_v1",
        **metrics,
        "schema_valid_rate": 1.0,
    })
comparison = {
    "protocol_version": "final_test250_n241_base_lora_comparison_v1",
    "dataset_sha256": readjudication["adjusted_test_sha256"],
    "dataset_status": "conditionally_readjudicated_priority73_v1",
    "total": 241,
    "checkpoint_selection_forbidden": True,
    "base_schema_validity": "guaranteed_by_token_constraint_not_model_capability",
    "decision_metrics_comparable": True,
    "generation_protocols_identical": False,
    "candidates": rows,
}
output_path.write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("=== FINAL TEST250 N=241 BASE / LORA COMPARISON ===")
for row in rows:
    print(
        f"{row['candidate']} TP={row['tp']} FN={row['fn']} FP={row['fp']} TN={row['tn']} "
        f"Recall={row['recall']:.2%} FPR={row['fpr']:.2%} "
        f"Accuracy={row['accuracy']:.2%} F1={row['f1']:.2%}"
    )
print(f"comparison={output_path}")
print("NOTE: Base schema validity is guaranteed by token constraint, not model capability.")
print("BASE_CONSTRAINED_BINARY_FINAL_TEST250: PASS")
PY
