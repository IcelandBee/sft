#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
TEST="$ROOT/evaluations/final_test250_v1/dataset/test.jsonl"
BUILD_SUMMARY="$ROOT/evaluations/final_test250_v1/dataset/build_summary.json"
OUT="$ROOT/evaluations/final_test250_v1/lora_checkpoints_v1"

E1="$ROOT/runs/e1_broad_clean_r16_e4_v1/v0-20260717-185936/checkpoint-1248"
E2="$ROOT/runs/e2_broad_clean_aligner_r16_e2_v1/v0-20260721-114449/checkpoint-1248"
E5R="$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-780"
E5B="$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-975"

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$TEST"
test -r "$BUILD_SUMMARY"
test -r "$REPO/scripts/evaluate_e1_dev.py"
for ADAPTER in "$E1" "$E2" "$E5R" "$E5B"; do
    test -r "$ADAPTER/adapter_config.json"
    compgen -G "$ADAPTER/adapter_model*.safetensors" >/dev/null
done
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

python - "$TEST" "$BUILD_SUMMARY" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

test_path, summary_path = map(Path, sys.argv[1:])
source = test_path.read_bytes()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
decisions = Counter(json.loads(row["messages"][-1]["content"])["decision"] for row in rows)
if len(rows) != 250 or decisions != {"GOOD": 186, "BAD": 64}:
    raise ValueError(f"unexpected Test contract: rows={len(rows)} labels={dict(decisions)}")
digest = hashlib.sha256(source).hexdigest()
if digest != summary.get("test_jsonl_sha256"):
    raise ValueError("Test JSONL sha256 differs from build summary")
if not summary.get("checkpoint_selection_forbidden"):
    raise ValueError("Test selection prohibition is not frozen")
print(f"TEST_CHECK: PASS rows=250 labels={dict(decisions)} sha256={digest}")
PY

for GPU in 4 5 6 7; do
    FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    echo "GPU $GPU free: ${FREE} MiB"
    if [[ "$FREE" -lt 70000 ]]; then
        echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
        exit 1
    fi
done

mkdir -p "$OUT"
python - "$OUT/run-manifest.json" "$TEST" "$E1" "$E2" "$E5R" "$E5B" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = {
    "protocol_version": "final_test250_lora_checkpoints_v1",
    "test": sys.argv[2],
    "checkpoint_selection_forbidden": True,
    "jobs": [
        {"name": "e1-1248", "gpu": 4, "adapter": sys.argv[3]},
        {"name": "e2-1248", "gpu": 5, "adapter": sys.argv[4]},
        {"name": "e5-780-recall", "gpu": 6, "adapter": sys.argv[5]},
        {"name": "e5-975-balanced", "gpu": 7, "adapter": sys.argv[6]},
    ],
}
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_job() {
    local NAME=$1
    local GPU=$2
    local ADAPTER=$3
    local STEP=$4
    local JOB="$OUT/$NAME"
    mkdir "$JOB"
    echo "LAUNCH $NAME on GPU $GPU"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        export IMAGE_MAX_TOKEN_NUM=1024
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        swift infer \
            --model "$MODEL" \
            --adapters "$ADAPTER" \
            --val_dataset "$TEST" \
            --val_dataset_shuffle false \
            --strict true \
            --lazy_tokenize true \
            --add_non_thinking_prefix true \
            --torch_dtype bfloat16 \
            --attn_impl flash_attention_2 \
            --infer_backend transformers \
            --max_new_tokens 128 \
            --temperature 0 \
            --stream false \
            --max_batch_size 1 \
            --write_batch_size 20 \
            --dataset_num_proc 1 \
            --seed 42 \
            --data_seed 42 \
            --result_path "$JOB/raw-result.jsonl" \
            > "$JOB/infer.log" 2>&1
        python "$REPO/scripts/evaluate_e1_dev.py" \
            --result "$JOB/raw-result.jsonl" \
            --output-dir "$JOB/evaluation" \
            --expected-count 250 \
            --checkpoint-step "$STEP" \
            --expected-dev "$TEST" \
            > "$JOB/evaluate.log" 2>&1
    )
    echo "COMPLETE $NAME on GPU $GPU"
}

run_job e1-1248 4 "$E1" 1248 & P1=$!
run_job e2-1248 5 "$E2" 1248 & P2=$!
run_job e5-780-recall 6 "$E5R" 780 & P3=$!
run_job e5-975-balanced 7 "$E5B" 975 & P4=$!

FAILED=0
for PID in "$P1" "$P2" "$P3" "$P4"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done
if [[ "$FAILED" -ne 0 ]]; then
    echo "ERROR: one or more final Test jobs failed; inspect per-job logs" >&2
    exit 1
fi

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = ["e1-1248", "e2-1248", "e5-780-recall", "e5-975-balanced"]
rows = []
for name in names:
    metrics = json.loads((root / name / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    rows.append({"candidate": name, **metrics})
summary = {
    "protocol_version": "final_test250_lora_summary_v1",
    "checkpoint_selection_forbidden": True,
    "candidates": rows,
}
(root / "final-test-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("=== FINAL TEST250 LORA RESULTS ===")
for row in rows:
    print(
        f"{row['candidate']} TP={row['tp']} FN={row['fn']} FP={row['fp']} TN={row['tn']} "
        f"Recall={row['recall']:.2%} FPR={row['fpr']:.2%} "
        f"Accuracy={row['accuracy']:.2%} F1={row['f1']:.2%} "
        f"Schema={row['schema_valid_rate']:.2%}"
    )
print(f"summary={root / 'final-test-summary.json'}")
print("FINAL_TEST250_LORA_EVALUATION: PASS")
PY
