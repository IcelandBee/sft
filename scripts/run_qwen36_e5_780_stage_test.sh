#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: Qwen3.6 E5-780 stage Test failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
ADAPTER="$ROOT/runs/qwen36_e5_crop_aux20_aligner_r16_s1560_v1/v0-20260728-204136/checkpoint-780"
DEV_SUMMARY="$ROOT/evaluations/qwen36_27b/e5_dev_8ckpt_v1/checkpoint-summary.json"
DEV_COMPARISON="$ROOT/evaluations/qwen36_27b/dev_comparison_v1.json"
TEST="$ROOT/evaluations/final_test250_v1/priority_review73_readjudication_v1/test_conditionally_readjudicated.jsonl"
OUT="$ROOT/evaluations/qwen36_27b/stage_test241_e5_780_v1"
EXPECTED_TEST_SHA256=c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f
GPU=${QWEN36_TEST_GPU:-4}

[[ -x "$ENV/bin/python" ]] || { echo "ERROR: missing executable: $ENV/bin/python" >&2; exit 1; }
[[ -x "$ENV/bin/swift" ]] || { echo "ERROR: missing executable: $ENV/bin/swift" >&2; exit 1; }
[[ -r "$MODEL/config.json" ]] || { echo "ERROR: missing model config" >&2; exit 1; }
[[ -r "$ADAPTER/adapter_config.json" ]] || { echo "ERROR: missing selected adapter config" >&2; exit 1; }
compgen -G "$ADAPTER/adapter_model*.safetensors" >/dev/null || {
    echo "ERROR: missing selected adapter weights" >&2
    exit 1
}
[[ -r "$DEV_SUMMARY" ]] || { echo "ERROR: missing E5 Dev selection summary" >&2; exit 1; }
[[ -r "$DEV_COMPARISON" ]] || { echo "ERROR: missing Dev comparison" >&2; exit 1; }
[[ -r "$TEST" ]] || { echo "ERROR: missing stage Test" >&2; exit 1; }
[[ -r "$REPO/scripts/evaluate_e1_dev.py" ]] || { echo "ERROR: missing evaluator" >&2; exit 1; }
if [[ -e "$OUT" ]]; then
    echo "ERROR: immutable output already exists: $OUT" >&2
    exit 1
fi

ACTUAL_TEST_SHA256=$(sha256sum "$TEST" | awk '{print $1}')
if [[ "$ACTUAL_TEST_SHA256" != "$EXPECTED_TEST_SHA256" ]]; then
    echo "ERROR: stage Test sha256 mismatch: $ACTUAL_TEST_SHA256" >&2
    exit 1
fi

"$ENV/bin/python" - "$DEV_SUMMARY" "$DEV_COMPARISON" "$TEST" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

dev_summary_path, comparison_path, test_path = map(Path, sys.argv[1:])
dev_summary = json.loads(dev_summary_path.read_text(encoding="utf-8"))
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
if dev_summary.get("selected_step") != 780:
    raise SystemExit(
        f"ERROR: E5 Dev-selected step is {dev_summary.get('selected_step')}, expected 780"
    )
if comparison.get("recommended") != {"experiment": "E5", "checkpoint_step": 780}:
    raise SystemExit(f"ERROR: Dev recommendation drift: {comparison.get('recommended')}")
if comparison.get("test_evaluated") is not False:
    raise SystemExit("ERROR: Dev comparison does not declare test_evaluated=false")

rows = [
    json.loads(line)
    for line in test_path.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
labels = Counter(json.loads(row["messages"][-1]["content"])["decision"] for row in rows)
if len(rows) != 241 or labels != {"GOOD": 175, "BAD": 66}:
    raise SystemExit(f"ERROR: stage Test contract mismatch: rows={len(rows)} labels={dict(labels)}")
missing = [image for row in rows for image in row["images"] if not Path(image).is_file()]
if missing:
    raise SystemExit(f"ERROR: missing Test images: {len(missing)}; first={missing[0]}")
print("SELECTION_LOCK: PASS E5 checkpoint-780")
print(f"TEST_CONTRACT: PASS rows={len(rows)} labels={dict(labels)}")
PY

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

mkdir -p "$OUT"
"$ENV/bin/python" - "$OUT/protocol-manifest.json" "$MODEL" "$ADAPTER" "$TEST" \
    "$EXPECTED_TEST_SHA256" "$DEV_SUMMARY" "$DEV_COMPARISON" "$GPU" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, model, adapter, test, test_sha256, dev_summary, comparison, gpu = sys.argv[1:]
manifest = {
    "protocol_version": "qwen36_dev_selected_stage_test241_v1",
    "model": model,
    "adapter": adapter,
    "selected_experiment": "E5",
    "selected_checkpoint_step": 780,
    "selection_source": dev_summary,
    "selection_source_sha256": hashlib.sha256(Path(dev_summary).read_bytes()).hexdigest(),
    "comparison_source": comparison,
    "comparison_source_sha256": hashlib.sha256(Path(comparison).read_bytes()).hexdigest(),
    "test": test,
    "test_sha256": test_sha256,
    "test_rows": 241,
    "test_labels": {"GOOD": 175, "BAD": 66},
    "infer_backend": "transformers",
    "structured_decoding": False,
    "add_non_thinking_prefix": True,
    "temperature": 0.0,
    "max_new_tokens": 128,
    "max_batch_size": 1,
    "image_max_token_num": 1024,
    "dtype": "bfloat16",
    "attention_implementation": "flash_attention_2",
    "seed": 42,
    "physical_gpu": int(gpu),
    "test_used_for_selection": False,
    "test_may_not_be_used_for_retuning": True,
}
Path(output).write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=1024
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV/bin/swift" infer \
    --model "$MODEL" \
    --adapters "$ADAPTER" \
    --val_dataset "$TEST" \
    --split_dataset_ratio 0 \
    --dataset_shuffle false \
    --val_dataset_shuffle false \
    --strict true \
    --lazy_tokenize true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --infer_backend transformers \
    --max_new_tokens 128 \
    --temperature 0 \
    --num_beams 1 \
    --stream false \
    --max_batch_size 1 \
    --write_batch_size 20 \
    --dataset_num_proc 1 \
    --load_from_cache_file false \
    --load_args false \
    --seed 42 \
    --data_seed 42 \
    --result_path "$OUT/raw-result.jsonl" \
    2>&1 | tee "$OUT/infer.log"

"$ENV/bin/python" "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/raw-result.jsonl" \
    --output-dir "$OUT/evaluation" \
    --expected-count 241 \
    --checkpoint-step 780 \
    --expected-dev "$TEST" \
    2>&1 | tee "$OUT/evaluate.log"

"$ENV/bin/python" - "$OUT/evaluation/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("=== QWEN3.6 E5-780 STAGE TEST N=241 ===")
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
print(f"invalid_by_gold={metrics['invalid_by_gold']}")
print("QWEN36_E5_780_STAGE_TEST241: PASS")
PY
