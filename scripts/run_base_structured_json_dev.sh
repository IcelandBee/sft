#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
OUT="$ROOT/evaluations/base_dev_v2/qwen35_27b_structured_json_v1"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
GPU=4

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$DEV"
test -r "$REPO/scripts/structured_json_protocol.py"
test -r "$REPO/scripts/evaluate_e1_dev.py"

if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

python - "$DEV" "$EXPECTED_DEV_SHA256" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

dev_path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
source = dev_path.read_bytes()
actual_sha256 = hashlib.sha256(source).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(f"ERROR: corrected Dev sha256 mismatch: {actual_sha256}")

rows = [json.loads(line) for line in source.decode("utf-8-sig").splitlines() if line.strip()]
decisions = Counter(json.loads(row["messages"][-1]["content"])["decision"] for row in rows)
if len(rows) != 200 or decisions != {"GOOD": 142, "BAD": 58}:
    raise SystemExit(f"ERROR: corrected Dev contract mismatch: rows={len(rows)} labels={dict(decisions)}")
if find_spec("vllm") is None:
    raise SystemExit("ERROR: vllm is not installed in the active environment")
print(f"DEV_CHECK: PASS rows={len(rows)} labels={dict(decisions)} sha256={actual_sha256}")
PY

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

REGEX=$(PYTHONPATH="$REPO" python "$REPO/scripts/structured_json_protocol.py")
mkdir -p "$OUT"

PYTHONPATH="$REPO" python - "$OUT/protocol-manifest.json" "$MODEL" "$DEV" "$EXPECTED_DEV_SHA256" "$REGEX" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, model, dev, dev_sha256, regex = sys.argv[1:]
manifest = {
    "protocol_version": "base_structured_json_v1",
    "model": model,
    "adapter": None,
    "dev": dev,
    "dev_sha256": dev_sha256,
    "infer_backend": "vllm",
    "structured_outputs_regex": regex,
    "add_non_thinking_prefix": True,
    "temperature": 0.0,
    "max_new_tokens": 128,
    "seed": 42,
    "gpu": 4,
    "test_untouched": True,
}
Path(manifest_path).write_text(
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

export CUDA_VISIBLE_DEVICES="$GPU"
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

swift infer \
    --model "$MODEL" \
    --val_dataset "$DEV" \
    --split_dataset_ratio 0 \
    --dataset_shuffle false \
    --val_dataset_shuffle false \
    --strict true \
    --lazy_tokenize true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_max_num_seqs 10 \
    --structured_outputs_regex "$REGEX" \
    --max_new_tokens 128 \
    --temperature 0 \
    --num_beams 1 \
    --stream false \
    --max_batch_size 10 \
    --write_batch_size 20 \
    --dataset_num_proc 1 \
    --load_from_cache_file false \
    --seed 42 \
    --data_seed 42 \
    --result_path "$OUT/raw-result.jsonl" \
    2>&1 | tee "$OUT/infer.log"

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
print("=== BASE STRUCTURED JSON RESULT ===")
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
print("BASE_STRUCTURED_JSON_DEV: PASS")
PY
