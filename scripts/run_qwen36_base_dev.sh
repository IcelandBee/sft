#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: Qwen3.6 Base Dev failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
OUT="$ROOT/evaluations/qwen36_27b/base_dev_natural_v1"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
GPU=4

require_executable() {
    [[ -x "$1" ]] || { echo "ERROR: missing executable: $1" >&2; exit 1; }
}

require_readable() {
    [[ -r "$1" ]] || { echo "ERROR: missing or unreadable file: $1" >&2; exit 1; }
}

require_executable "$ENV/bin/python"
require_executable "$ENV/bin/swift"
require_readable "$MODEL/config.json"
require_readable "$DEV"
require_readable "$REPO/scripts/evaluate_e1_dev.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

ACTUAL_DEV_SHA256=$(sha256sum "$DEV" | awk '{print $1}')
if [[ "$ACTUAL_DEV_SHA256" != "$EXPECTED_DEV_SHA256" ]]; then
    echo "ERROR: corrected Dev sha256 mismatch: $ACTUAL_DEV_SHA256" >&2
    exit 1
fi

"$ENV/bin/python" - "$DEV" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
decisions = Counter(json.loads(row["messages"][-1]["content"])["decision"] for row in rows)
missing = [image for row in rows for image in row["images"] if not Path(image).is_file()]
if len(rows) != 200 or decisions != {"GOOD": 142, "BAD": 58}:
    raise SystemExit(f"ERROR: Dev contract mismatch: rows={len(rows)} labels={dict(decisions)}")
if missing:
    raise SystemExit(f"ERROR: missing Dev images: {len(missing)}; first={missing[0]}")
print(f"DEV_CHECK: PASS rows={len(rows)} labels={dict(decisions)}")
PY

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

mkdir -p "$OUT"
"$ENV/bin/python" - "$OUT/protocol-manifest.json" "$MODEL" "$DEV" "$EXPECTED_DEV_SHA256" "$GPU" <<'PY'
import json
import sys
from pathlib import Path

output, model, dev, dev_sha256, gpu = sys.argv[1:]
manifest = {
    "protocol_version": "qwen36_base_dev_natural_generation_v1",
    "model": model,
    "adapter": None,
    "dev": dev,
    "dev_sha256": dev_sha256,
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
    "checkpoint_selection": False,
    "test_untouched": True,
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
    --val_dataset "$DEV" \
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
    --seed 42 \
    --data_seed 42 \
    --result_path "$OUT/raw-result.jsonl" \
    2>&1 | tee "$OUT/infer.log"

"$ENV/bin/python" "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/raw-result.jsonl" \
    --output-dir "$OUT/evaluation" \
    --expected-count 200 \
    --checkpoint-step 0 \
    --expected-dev "$DEV" \
    2>&1 | tee "$OUT/evaluate.log"

"$ENV/bin/python" - "$OUT/evaluation/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("=== QWEN3.6 BASE FIXED DEV ===")
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
print("QWEN36_BASE_DEV_NATURAL: PASS")
PY
