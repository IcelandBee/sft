#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
ADAPTER="$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-975"
BENCHMARK_ROOT="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1"
MERGED="$BENCHMARK_ROOT/merged-e5-975"
LOG="$BENCHMARK_ROOT/merge-e5-975.log"
SUMMARY="$BENCHMARK_ROOT/merge-summary.json"
TRAIN_ENV=/home/data/h30082292/miniconda3/envs/qwen35_27b
PYTHON="$TRAIN_ENV/bin/python"
SWIFT="$TRAIN_ENV/bin/swift"
GPU=${BENCHMARK_GPU:-2}

test -x "$PYTHON"
test -x "$SWIFT"
test -r "$MODEL/config.json"
test -r "$ADAPTER/adapter_config.json"
compgen -G "$ADAPTER/adapter_model*.safetensors" >/dev/null

if [[ -e "$MERGED" || -e "$LOG" || -e "$SUMMARY" ]]; then
    echo "ERROR: merge output already exists under: $BENCHMARK_ROOT" >&2
    exit 1
fi

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

mkdir -p "$BENCHMARK_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_NS=$(date +%s%N)
"$SWIFT" export \
    --model "$MODEL" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir "$MERGED" \
    2>&1 | tee "$LOG"
END_NS=$(date +%s%N)

test -r "$MERGED/config.json"
compgen -G "$MERGED/model*.safetensors" >/dev/null

"$PYTHON" - "$MODEL" "$ADAPTER" "$MERGED" "$SUMMARY" "$START_NS" "$END_NS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

base, adapter, merged, output = map(Path, sys.argv[1:5])
elapsed = (int(sys.argv[6]) - int(sys.argv[5])) / 1_000_000_000

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
merged_config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
weights = sorted(merged.glob("model*.safetensors"))
if not weights:
    raise SystemExit("ERROR: merged model has no safetensors weights")
if merged_config.get("model_type") != base_config.get("model_type"):
    raise SystemExit("ERROR: merged model_type differs from base")
if merged_config.get("architectures") != base_config.get("architectures"):
    raise SystemExit("ERROR: merged architectures differ from base")

summary = {
    "protocol_version": "e5_975_merge_for_vllm_v1",
    "base_model": str(base),
    "adapter": str(adapter),
    "merged_model": str(merged),
    "model_type": merged_config.get("model_type"),
    "architectures": merged_config.get("architectures"),
    "adapter_rank": adapter_config.get("r"),
    "adapter_alpha": adapter_config.get("lora_alpha"),
    "merge_seconds": elapsed,
    "weight_files": len(weights),
    "weight_bytes": sum(path.stat().st_size for path in weights),
    "base_config_sha256": sha256(base / "config.json"),
    "merged_config_sha256": sha256(merged / "config.json"),
}
output.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
print("E5_975_VLLM_MERGE: PASS")
PY
