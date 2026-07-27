#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
ADAPTER="$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-975"
BENCHMARK_ROOT="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1"

command -v python >/dev/null
command -v swift >/dev/null
command -v nvidia-smi >/dev/null
test -r "$MODEL/config.json"
test -r "$ADAPTER/adapter_config.json"
compgen -G "$ADAPTER/adapter_model*.safetensors" >/dev/null

echo "=== RUNTIME ==="
echo "python=$(command -v python)"
echo "swift=$(command -v swift)"
echo "conda_prefix=${CONDA_PREFIX:-UNSET}"
python - <<'PY'
import importlib.metadata
import json
import platform

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"

print(f"python_version={platform.python_version()}")
for package in (
    "ms-swift", "torch", "transformers", "peft", "safetensors",
    "flash-attn", "qwen-vl-utils", "vllm",
):
    print(f"{package}={version(package)}")
try:
    import torch
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu0_capability={torch.cuda.get_device_capability(0)}")
except Exception as exc:
    print(f"torch_probe_error={type(exc).__name__}:{exc}")
PY

echo
echo "=== DRIVER / GPU ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free \
    --format=csv,noheader
if command -v nvcc >/dev/null; then
    nvcc --version | tail -n 1
else
    echo "nvcc=NOT_FOUND"
fi

echo
echo "=== MODEL / ADAPTER ==="
python - "$MODEL" "$ADAPTER" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
from safetensors import safe_open

model = Path(sys.argv[1])
adapter = Path(sys.argv[2])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
adapter_config = json.loads(
    (adapter / "adapter_config.json").read_text(encoding="utf-8")
)
weights = sorted(adapter.glob("adapter_model*.safetensors"))
counts = Counter()
examples = {"tower": [], "connector": [], "language": []}
for weight in weights:
    with safe_open(weight, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            lowered = key.lower()
            if ".visual.blocks." in lowered or ".vision_tower." in lowered:
                component = "tower"
            elif ".visual.merger." in lowered or "aligner" in lowered or "connector" in lowered:
                component = "connector"
            else:
                component = "language"
            counts[component] += 1
            if len(examples[component]) < 4:
                examples[component].append(key)

print(f"model_type={config.get('model_type')}")
print(f"architectures={config.get('architectures')}")
print(f"adapter_rank={adapter_config.get('r')}")
print(f"adapter_alpha={adapter_config.get('lora_alpha')}")
print(f"adapter_tensor_counts={dict(counts)}")
for component in ("tower", "connector", "language"):
    print(f"{component}_examples={examples[component]}")
if counts["tower"] or counts["connector"]:
    print("DIRECT_VLLM_ADAPTER_MODE=UNSAFE_WITHOUT_VERIFIED_TOWER_CONNECTOR_LORA")
    print("MERGED_MODEL_VLLM_MODE=RECOMMENDED")
else:
    print("DIRECT_VLLM_ADAPTER_MODE=LANGUAGE_ONLY_CANDIDATE")
PY

echo
echo "=== DISK ==="
df -BG "$MODEL" "$ROOT" | awk 'NR == 1 || !seen[$1]++'
MODEL_GIB=$(du -sBG "$MODEL" | awk '{gsub(/G/, "", $1); print $1}')
FREE_GIB=$(df -BG --output=avail "$ROOT" | tail -n 1 | tr -dc '0-9')
REQUIRED_GIB=$((MODEL_GIB + 20))
echo "model_size_gib=$MODEL_GIB"
echo "benchmark_root=$BENCHMARK_ROOT"
echo "estimated_required_free_gib=$REQUIRED_GIB"
echo "available_gib=$FREE_GIB"
if [[ "$FREE_GIB" -ge "$REQUIRED_GIB" ]]; then
    echo "MERGE_DISK_CHECK=PASS"
else
    echo "MERGE_DISK_CHECK=FAILED"
fi

echo
echo "=== SWIFT VLLM CLI CONTRACT ==="
HELP=$(mktemp)
trap 'rm -f "$HELP"' EXIT
swift infer --help > "$HELP" 2>&1
for ARG in infer_backend vllm_tensor_parallel_size vllm_gpu_memory_utilization \
    vllm_max_lora_rank vllm_engine_kwargs adapters result_path; do
    if grep -q -- "--${ARG//_/-}\|--$ARG" "$HELP"; then
        echo "$ARG=SUPPORTED"
    else
        echo "$ARG=MISSING"
    fi
done

echo
echo "VLLM_LORA_BENCHMARK_PREFLIGHT: PASS"
