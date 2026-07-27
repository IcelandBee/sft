#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/home/data/h30082292/miniconda3/envs/vllm_qwen35
PYTHON="$ENV_ROOT/bin/python"
SWIFT="$ENV_ROOT/bin/swift"

test -x "$PYTHON" || {
    echo "ERROR: environment Python is missing: $PYTHON" >&2
    exit 1
}

echo "=== VLLM_QWEN35 ENVIRONMENT ==="
echo "env_root=$ENV_ROOT"
echo "python=$PYTHON"
if [[ -x "$SWIFT" ]]; then
    echo "swift=$SWIFT"
else
    echo "swift=NOT_INSTALLED"
fi

"$PYTHON" - <<'PY'
import importlib.metadata
import json
import platform

packages = (
    "vllm", "ms-swift", "torch", "transformers", "tokenizers", "peft",
    "safetensors", "flash-attn", "qwen-vl-utils", "triton", "xformers",
)

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"

print(f"python_version={platform.python_version()}")
for package in packages:
    print(f"{package}={version(package)}")

checks = {}
try:
    import torch
    checks["torch_import"] = "PASS"
    checks["torch_cuda"] = torch.version.cuda
    checks["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        checks["gpu0_name"] = torch.cuda.get_device_name(0)
        checks["gpu0_capability"] = list(torch.cuda.get_device_capability(0))
except Exception as exc:
    checks["torch_import"] = f"FAILED:{type(exc).__name__}:{exc}"

try:
    import vllm
    from vllm import LLM  # noqa: F401
    checks["vllm_import"] = "PASS"
    checks["vllm_version"] = vllm.__version__
except Exception as exc:
    checks["vllm_import"] = f"FAILED:{type(exc).__name__}:{exc}"

try:
    import swift
    checks["swift_import"] = "PASS"
    checks["swift_version"] = swift.__version__
except Exception as exc:
    checks["swift_import"] = f"FAILED:{type(exc).__name__}:{exc}"

print("import_checks=" + json.dumps(checks, ensure_ascii=False, sort_keys=True))
PY

echo
echo "=== PACKAGE CONSISTENCY ==="
if "$PYTHON" -m pip check; then
    echo "PIP_CHECK=PASS"
else
    echo "PIP_CHECK=FAILED"
fi

echo
echo "=== CLI CONTRACT ==="
if [[ -x "$SWIFT" ]]; then
    HELP=$(mktemp)
    trap 'rm -f "$HELP"' EXIT
    "$SWIFT" infer --help > "$HELP" 2>&1
    for ARG in infer_backend adapters vllm_tensor_parallel_size \
        vllm_gpu_memory_utilization vllm_engine_kwargs result_path; do
        if grep -q -- "--${ARG//_/-}\|--$ARG" "$HELP"; then
            echo "$ARG=SUPPORTED"
        else
            echo "$ARG=MISSING"
        fi
    done
else
    echo "SWIFT_CLI_CHECK=SKIPPED"
fi

echo
echo "=== GPU SNAPSHOT ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader

echo
echo "VLLM_QWEN35_ENV_CHECK: PASS"
