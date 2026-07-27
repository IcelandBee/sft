#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/home/data/h30082292/miniconda3/envs/vllm_qwen35
PYTHON="$ENV_ROOT/bin/python"

test -x "$PYTHON" || {
    echo "ERROR: environment Python is missing: $PYTHON" >&2
    exit 1
}

echo "=== BEFORE ==="
"$PYTHON" - <<'PY'
import importlib.metadata

for package in ("vllm", "ms-swift", "torch", "transformers", "peft", "qwen-vl-utils"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "NOT_INSTALLED"
    print(f"{package}={version}")
PY

echo
echo "=== INSTALL PROTOCOL DEPENDENCIES ==="
"$PYTHON" -m pip install \
    --upgrade-strategy only-if-needed \
    "ms-swift==4.4.1" \
    "peft==0.19.1" \
    "qwen-vl-utils==0.0.14"

echo
echo "=== AFTER ==="
"$PYTHON" - <<'PY'
import importlib.metadata
import json

import torch
import vllm
import swift
import peft

versions = {
    package: importlib.metadata.version(package)
    for package in (
        "vllm", "ms-swift", "torch", "transformers", "tokenizers",
        "peft", "safetensors", "qwen-vl-utils", "triton",
    )
}
print("versions=" + json.dumps(versions, ensure_ascii=False, sort_keys=True))
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"vllm_import={vllm.__version__}")
print(f"swift_import={swift.__version__}")
print(f"peft_import={peft.__version__}")
PY

"$PYTHON" -m pip check
test -x "$ENV_ROOT/bin/swift"

echo
echo "VLLM_QWEN35_ENV_PREPARE: PASS"
