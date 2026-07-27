#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MERGED="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1/merged-e5-975"
TEST="$ROOT/evaluations/final_test250_v1/priority_review73_readjudication_v1/test_conditionally_readjudicated.jsonl"
REFERENCE_PARSED="$ROOT/evaluations/final_test250_v1/lora_checkpoints_v1/e5-975-balanced/evaluation/parsed.jsonl"
OUT="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1/comparison-v1"
TRAIN_PYTHON=/home/data/h30082292/miniconda3/envs/qwen35_27b/bin/python
VLLM_PYTHON=/home/data/h30082292/miniconda3/envs/vllm_qwen35/bin/python
EXPECTED_SHA256=c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f
GPU=${BENCHMARK_GPU:-2}

test -x "$TRAIN_PYTHON"
test -x "$VLLM_PYTHON"
test -r "$MERGED/config.json"
compgen -G "$MERGED/model*.safetensors" >/dev/null
test -r "$TEST"
test -r "$REFERENCE_PARSED"

if [[ -e "$OUT" ]]; then
    echo "ERROR: benchmark output already exists: $OUT" >&2
    exit 1
fi

ACTUAL_SHA256=$(sha256sum "$TEST" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: Test N=241 sha256 mismatch: $ACTUAL_SHA256" >&2
    exit 1
fi

"$VLLM_PYTHON" - <<'PY'
import torch
import vllm
print(f"VLLM_RUNTIME: vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda}")
PY

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

mkdir -p "$OUT/transformers" "$OUT/vllm"
"$TRAIN_PYTHON" "$REPO/scripts/build_vllm_benchmark_requests.py" \
    --dataset "$TEST" \
    --output "$OUT/requests.jsonl" \
    --manifest "$OUT/request-manifest.json"

echo "=== TRANSFORMERS BENCHMARK ON GPU $GPU ==="
(
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$TRAIN_PYTHON" "$REPO/scripts/run_transformers_merged_benchmark.py" \
        --model "$MERGED" \
        --requests "$OUT/requests.jsonl" \
        --result "$OUT/transformers/raw-result.jsonl" \
        --stats "$OUT/transformers/stats.json"
) 2>&1 | tee "$OUT/transformers/infer.log"

"$TRAIN_PYTHON" "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/transformers/raw-result.jsonl" \
    --output-dir "$OUT/transformers/evaluation" \
    --expected-count 241 \
    --checkpoint-step 975 \
    --expected-dev "$TEST" \
    | tee "$OUT/transformers/evaluate.log"

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free before vLLM: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU memory was not released after Transformers" >&2
    exit 1
fi

echo "=== VLLM BENCHMARK ON GPU $GPU ==="
(
    export CUDA_VISIBLE_DEVICES="$GPU"
    export VLLM_USE_V1=1
    "$VLLM_PYTHON" "$REPO/scripts/run_vllm_merged_benchmark.py" \
        --model "$MERGED" \
        --requests "$OUT/requests.jsonl" \
        --result "$OUT/vllm/raw-result.jsonl" \
        --stats "$OUT/vllm/stats.json" \
        --gpu-memory-utilization 0.84 \
        --max-num-seqs 8
) 2>&1 | tee "$OUT/vllm/infer.log"

"$TRAIN_PYTHON" "$REPO/scripts/evaluate_e1_dev.py" \
    --result "$OUT/vllm/raw-result.jsonl" \
    --output-dir "$OUT/vllm/evaluation" \
    --expected-count 241 \
    --checkpoint-step 975 \
    --expected-dev "$TEST" \
    | tee "$OUT/vllm/evaluate.log"

"$TRAIN_PYTHON" "$REPO/scripts/summarize_vllm_lora_benchmark.py" \
    --root "$OUT" \
    --reference-parsed "$REFERENCE_PARSED"
