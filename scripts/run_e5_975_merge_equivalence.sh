#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
ADAPTER="$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-975"
MERGED="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1/merged-e5-975"
TEST="$ROOT/evaluations/final_test250_v1/priority_review73_readjudication_v1/test_conditionally_readjudicated.jsonl"
OUT="$ROOT/evaluations/final_test250_v1/vllm_benchmark_e5_975_v1/merge-equivalence-swift-pt-v1"
EXPECTED_SHA256=c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f
TRAIN_ENV=/home/data/h30082292/miniconda3/envs/qwen35_27b
PYTHON="$TRAIN_ENV/bin/python"
SWIFT="$TRAIN_ENV/bin/swift"
GPU=${BENCHMARK_GPU:-2}

test -x "$PYTHON"
test -x "$SWIFT"
test -r "$MODEL/config.json"
test -r "$ADAPTER/adapter_config.json"
compgen -G "$ADAPTER/adapter_model*.safetensors" >/dev/null
test -r "$MERGED/config.json"
compgen -G "$MERGED/model*.safetensors" >/dev/null
test -r "$TEST"
test -r "$REPO/scripts/evaluate_e1_dev.py"
test -r "$REPO/scripts/analyze_e5_975_merge_equivalence.py"

if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

ACTUAL_SHA256=$(sha256sum "$TEST" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: Test N=241 sha256 mismatch: $ACTUAL_SHA256" >&2
    exit 1
fi

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi

mkdir -p "$OUT/adapter" "$OUT/merged"

run_infer() {
    local NAME=$1
    shift
    local JOB="$OUT/$NAME"
    echo "=== RUN $NAME ON GPU $GPU ==="
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        export IMAGE_MAX_TOKEN_NUM=1024
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        "$SWIFT" infer \
            "$@" \
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
            --result_path "$JOB/raw-result.jsonl"
    ) 2>&1 | tee "$JOB/infer.log"

    "$PYTHON" "$REPO/scripts/evaluate_e1_dev.py" \
        --result "$JOB/raw-result.jsonl" \
        --output-dir "$JOB/evaluation" \
        --expected-count 241 \
        --checkpoint-step 975 \
        --expected-dev "$TEST" \
        | tee "$JOB/evaluate.log"
}

run_infer adapter --model "$MODEL" --adapters "$ADAPTER"

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU $GPU free before merged run: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU memory was not released after adapter inference" >&2
    exit 1
fi

run_infer merged --model "$MERGED"

"$PYTHON" "$REPO/scripts/analyze_e5_975_merge_equivalence.py" --root "$OUT"
