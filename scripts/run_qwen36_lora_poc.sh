#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: LoRA PoC failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
TRAIN="$ROOT/ms_swift/e5_crop_aux20_json_v1/train.jsonl"
OUT="$ROOT/runs/qwen36_27b_lora_poc2_v1"
POC="$OUT/input/poc20.jsonl"
MANIFEST="$OUT/input/manifest.json"
RUN_ROOT="$OUT/run"
GPU_TRACE="$OUT/gpu-memory.csv"

require_executable() {
    [[ -x "$1" ]] || { echo "ERROR: missing executable: $1" >&2; exit 1; }
}

require_readable() {
    [[ -r "$1" ]] || { echo "ERROR: missing or unreadable file: $1" >&2; exit 1; }
}

require_executable "$ENV/bin/python"
require_executable "$ENV/bin/swift"
require_readable "$MODEL/config.json"
require_readable "$TRAIN"
require_readable "$REPO/scripts/build_qwen36_lora_poc_dataset.py"
require_readable "$REPO/scripts/validate_qwen36_lora_poc.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

echo "=== GPU PREFLIGHT ==="
for GPU in 4 5 6 7; do
    FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    echo "GPU $GPU free: ${FREE} MiB"
    if [[ "$FREE" -lt 70000 ]]; then
        echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
        exit 1
    fi
done

mkdir -p "$OUT/input"
"$ENV/bin/python" "$REPO/scripts/build_qwen36_lora_poc_dataset.py" \
    --train "$TRAIN" \
    --output "$POC" \
    --manifest "$MANIFEST"

echo "timestamp,gpu,memory_used_mib,memory_free_mib" > "$GPU_TRACE"
monitor_gpus() {
    while true; do
        TIMESTAMP=$(date +%s)
        nvidia-smi -i 4,5,6,7 \
            --query-gpu=index,memory.used,memory.free \
            --format=csv,noheader,nounits \
            | awk -F',' -v timestamp="$TIMESTAMP" '{gsub(/ /, ""); print timestamp "," $1 "," $2 "," $3}' \
            >> "$GPU_TRACE"
        sleep 2
    done
}
monitor_gpus &
MONITOR_PID=$!
cleanup_monitor() {
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup_monitor EXIT

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT=29636
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=1024
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$ENV/bin/swift" sft \
    --model "$MODEL" \
    --tuner_backend peft \
    --tuner_type lora \
    --dataset "$POC" \
    --split_dataset_ratio 0 \
    --dataset_shuffle false \
    --strict true \
    --lazy_tokenize false \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --max_length 3072 \
    --deepspeed zero2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 1 \
    --save_only_model false \
    --logging_steps 1 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --report_to none \
    --seed 42 \
    --data_seed 42 \
    --output_dir "$RUN_ROOT" \
    2>&1 | tee "$OUT/train.log"

cleanup_monitor
trap - EXIT

"$ENV/bin/python" "$REPO/scripts/validate_qwen36_lora_poc.py" \
    --run-root "$RUN_ROOT" \
    --gpu-trace "$GPU_TRACE" \
    --output "$OUT/validation.json" \
    --expected-step 2 \
    2>&1 | tee "$OUT/validate.log"

echo "QWEN36_LORA_DEEPSPEED_POC2: PASS"
