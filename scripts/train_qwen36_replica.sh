#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: Qwen3.6 replica failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

EXPERIMENT=${1:-}
MODE=${2:-run}
if [[ ! "$EXPERIMENT" =~ ^E[1-5]$ ]] || [[ "$MODE" != "run" && "$MODE" != "--preflight-only" ]]; then
    echo "Usage: $0 E1|E2|E3|E4|E5 [--preflight-only]" >&2
    exit 2
fi

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"

case "$EXPERIMENT" in
    E1)
        TRAIN="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"
        OUT="$ROOT/runs/qwen36_e1_broad_clean_llm_r16_s2496_v1"
        FREEZE_VIT=true; FREEZE_ALIGNER=true
        MAX_STEPS=2496; MAX_LENGTH=2048; EVAL_STEPS=156; SAVE_STEPS=312; MASTER_PORT=29701
        ;;
    E2)
        TRAIN="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"
        OUT="$ROOT/runs/qwen36_e2_broad_clean_aligner_r16_s1248_v1"
        FREEZE_VIT=true; FREEZE_ALIGNER=false
        MAX_STEPS=1248; MAX_LENGTH=2048; EVAL_STEPS=156; SAVE_STEPS=156; MASTER_PORT=29702
        ;;
    E3)
        TRAIN="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"
        OUT="$ROOT/runs/qwen36_e3_broad_clean_vit_aligner_r16_s1248_v1"
        FREEZE_VIT=false; FREEZE_ALIGNER=false
        MAX_STEPS=1248; MAX_LENGTH=2048; EVAL_STEPS=156; SAVE_STEPS=156; MASTER_PORT=29703
        ;;
    E4)
        TRAIN="$ROOT/ms_swift/e4_crop_aux_json_v1/train.jsonl"
        OUT="$ROOT/runs/qwen36_e4_crop_aux_aligner_r16_s2080_v1"
        FREEZE_VIT=true; FREEZE_ALIGNER=false
        MAX_STEPS=2080; MAX_LENGTH=3072; EVAL_STEPS=260; SAVE_STEPS=260; MASTER_PORT=29704
        ;;
    E5)
        TRAIN="$ROOT/ms_swift/e5_crop_aux20_json_v1/train.jsonl"
        OUT="$ROOT/runs/qwen36_e5_crop_aux20_aligner_r16_s1560_v1"
        FREEZE_VIT=true; FREEZE_ALIGNER=false
        MAX_STEPS=1560; MAX_LENGTH=3072; EVAL_STEPS=195; SAVE_STEPS=195; MASTER_PORT=29705
        ;;
esac

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
require_readable "$DEV"
require_readable "$REPO/scripts/validate_qwen36_replica_data.py"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

"$ENV/bin/python" "$REPO/scripts/validate_qwen36_replica_data.py" \
    --experiment "$EXPERIMENT" --train "$TRAIN" --dev "$DEV"

echo "=== GPU CHECK: $EXPERIMENT ==="
for GPU in 4 5 6 7; do
    FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    echo "GPU $GPU free: ${FREE} MiB"
    if [[ "$FREE" -lt 70000 ]]; then
        echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
        exit 1
    fi
done

echo "$EXPERIMENT: steps=$MAX_STEPS max_length=$MAX_LENGTH freeze_vit=$FREEZE_VIT freeze_aligner=$FREEZE_ALIGNER"
if [[ "$MODE" == "--preflight-only" ]]; then
    echo "QWEN36_${EXPERIMENT}_PREFLIGHT: PASS"
    exit 0
fi

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT
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
    --dataset "$TRAIN" \
    --val_dataset "$DEV" \
    --split_dataset_ratio 0 \
    --dataset_shuffle true \
    --val_dataset_shuffle false \
    --strict true \
    --lazy_tokenize true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALIGNER" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length "$MAX_LENGTH" \
    --deepspeed zero2 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 8 \
    --save_only_model false \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 2 \
    --report_to none \
    --seed 42 \
    --data_seed 42 \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log"

echo "QWEN36_${EXPERIMENT}_TRAIN: PASS"
