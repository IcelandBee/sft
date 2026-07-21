#!/usr/bin/env bash
set -euo pipefail

MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
DATA=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e1_broad_clean_json_v1
OUT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e2_broad_clean_aligner_r16_e2_v1

TRAIN="$DATA/train.jsonl"
DEV="$DATA/dev.jsonl"
MODE="${1:-run}"

if [[ "$MODE" != "run" && "$MODE" != "--preflight-only" ]]; then
    echo "Usage: $0 [--preflight-only]"
    exit 2
fi

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$TRAIN"
test -r "$DEV"

if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT"
    exit 1
fi

python - "$TRAIN" "$DEV" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

train_path, dev_path = map(Path, sys.argv[1:])

def load(path):
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {number} in {path} is not an object")
            rows.append(row)
    return rows

def decision(row):
    return json.loads(row["messages"][-1]["content"])["decision"]

train = load(train_path)
dev = load(dev_path)
if len(train) != 9978 or len(dev) != 200:
    raise ValueError(f"unexpected rows: train={len(train)} dev={len(dev)}")

train_counts = Counter(decision(row) for row in train)
dev_counts = Counter(decision(row) for row in dev)
if train_counts != {"GOOD": 6074, "BAD": 3904}:
    raise ValueError(f"unexpected Train decisions: {dict(train_counts)}")
if dev_counts != {"GOOD": 149, "BAD": 51}:
    raise ValueError(f"unexpected Dev decisions: {dict(dev_counts)}")

train_images = [row["images"][0] for row in train]
dev_images = [row["images"][0] for row in dev]
if len(set(train_images)) != 8026:
    raise ValueError(f"unexpected unique Train images: {len(set(train_images))}")
if len(set(dev_images)) != 200:
    raise ValueError(f"unexpected unique Dev images: {len(set(dev_images))}")
if set(train_images) & set(dev_images):
    raise ValueError("Train/Dev image overlap detected")

missing = [path for path in set(train_images + dev_images) if not Path(path).is_file()]
if missing:
    raise ValueError(f"missing image files: {len(missing)}; first={missing[0]}")

print(
    "DATA_CHECK: PASS",
    f"train_rows={len(train)}",
    f"train_unique={len(set(train_images))}",
    f"dev_rows={len(dev)}",
)
PY

echo "=== GPU CHECK ==="
for GPU in 4 5 6 7; do
    FREE=$(nvidia-smi -i "$GPU" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -dc '0-9')
    echo "GPU $GPU free: ${FREE} MiB"
    if [[ "$FREE" -lt 70000 ]]; then
        echo "ERROR: GPU $GPU has less than 70000 MiB free"
        exit 1
    fi
done

echo "E2_PROTOCOL: broad-clean BADx2, LLM+aligner LoRA r16, ViT frozen, 2 epochs"

if [[ "$MODE" == "--preflight-only" ]]; then
    echo "E2_PREFLIGHT_CHECK: PASS"
    exit 0
fi

mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT=29614
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export IMAGE_MAX_TOKEN_NUM=1024

swift sft \
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
    --freeze_vit true \
    --freeze_aligner false \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length 2048 \
    --deepspeed zero2 \
    --eval_strategy steps \
    --eval_steps 156 \
    --save_strategy steps \
    --save_steps 156 \
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
