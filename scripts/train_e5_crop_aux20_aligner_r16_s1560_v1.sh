#!/usr/bin/env bash
set -euo pipefail

MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
DATA=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e5_crop_aux20_json_v1
OUT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e5_crop_aux20_aligner_r16_s1560_v1
TRAIN="$DATA/train.jsonl"
DEV="$DATA/dev.jsonl"
SUMMARY="$DATA/build_summary.json"
MODE="${1:-run}"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb

if [[ "$MODE" != "run" && "$MODE" != "--preflight-only" ]]; then
    echo "Usage: $0 [--preflight-only]" >&2
    exit 2
fi

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$TRAIN"
test -r "$DEV"
test -r "$SUMMARY"
if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT" >&2
    exit 1
fi

python - "$TRAIN" "$DEV" "$SUMMARY" "$EXPECTED_DEV_SHA256" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

train_path, dev_path, summary_path = map(Path, sys.argv[1:4])
expected_dev_sha = sys.argv[4]

def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def decision(row):
    return json.loads(row["messages"][-1]["content"])["decision"]

train = load(train_path)
dev = load(dev_path)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
train_sha = sha(train_path)
dev_sha = sha(dev_path)
if len(train) != 12472 or len(dev) != 200:
    raise ValueError(f"unexpected rows: train={len(train)} dev={len(dev)}")
if Counter(len(row["images"]) for row in train) != {1: 9978, 2: 2494}:
    raise ValueError("unexpected single/two-image distribution")
if Counter(decision(row) for row in train) != {"GOOD": 7321, "BAD": 5151}:
    raise ValueError("unexpected Train decisions")
if Counter(decision(row) for row in dev) != {"GOOD": 142, "BAD": 58}:
    raise ValueError("unexpected Dev decisions")
if summary.get("sample_type_counts") != {
    "T1_FULL": 9978,
    "T2_BAD": 1247,
    "T3_GOOD": 1247,
}:
    raise ValueError("unexpected summary sample types")
if summary.get("train_sha256") != train_sha:
    raise ValueError("Train sha256 differs from build summary")
if dev_sha != expected_dev_sha or summary.get("dev_sha256") != dev_sha:
    raise ValueError("corrected Dev sha256 mismatch")
if summary.get("test_untouched") is not True:
    raise ValueError("Test isolation not confirmed")
missing = [
    image
    for image in {image for row in train + dev for image in row["images"]}
    if not Path(image).is_file()
]
if missing:
    raise ValueError(f"missing images: {len(missing)}; first={missing[0]}")
train_sources = {row["images"][0] for row in train}
dev_sources = {row["images"][0] for row in dev}
if len(train_sources) != 8026 or len(dev_sources) != 200:
    raise ValueError("unexpected unique source image counts")
if train_sources & dev_sources:
    raise ValueError("Train/Dev image overlap")
print(
    "DATA_CHECK: PASS",
    f"train_rows={len(train)}",
    f"train_decisions={dict(Counter(decision(row) for row in train))}",
    f"dev_sha256={dev_sha}",
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

echo "E5_PROTOCOL: T1/T2/T3=9978/1247/1247, LLM+aligner LoRA r16, ViT frozen, max_steps=1560 (~2 epochs), max_length=3072"
if [[ "$MODE" == "--preflight-only" ]]; then
    echo "E5_PREFLIGHT_CHECK: PASS"
    exit 0
fi

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT=29620
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
    --max_steps 1560 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length 3072 \
    --deepspeed zero2 \
    --eval_strategy steps \
    --eval_steps 195 \
    --save_strategy steps \
    --save_steps 195 \
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
