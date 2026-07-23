#!/usr/bin/env bash
set -euo pipefail

MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
DATA=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e4_crop_aux_json_v1
OUT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e4_crop_aux_aligner_r16_s2080_v1

TRAIN="$DATA/train.jsonl"
DEV="$DATA/dev.jsonl"
MANIFEST="$DATA/local_manifest.jsonl"
SUMMARY="$DATA/build_summary.json"
MODE="${1:-run}"

EXPECTED_TRAIN_SHA256=23b61202f3d92b87847c13f6c3df3597db93b6390ef3e9896ecf9be886e34a08
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb

if [[ "$MODE" != "run" && "$MODE" != "--preflight-only" ]]; then
    echo "Usage: $0 [--preflight-only]"
    exit 2
fi

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$TRAIN"
test -r "$DEV"
test -r "$MANIFEST"
test -r "$SUMMARY"

if [[ -e "$OUT" ]]; then
    echo "ERROR: output directory already exists: $OUT"
    exit 1
fi

python - \
    "$TRAIN" \
    "$DEV" \
    "$MANIFEST" \
    "$SUMMARY" \
    "$EXPECTED_TRAIN_SHA256" \
    "$EXPECTED_DEV_SHA256" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

train_path, dev_path, manifest_path, summary_path = map(Path, sys.argv[1:5])
expected_train_sha256, expected_dev_sha256 = sys.argv[5:7]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("every row must contain exactly three messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("invalid message roles")
    payload = json.loads(messages[-1]["content"])
    value = payload.get("decision")
    if value not in {"GOOD", "BAD"}:
        raise ValueError(f"invalid decision: {value}")
    return value


train_sha256 = sha256(train_path)
dev_sha256 = sha256(dev_path)
if train_sha256 != expected_train_sha256:
    raise ValueError(
        f"unexpected Train sha256: expected={expected_train_sha256} actual={train_sha256}"
    )
if dev_sha256 != expected_dev_sha256:
    raise ValueError(
        f"unexpected Dev sha256: expected={expected_dev_sha256} actual={dev_sha256}"
    )

train = load(train_path)
dev = load(dev_path)
manifest = load(manifest_path)
summary = json.loads(summary_path.read_text(encoding="utf-8"))

if len(train) != 16630 or len(dev) != 200 or len(manifest) != 6652:
    raise ValueError(
        f"unexpected rows: train={len(train)} dev={len(dev)} manifest={len(manifest)}"
    )

train_image_counts = Counter(len(row.get("images", [])) for row in train)
if train_image_counts != {1: 9978, 2: 6652}:
    raise ValueError(f"unexpected Train image counts: {dict(train_image_counts)}")
train_decisions = Counter(decision(row) for row in train)
dev_decisions = Counter(decision(row) for row in dev)
if train_decisions != {"GOOD": 9400, "BAD": 7230}:
    raise ValueError(f"unexpected Train decisions: {dict(train_decisions)}")
if dev_decisions != {"GOOD": 142, "BAD": 58}:
    raise ValueError(f"unexpected Dev decisions: {dict(dev_decisions)}")

manifest_types = Counter(row.get("sample_type") for row in manifest)
if manifest_types != {"T2_BAD": 3326, "T3_GOOD": 3326}:
    raise ValueError(f"unexpected local manifest types: {dict(manifest_types)}")
if len({row["train_output_index"] for row in manifest}) != len(manifest):
    raise ValueError("duplicate train_output_index in local manifest")
for row in manifest:
    index = row["train_output_index"]
    if not isinstance(index, int) or not 0 <= index < len(train):
        raise ValueError(f"invalid train_output_index: {index}")
    expected_decision = "BAD" if row["sample_type"] == "T2_BAD" else "GOOD"
    if len(train[index]["images"]) != 2 or decision(train[index]) != expected_decision:
        raise ValueError(f"manifest/train mismatch at train row {index}")

expected_sample_types = {"T1_FULL": 9978, "T2_BAD": 3326, "T3_GOOD": 3326}
if summary.get("output", {}).get("sample_type_counts") != expected_sample_types:
    raise ValueError("build summary sample counts differ from the frozen protocol")
if summary.get("output", {}).get("train_sha256") != train_sha256:
    raise ValueError("build summary Train sha256 mismatch")
if summary.get("output", {}).get("dev_sha256") != dev_sha256:
    raise ValueError("build summary Dev sha256 mismatch")
if summary.get("test_untouched") is not True:
    raise ValueError("build summary does not confirm Test isolation")

first_train_images = {row["images"][0] for row in train}
dev_images = {row["images"][0] for row in dev}
if len(first_train_images) != 8026:
    raise ValueError(f"unexpected unique Train source images: {len(first_train_images)}")
if len(dev_images) != 200:
    raise ValueError(f"unexpected unique Dev images: {len(dev_images)}")
if first_train_images & dev_images:
    raise ValueError("Train/Dev image overlap detected")

all_images = {image for row in train + dev for image in row["images"]}
missing = [image for image in all_images if not Path(image).is_file()]
if missing:
    raise ValueError(f"missing image files: {len(missing)}; first={missing[0]}")

print(
    "DATA_CHECK: PASS",
    f"train_rows={len(train)}",
    f"sample_types={expected_sample_types}",
    f"train_decisions={dict(train_decisions)}",
    f"dev_decisions={dict(dev_decisions)}",
    f"train_sha256={train_sha256}",
    f"dev_sha256={dev_sha256}",
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

echo "E4_PROTOCOL: T1/T2/T3=9978/3326/3326, LLM+aligner all-linear LoRA r16, ViT frozen, max_steps=2080 (~2 epochs), max_length=3072"

if [[ "$MODE" == "--preflight-only" ]]; then
    echo "E4_PREFLIGHT_CHECK: PASS"
    exit 0
fi

mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT=29618
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
    --max_steps 2080 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length 3072 \
    --deepspeed zero2 \
    --eval_strategy steps \
    --eval_steps 260 \
    --save_strategy steps \
    --save_steps 260 \
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
