#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
BASE_MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
EXPORT_BASE=/home/data/h30082292/DATA_71/h30082292/models
OUT="$EXPORT_BASE/qwen35_27b_four_merged_models_v1"
TRAIN_ENV=/home/data/h30082292/miniconda3/envs/qwen35_27b
PYTHON="$TRAIN_ENV/bin/python"
SWIFT="$TRAIN_ENV/bin/swift"
GPU=${MERGE_GPU:-2}
REQUIRED_FREE_GIB=260

NAMES=(
    qwen35-27b-e1-1248-merged-bf16
    qwen35-27b-e2-1248-merged-bf16
    qwen35-27b-e5-780-recall-merged-bf16
    qwen35-27b-e5-975-balanced-merged-bf16
)
ADAPTERS=(
    "$ROOT/runs/e1_broad_clean_r16_e4_v1/v0-20260717-185936/checkpoint-1248"
    "$ROOT/runs/e2_broad_clean_aligner_r16_e2_v1/v0-20260721-114449/checkpoint-1248"
    "$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-780"
    "$ROOT/runs/e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158/checkpoint-975"
)

test -x "$PYTHON"
test -x "$SWIFT"
command -v nvidia-smi >/dev/null
test -r "$BASE_MODEL/config.json"
test -r "$BASE_MODEL/generation_config.json"
test -r "$BASE_MODEL/tokenizer_config.json"
test -r "$BASE_MODEL/preprocessor_config.json"
test -d "$EXPORT_BASE"
test -w "$EXPORT_BASE"
test -r "$REPO/scripts/validate_four_merged_models.py"

for index in "${!NAMES[@]}"; do
    adapter=${ADAPTERS[$index]}
    test -r "$adapter/adapter_config.json"
    compgen -G "$adapter/adapter_model*.safetensors" >/dev/null
    echo "ADAPTER_CHECK: ${NAMES[$index]} <- $adapter"
done

FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
FREE_GIB=$(df -BG --output=avail "$EXPORT_BASE" | tail -n 1 | tr -dc '0-9')
echo "GPU $GPU free: ${FREE_MIB} MiB"
echo "disk free: ${FREE_GIB} GiB"
echo "required output space: ${REQUIRED_FREE_GIB} GiB"
if [[ "$FREE_MIB" -lt 70000 ]]; then
    echo "ERROR: GPU $GPU has less than 70000 MiB free" >&2
    exit 1
fi
if [[ "$FREE_GIB" -lt "$REQUIRED_FREE_GIB" ]]; then
    echo "ERROR: less than ${REQUIRED_FREE_GIB} GiB disk space is available" >&2
    exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "ERROR: output already exists: $OUT" >&2
    exit 1
fi

if [[ "${1:-}" == "--preflight-only" ]]; then
    echo "FOUR_MODEL_MERGE_PREFLIGHT: PASS"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "ERROR: unsupported argument: $1" >&2
    exit 1
fi

mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for index in "${!NAMES[@]}"; do
    name=${NAMES[$index]}
    adapter=${ADAPTERS[$index]}
    merged="$OUT/$name"
    log="$OUT/logs/$name-merge.log"

    echo "=== MERGE $name ($((index + 1))/${#NAMES[@]}) ==="
    "$SWIFT" export \
        --model "$BASE_MODEL" \
        --adapters "$adapter" \
        --merge_lora true \
        --torch_dtype bfloat16 \
        --output_dir "$merged" \
        2>&1 | tee "$log"

    test -r "$merged/config.json"
    compgen -G "$merged/model-*.safetensors" >/dev/null
    if compgen -G "$merged/adapter_model*.safetensors" >/dev/null; then
        echo "ERROR: merged output still contains adapter weights: $merged" >&2
        exit 1
    fi
    echo "MERGE_COMPLETE: $merged"
done

"$PYTHON" "$REPO/scripts/validate_four_merged_models.py" \
    --root "$OUT" \
    --base-model "$BASE_MODEL" \
    --entry "${NAMES[0]}=${ADAPTERS[0]}" \
    --entry "${NAMES[1]}=${ADAPTERS[1]}" \
    --entry "${NAMES[2]}=${ADAPTERS[2]}" \
    --entry "${NAMES[3]}=${ADAPTERS[3]}" \
    | tee "$OUT/validation.log"

echo "=== FOUR MERGED MODELS ==="
du -sh "$OUT"/qwen35-27b-*-merged-bf16
echo "manifest=$OUT/four-merged-models-manifest.json"
echo "FOUR_MODEL_MERGE: PASS"
