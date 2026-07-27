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

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_executable() {
    [[ -x "$1" ]] || fail "executable does not exist or is not executable: $1"
}

require_readable_file() {
    [[ -r "$1" ]] || fail "required file is missing or unreadable: $1"
}

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

if [[ $# -gt 1 ]] || { [[ $# -eq 1 ]] && [[ "$1" != "--preflight-only" ]]; }; then
    fail "usage: $0 [--preflight-only]"
fi

echo "=== FOUR MODEL MERGE PREFLIGHT ==="
echo "base_model=$BASE_MODEL"
echo "output_root=$OUT"
echo "merge_gpu=$GPU"

require_executable "$PYTHON"
require_executable "$SWIFT"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not available"
require_readable_file "$BASE_MODEL/config.json"
require_readable_file "$BASE_MODEL/generation_config.json"
require_readable_file "$BASE_MODEL/tokenizer_config.json"
require_readable_file "$BASE_MODEL/preprocessor_config.json"
require_readable_file "$REPO/scripts/validate_four_merged_models.py"

EXPORT_PARENT=$(dirname "$EXPORT_BASE")
[[ -d "$EXPORT_PARENT" ]] || fail "output parent directory does not exist: $EXPORT_PARENT"
[[ -w "$EXPORT_PARENT" ]] || fail "output parent directory is not writable: $EXPORT_PARENT"
if [[ -e "$EXPORT_BASE" ]] && [[ ! -d "$EXPORT_BASE" ]]; then
    fail "output base exists but is not a directory: $EXPORT_BASE"
fi
if [[ -d "$EXPORT_BASE" ]]; then
    [[ -w "$EXPORT_BASE" ]] || fail "output base directory is not writable: $EXPORT_BASE"
    DISK_PROBE=$EXPORT_BASE
    echo "output_base_status=EXISTS_WRITABLE"
else
    DISK_PROBE=$EXPORT_PARENT
    echo "output_base_status=MISSING_WILL_CREATE"
fi

for index in "${!NAMES[@]}"; do
    adapter=${ADAPTERS[$index]}
    require_readable_file "$adapter/adapter_config.json"
    compgen -G "$adapter/adapter_model*.safetensors" >/dev/null || \
        fail "adapter weights are missing: $adapter/adapter_model*.safetensors"
    echo "ADAPTER_CHECK: ${NAMES[$index]} <- $adapter"
done

GPU_QUERY=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits) || \
    fail "cannot query GPU index $GPU"
FREE_MIB=$(printf '%s' "$GPU_QUERY" | tr -dc '0-9')
[[ -n "$FREE_MIB" ]] || fail "GPU free-memory query returned no numeric value for GPU $GPU"
DISK_QUERY=$(df -BG --output=avail "$DISK_PROBE") || fail "cannot query disk space for: $DISK_PROBE"
FREE_GIB=$(printf '%s\n' "$DISK_QUERY" | tail -n 1 | tr -dc '0-9')
[[ -n "$FREE_GIB" ]] || fail "disk-space query returned no numeric value for: $DISK_PROBE"
echo "GPU $GPU free: ${FREE_MIB} MiB"
echo "disk free: ${FREE_GIB} GiB"
echo "required output space: ${REQUIRED_FREE_GIB} GiB"
if [[ "$FREE_MIB" -lt 70000 ]]; then
    fail "GPU $GPU has less than 70000 MiB free"
fi
if [[ "$FREE_GIB" -lt "$REQUIRED_FREE_GIB" ]]; then
    fail "less than ${REQUIRED_FREE_GIB} GiB disk space is available"
fi
if [[ -e "$OUT" ]]; then
    fail "output already exists: $OUT"
fi

if [[ "${1:-}" == "--preflight-only" ]]; then
    echo "FOUR_MODEL_MERGE_PREFLIGHT: PASS"
    exit 0
fi

mkdir -p "$EXPORT_BASE"
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

    require_readable_file "$merged/config.json"
    compgen -G "$merged/model-*.safetensors" >/dev/null || \
        fail "merged model shards are missing: $merged/model-*.safetensors"
    if compgen -G "$merged/adapter_model*.safetensors" >/dev/null; then
        fail "merged output still contains adapter weights: $merged"
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
