#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/data/h30082292/code/sft
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
CHECKPOINT_ROOT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e3_broad_clean_vit_aligner_r16_e2_v1/v0-20260722-100552
DEV=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/dev_adjudicated_v1/dev.jsonl
OUTPUT_ROOT=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e3_dev_v1/e3_vit_aligner_8ckpt_v1
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
MODE=${1:-run}

if [[ "$MODE" != "run" && "$MODE" != "--dry-run" ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

command -v swift >/dev/null
test -r "$MODEL/config.json"
test -r "$DEV"
test -d "$CHECKPOINT_ROOT"
test -r "$ROOT/scripts/run_e1_dev_checkpoints.py"

ACTUAL_DEV_SHA256=$(sha256sum "$DEV" | awk '{print $1}')
if [[ "$ACTUAL_DEV_SHA256" != "$EXPECTED_DEV_SHA256" ]]; then
    echo "ERROR: corrected Dev sha256 mismatch: $ACTUAL_DEV_SHA256" >&2
    exit 1
fi

ARGS=(
    --model "$MODEL"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --dev "$DEV"
    --output-root "$OUTPUT_ROOT"
    --gpus 4 5 6 7
    --steps 156 312 468 624 780 936 1092 1248
    --expected-good 142
    --expected-bad 58
)

if [[ "$MODE" == "--dry-run" ]]; then
    ARGS+=(--dry-run)
fi

python "$ROOT/scripts/run_e1_dev_checkpoints.py" "${ARGS[@]}"
