#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "ERROR: Qwen3.6 Dev checkpoint evaluation failed at line $LINENO: $BASH_COMMAND (exit=$status)" >&2' ERR

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B
DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"
EXPECTED_DEV_SHA256=cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb
EXPERIMENT=${1:-}
MODE=${2:-run}

if [[ ! "$EXPERIMENT" =~ ^E[1-5]$ ]] || [[ "$MODE" != "run" && "$MODE" != "--dry-run" ]]; then
    echo "Usage: $0 E1|E2|E3|E4|E5 [--dry-run]" >&2
    exit 2
fi

case "$EXPERIMENT" in
    E1)
        RUN_ROOT="$ROOT/runs/qwen36_e1_broad_clean_llm_r16_s2496_v1/v0-20260729-015806"
        OUTPUT_ROOT="$ROOT/evaluations/qwen36_27b/e1_dev_8ckpt_v1"
        STEPS=(312 624 936 1248 1560 1872 2184 2496)
        ;;
    E2)
        RUN_ROOT="$ROOT/runs/qwen36_e2_broad_clean_aligner_r16_s1248_v1/v0-20260728-233940"
        OUTPUT_ROOT="$ROOT/evaluations/qwen36_27b/e2_dev_8ckpt_v1"
        STEPS=(156 312 468 624 780 936 1092 1248)
        ;;
    E3)
        RUN_ROOT="$ROOT/runs/qwen36_e3_broad_clean_vit_aligner_r16_s1248_v1/v0-20260729-062123"
        OUTPUT_ROOT="$ROOT/evaluations/qwen36_27b/e3_dev_8ckpt_v1"
        STEPS=(156 312 468 624 780 936 1092 1248)
        ;;
    E4)
        RUN_ROOT="$ROOT/runs/qwen36_e4_crop_aux_aligner_r16_s2080_v1/v0-20260729-084616"
        OUTPUT_ROOT="$ROOT/evaluations/qwen36_27b/e4_dev_8ckpt_v1"
        STEPS=(260 520 780 1040 1300 1560 1820 2080)
        ;;
    E5)
        RUN_ROOT="$ROOT/runs/qwen36_e5_crop_aux20_aligner_r16_s1560_v1/v0-20260728-204136"
        OUTPUT_ROOT="$ROOT/evaluations/qwen36_27b/e5_dev_8ckpt_v1"
        STEPS=(195 390 585 780 975 1170 1365 1560)
        ;;
esac

[[ -x "$ENV/bin/python" ]] || { echo "ERROR: missing executable: $ENV/bin/python" >&2; exit 1; }
[[ -x "$ENV/bin/swift" ]] || { echo "ERROR: missing executable: $ENV/bin/swift" >&2; exit 1; }
[[ -r "$MODEL/config.json" ]] || { echo "ERROR: missing model config: $MODEL/config.json" >&2; exit 1; }
[[ -r "$DEV" ]] || { echo "ERROR: missing Dev: $DEV" >&2; exit 1; }
[[ -r "$REPO/scripts/run_e1_dev_checkpoints.py" ]] || { echo "ERROR: missing checkpoint runner" >&2; exit 1; }

ACTUAL_DEV_SHA256=$(sha256sum "$DEV" | awk '{print $1}')
if [[ "$ACTUAL_DEV_SHA256" != "$EXPECTED_DEV_SHA256" ]]; then
    echo "ERROR: corrected Dev sha256 mismatch: $ACTUAL_DEV_SHA256" >&2
    exit 1
fi

export PATH="$ENV/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ARGS=(
    --model "$MODEL"
    --checkpoint-root "$RUN_ROOT"
    --dev "$DEV"
    --output-root "$OUTPUT_ROOT"
    --gpus 4 5 6 7
    --steps "${STEPS[@]}"
    --expected-good 142
    --expected-bad 58
)
if [[ "$MODE" == "--dry-run" ]]; then
    ARGS+=(--dry-run)
fi

echo "=== QWEN3.6 $EXPERIMENT FIXED DEV CHECKPOINTS ==="
echo "run_root=$RUN_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "steps=${STEPS[*]}"

if "$ENV/bin/python" "$REPO/scripts/run_e1_dev_checkpoints.py" "${ARGS[@]}"; then
    STATUS=0
else
    STATUS=$?
fi

if [[ "$STATUS" -eq 3 ]]; then
    echo "QWEN36_${EXPERIMENT}_DEV_CHECKPOINTS: COMPLETE_NO_ELIGIBLE_CHECKPOINT"
    exit 0
fi
if [[ "$STATUS" -ne 0 ]]; then
    exit "$STATUS"
fi
if [[ "$MODE" == "run" ]]; then
    "$ENV/bin/python" - "$EXPERIMENT" "$OUTPUT_ROOT/checkpoint-summary.json" <<'PY'
import json
import sys
from pathlib import Path

experiment, path = sys.argv[1:]
summary = json.loads(Path(path).read_text(encoding="utf-8"))
step = summary["selected_step"]
selected = next(row for row in summary["checkpoints"] if row["checkpoint_step"] == step)
print(
    f"SELECTED {experiment} checkpoint-{step}: "
    f"TP={selected['tp']} FN={selected['fn']} FP={selected['fp']} TN={selected['tn']} "
    f"Recall={selected['recall']:.2%} FPR={selected['fpr']:.2%} "
    f"Accuracy={selected['accuracy']:.2%} F1={selected['f1']:.2%} "
    f"JSON={selected['schema_valid_rate']:.2%}"
)
PY
fi
echo "QWEN36_${EXPERIMENT}_DEV_CHECKPOINTS: PASS"
