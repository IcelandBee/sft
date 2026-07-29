#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ENV=/home/data/h30082292/miniconda3/envs/qwen36_27b
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
EVAL_ROOT="$ROOT/evaluations/qwen36_27b"
QUEUE_ROOT="$EVAL_ROOT/dev_checkpoint_queue_v1"
STATUS="$QUEUE_ROOT/status.tsv"
COMPARISON="$EVAL_ROOT/dev_comparison_v1.json"
DEFAULT_ORDER=(E5 E2 E1 E3 E4)

if [[ $# -eq 0 ]]; then
    EXPERIMENTS=("${DEFAULT_ORDER[@]}")
else
    EXPERIMENTS=("$@")
fi
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    if [[ ! "$EXPERIMENT" =~ ^E[1-5]$ ]]; then
        echo "Usage: $0 [E1 E2 E3 E4 E5]" >&2
        exit 2
    fi
done
if [[ -e "$QUEUE_ROOT" ]]; then
    echo "ERROR: queue directory already exists: $QUEUE_ROOT" >&2
    exit 1
fi
if [[ -e "$COMPARISON" ]]; then
    echo "ERROR: comparison output already exists: $COMPARISON" >&2
    exit 1
fi
[[ -x "$ENV/bin/python" ]] || { echo "ERROR: missing executable: $ENV/bin/python" >&2; exit 1; }

mkdir -p "$QUEUE_ROOT"
printf 'stage\tstarted\tended\tstatus\n' > "$STATUS"

run_stage() {
    local STAGE=$1
    shift
    local STARTED ENDED
    STARTED=$(date --iso-8601=seconds)
    echo "=== START $STAGE at $STARTED ==="
    if "$@"; then
        ENDED=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\tcompleted\n' "$STAGE" "$STARTED" "$ENDED" >> "$STATUS"
        echo "=== COMPLETE $STAGE at $ENDED ==="
    else
        local EXIT_CODE=$?
        ENDED=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\tfailed(%s)\n' "$STAGE" "$STARTED" "$ENDED" "$EXIT_CODE" >> "$STATUS"
        echo "ERROR: stage $STAGE failed with exit $EXIT_CODE" >&2
        return "$EXIT_CODE"
    fi
}

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    run_stage "$EXPERIMENT" bash "$REPO/scripts/run_qwen36_dev_checkpoints.sh" "$EXPERIMENT"
done

run_stage SUMMARY "$ENV/bin/python" "$REPO/scripts/summarize_qwen36_dev_results.py" \
    --root "$EVAL_ROOT" \
    --output "$COMPARISON" \
    --experiments "${EXPERIMENTS[@]}"

echo "status=$STATUS"
echo "comparison=$COMPARISON"
echo "QWEN36_DEV_EVALUATION_QUEUE: PASS"
