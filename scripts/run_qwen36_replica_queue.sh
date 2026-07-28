#!/usr/bin/env bash
set -euo pipefail

REPO=/home/data/h30082292/code/sft
ROOT=/home/data/h30082292/data/pose/artifact_detection_training
QUEUE_ROOT="$ROOT/runs/qwen36_replica_queue_v1"
STATUS="$QUEUE_ROOT/status.tsv"
DEFAULT_ORDER=(E5 E2 E1 E3 E4)

if [[ $# -eq 0 ]]; then
    EXPERIMENTS=("${DEFAULT_ORDER[@]}")
else
    EXPERIMENTS=("$@")
fi
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    if [[ ! "$EXPERIMENT" =~ ^E[1-5]$ ]]; then
        echo "ERROR: invalid experiment: $EXPERIMENT" >&2
        exit 2
    fi
done
if [[ -e "$QUEUE_ROOT" ]]; then
    echo "ERROR: queue directory already exists: $QUEUE_ROOT" >&2
    exit 1
fi

echo "=== PREFLIGHT ALL REPLICA EXPERIMENTS ==="
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    bash "$REPO/scripts/train_qwen36_replica.sh" "$EXPERIMENT" --preflight-only
done

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

run_stage BASE_DEV bash "$REPO/scripts/run_qwen36_base_dev.sh"
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    run_stage "$EXPERIMENT" bash "$REPO/scripts/train_qwen36_replica.sh" "$EXPERIMENT"
done

echo "QWEN36_REPLICA_QUEUE: PASS"
