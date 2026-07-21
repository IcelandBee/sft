#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_e2_dev_boundary_audit_v1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

test -r "$DATA_DIR/review.jsonl"

exec python "$SCRIPT_DIR/dev_audit_web.py" --data-dir "$DATA_DIR" "$@"
