#!/usr/bin/env bash
set -euo pipefail

DEFAULT_RUN=/home/data/h30082292/data/pose/artifact_detection_training/runs/e3_broad_clean_vit_aligner_r16_e2_v1/v0-20260722-100552
RUN_DIR=${1:-$DEFAULT_RUN}

test -d "$RUN_DIR" || {
    echo "ERROR: run directory does not exist: $RUN_DIR" >&2
    exit 1
}

python - "$RUN_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open


run_dir = Path(sys.argv[1]).resolve()
checkpoint_re = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    match = checkpoint_re.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory: {path}")
    return int(match.group(1))


checkpoints = sorted(
    (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
    key=checkpoint_step,
)
if not checkpoints:
    raise SystemExit(f"ERROR: no checkpoints found under {run_dir}")

print(f"run_dir: {run_dir}")
print("=== CHECKPOINTS ===")
checkpoint_failures = []
for checkpoint in checkpoints:
    weights = sorted(checkpoint.glob("adapter_model*.safetensors"))
    config_ok = (checkpoint / "adapter_config.json").is_file()
    state_ok = (checkpoint / "trainer_state.json").is_file()
    weights_ok = bool(weights)
    size_gib = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    ) / 1024**3
    print(
        f"{checkpoint.name} size={size_gib:.3f}GiB "
        f"weights={'OK' if weights_ok else 'MISSING'} "
        f"config={'OK' if config_ok else 'MISSING'} "
        f"state={'OK' if state_ok else 'MISSING'}"
    )
    if not (weights_ok and config_ok and state_ok):
        checkpoint_failures.append(checkpoint.name)

if checkpoint_failures:
    raise SystemExit(
        "ERROR: incomplete checkpoints: " + ", ".join(checkpoint_failures)
    )

latest = checkpoints[-1]
state_path = latest / "trainer_state.json"
with state_path.open(encoding="utf-8") as handle:
    state = json.load(handle)

eval_rows = [
    row
    for row in state.get("log_history", [])
    if "eval_loss" in row and "step" in row
]
if not eval_rows:
    raise SystemExit(f"ERROR: no eval history in {state_path}")

print("\n=== EVAL HISTORY ===")
for row in eval_rows:
    token_acc = row.get("eval_token_acc")
    print(
        f"step={row['step']} epoch={row.get('epoch')} "
        f"eval_loss={row['eval_loss']} token_acc={token_acc}"
    )

print("\n=== TRAINER SELECTION ===")
print(f"best_model_checkpoint: {state.get('best_model_checkpoint')}")
print(f"best_metric: {state.get('best_metric')}")
print(f"global_step: {state.get('global_step')}")

weight_paths = sorted(latest.glob("adapter_model*.safetensors"))
tensor_names = []
for weight_path in weight_paths:
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        tensor_names.extend(handle.keys())

visual_names = [name for name in tensor_names if ".visual." in name]
aligner_names = [
    name
    for name in visual_names
    if any(marker in name.lower() for marker in ("merger", "aligner", "projector"))
]
vit_names = [name for name in visual_names if name not in set(aligner_names)]

print("\n=== LORA TENSOR COVERAGE ===")
print(f"adapter_tensor_count: {len(tensor_names)}")
print(f"visual_tensor_count: {len(visual_names)}")
print(f"aligner_tensor_count: {len(aligner_names)}")
print(f"vit_tensor_count: {len(vit_names)}")
print("visual_tensor_examples:")
for name in visual_names[:12]:
    print(name)

expected_steps = list(range(156, 1248 + 1, 156))
actual_steps = [checkpoint_step(path) for path in checkpoints]
eval_steps = [int(row["step"]) for row in eval_rows]
if run_dir == Path("/home/data/h30082292/data/pose/artifact_detection_training/runs/e3_broad_clean_vit_aligner_r16_e2_v1/v0-20260722-100552"):
    if actual_steps != expected_steps:
        raise SystemExit(
            f"ERROR: E3 checkpoint steps mismatch: {actual_steps} != {expected_steps}"
        )
    if eval_steps != expected_steps:
        raise SystemExit(
            f"ERROR: E3 eval steps mismatch: {eval_steps} != {expected_steps}"
        )
    if state.get("global_step") != 1248:
        raise SystemExit(
            f"ERROR: E3 global_step mismatch: {state.get('global_step')} != 1248"
        )
    if not aligner_names or not vit_names:
        raise SystemExit("ERROR: E3 adapter lacks aligner or ViT LoRA tensors")

print("LORA_RUN_CHECK: PASS")
PY
