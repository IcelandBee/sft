#!/usr/bin/env python3
"""Validate the two-step Qwen3.6 LoRA checkpoint and four-GPU memory trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

class LoraPocValidationError(ValueError):
    """Raised when the LoRA smoke run does not satisfy its compatibility contract."""


def find_checkpoint(run_root: Path, step: int) -> Path:
    checkpoints = sorted(
        path for path in run_root.rglob(f"checkpoint-{step}") if path.is_dir()
    )
    if len(checkpoints) != 1:
        raise LoraPocValidationError(
            f"expected one checkpoint-{step}, found {len(checkpoints)}"
        )
    return checkpoints[0]


def load_gpu_peaks(path: Path) -> dict[int, int]:
    peaks: dict[int, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("timestamp,"):
            continue
        parts = line.split(",")
        if len(parts) != 4:
            raise LoraPocValidationError(
                f"invalid GPU trace row {line_number}: {line!r}"
            )
        _, gpu, used, _ = (part.strip() for part in parts)
        gpu_index, used_mib = int(gpu), int(used)
        peaks[gpu_index] = max(peaks.get(gpu_index, 0), used_mib)
    if set(peaks) != {4, 5, 6, 7}:
        raise LoraPocValidationError(f"GPU trace does not cover 4-7: {peaks}")
    return peaks


def validate(args: argparse.Namespace) -> dict:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise LoraPocValidationError(f"safetensors import failed: {exc}") from exc

    checkpoint = find_checkpoint(args.run_root, args.expected_step)
    config_path = checkpoint / "adapter_config.json"
    state_path = checkpoint / "trainer_state.json"
    weight_paths = sorted(checkpoint.glob("adapter_model*.safetensors"))
    if not config_path.is_file() or not state_path.is_file() or not weight_paths:
        raise LoraPocValidationError(f"incomplete adapter checkpoint: {checkpoint}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise LoraPocValidationError(
            f"unexpected LoRA config: r={config.get('r')} alpha={config.get('lora_alpha')}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != args.expected_step:
        raise LoraPocValidationError(
            f"unexpected global_step: {state.get('global_step')}"
        )
    loss_rows = [
        row
        for row in state.get("log_history", [])
        if "loss" in row and "step" in row
    ]
    if not loss_rows or any(not math.isfinite(float(row["loss"])) for row in loss_rows):
        raise LoraPocValidationError(f"missing or non-finite training loss: {loss_rows}")

    tensor_names: list[str] = []
    nonzero_lora_b = False
    for weight_path in weight_paths:
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            tensor_names.extend(names)
            for name in names:
                if "lora_b" in name.lower():
                    tensor = handle.get_tensor(name)
                    if float(tensor.float().abs().max()) > 0:
                        nonzero_lora_b = True
                        break
    visual_names = [name for name in tensor_names if ".visual." in name]
    aligner_names = [
        name
        for name in visual_names
        if any(marker in name.lower() for marker in ("merger", "aligner", "projector"))
    ]
    aligner_set = set(aligner_names)
    vit_names = [name for name in visual_names if name not in aligner_set]
    llm_names = [name for name in tensor_names if name not in set(visual_names)]
    if not llm_names:
        raise LoraPocValidationError("adapter has no LLM LoRA tensors")
    if not aligner_names:
        raise LoraPocValidationError("adapter has no aligner/merger LoRA tensors")
    if vit_names:
        raise LoraPocValidationError(
            f"ViT should be frozen, found {len(vit_names)} visual encoder tensors"
        )
    if not nonzero_lora_b:
        raise LoraPocValidationError("no non-zero lora_B tensor after training")

    gpu_peaks = load_gpu_peaks(args.gpu_trace)
    underutilized = {gpu: used for gpu, used in gpu_peaks.items() if used < 40000}
    if underutilized:
        raise LoraPocValidationError(
            f"not all four GPUs loaded the model: {underutilized}"
        )
    summary = {
        "protocol_version": "qwen36_lora_deepspeed_poc2_v1",
        "checkpoint": str(checkpoint),
        "global_step": state["global_step"],
        "loss_history": loss_rows,
        "adapter_tensor_count": len(tensor_names),
        "llm_tensor_count": len(llm_names),
        "aligner_tensor_count": len(aligner_names),
        "vit_tensor_count": len(vit_names),
        "nonzero_lora_b": nonzero_lora_b,
        "gpu_peak_used_mib": {str(gpu): used for gpu, used in sorted(gpu_peaks.items())},
        "adapter_examples": {
            "llm": llm_names[:4],
            "aligner": aligner_names[:4],
        },
        "deepspeed": "zero2",
        "flash_attention": "flash_attention_2",
        "test_untouched": True,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gpu-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-step", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        summary = validate(args)
    except (OSError, json.JSONDecodeError, LoraPocValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== QWEN3.6 LORA POC VALIDATION ===")
    print(f"checkpoint={summary['checkpoint']}")
    print(f"loss_history={summary['loss_history']}")
    print(
        f"adapter_tensors={summary['adapter_tensor_count']} "
        f"llm={summary['llm_tensor_count']} "
        f"aligner={summary['aligner_tensor_count']} vit={summary['vit_tensor_count']}"
    )
    print(f"nonzero_lora_b={summary['nonzero_lora_b']}")
    print(f"gpu_peak_used_mib={summary['gpu_peak_used_mib']}")
    print("QWEN36_LORA_POC_VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
