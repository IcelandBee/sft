#!/usr/bin/env python3
"""Run fixed E1-compatible Dev inference for explicit checkpoints on GPUs 4-7."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

if __package__:
    from .evaluate_e1_dev import SYSTEM_PROMPT, USER_PROMPT, validate_payload
    from .select_e1_checkpoint import (
        EXPECTED_STEPS,
        SelectionError,
        normalize_expected_steps,
        run_selection,
    )
else:
    from evaluate_e1_dev import SYSTEM_PROMPT, USER_PROMPT, validate_payload  # type: ignore
    from select_e1_checkpoint import (  # type: ignore
        EXPECTED_STEPS,
        SelectionError,
        normalize_expected_steps,
        run_selection,
    )


DEFAULT_MODEL = Path("/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B")
DEFAULT_CHECKPOINT_ROOT = Path(
    "/home/data/h30082292/data/pose/artifact_detection_training/runs/"
    "e1_broad_clean_r16_e4_v1/v0-20260717-185936"
)
DEFAULT_DEV = Path(
    "/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/"
    "e1_broad_clean_json_v1/dev.jsonl"
)
DEFAULT_GPUS = (4, 5, 6, 7)
class RunnerError(ValueError):
    """Raised when a batch run violates its fixed execution contract."""


def build_swift_command(
    model: Path, adapter: Path, dev: Path, result: Path
) -> list[str]:
    """Build the fixed ms-swift command for one checkpoint."""
    return [
        "swift", "infer",
        "--model", str(model),
        "--adapters", str(adapter),
        "--val_dataset", str(dev),
        "--val_dataset_shuffle", "false",
        "--strict", "true",
        "--lazy_tokenize", "true",
        "--add_non_thinking_prefix", "true",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "flash_attention_2",
        "--infer_backend", "transformers",
        "--max_new_tokens", "128",
        "--temperature", "0",
        "--stream", "false",
        "--max_batch_size", "1",
        "--write_batch_size", "20",
        "--dataset_num_proc", "1",
        "--seed", "42",
        "--data_seed", "42",
        "--result_path", str(result),
    ]


def assign_jobs(
    steps: tuple[int, ...], gpus: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Assign checkpoint steps round-robin to unique physical GPUs."""
    steps = normalize_expected_steps(steps)
    if not gpus:
        raise RunnerError("at least one GPU is required")
    if len(set(gpus)) != len(gpus):
        raise RunnerError("GPU identifiers must be unique")
    return [(step, gpus[index % len(gpus)]) for index, step in enumerate(steps)]


def build_manifest(
    model: Path,
    checkpoint_root: Path,
    dev: Path,
    output_root: Path,
    steps: tuple[int, ...],
    gpus: tuple[int, ...],
    dev_sha256: str,
) -> dict:
    """Build a serializable audit manifest without executing subprocesses."""
    jobs: list[dict] = []
    for step, gpu in assign_jobs(steps, gpus):
        checkpoint_dir = Path(output_root) / f"checkpoint-{step}"
        adapter = Path(checkpoint_root) / f"checkpoint-{step}"
        result = checkpoint_dir / "raw-result.jsonl"
        jobs.append(
            {
                "step": step,
                "gpu": gpu,
                "dev": str(dev),
                "adapter": str(adapter),
                "result": str(result),
                "log": str(checkpoint_dir / "infer.log"),
                "evaluation_dir": str(checkpoint_dir / "evaluation"),
                "command": build_swift_command(model, adapter, dev, result),
            }
        )
    return {
        "protocol_version": "e1_dev_batch_inference_v1",
        "expected_steps": list(steps),
        "model": str(model),
        "checkpoint_root": str(checkpoint_root),
        "dev": str(dev),
        "dev_sha256": dev_sha256,
        "output_root": str(output_root),
        "image_max_token_num": 1024,
        "prompt_sha256": {
            "system": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "user": hashlib.sha256(USER_PROMPT.encode("utf-8")).hexdigest(),
        },
        "gpus": list(gpus),
        "jobs": jobs,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write a manifest exclusively so an earlier audit cannot be overwritten."""
    path = Path(path)
    if path.exists():
        raise RunnerError(f"manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def validate_dev_file(
    path: Path,
    expected_count: int = 200,
    expected_good: int = 149,
    expected_bad: int = 51,
) -> dict:
    """Validate the fixed ms-swift Dev data and return its counts and digest."""
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"Dev file does not exist: {path}")
    source = path.read_bytes()
    lines = [line for line in source.decode("utf-8-sig").splitlines() if line.strip()]
    if len(lines) != expected_count:
        raise RunnerError(f"Dev must contain {expected_count} rows, got {len(lines)}")

    decisions: Counter[str] = Counter()
    seen_images: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"invalid Dev JSON at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"images", "messages"}:
            raise RunnerError(f"invalid Dev fields at line {line_number}")
        images = row["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise RunnerError(f"Dev row {line_number} must contain one image path")
        image = images[0]
        if image in seen_images:
            raise RunnerError(
                f"duplicate Dev image at lines {seen_images[image]} and {line_number}: {image}"
            )
        seen_images[image] = line_number
        if not Path(image).is_file():
            raise RunnerError(f"Dev image does not exist at line {line_number}: {image}")

        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [message.get("role") for message in messages]
            != ["system", "user", "assistant"]
        ):
            raise RunnerError(f"invalid Dev messages at line {line_number}")
        if messages[0].get("content") != SYSTEM_PROMPT:
            raise RunnerError(f"system prompt drift at Dev line {line_number}")
        if messages[1].get("content") != USER_PROMPT:
            raise RunnerError(f"user prompt drift at Dev line {line_number}")
        try:
            gold = json.loads(messages[2].get("content"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"invalid gold JSON at Dev line {line_number}") from exc
        schema_error = validate_payload(gold)
        if schema_error is not None:
            raise RunnerError(
                f"invalid gold schema at Dev line {line_number}: {schema_error}"
            )
        decisions[gold["decision"]] += 1

    if decisions != Counter({"GOOD": expected_good, "BAD": expected_bad}):
        raise RunnerError(
            f"Dev label counts must be GOOD={expected_good}, BAD={expected_bad}; "
            f"got {dict(decisions)}"
        )
    return {
        "counts": {"GOOD": decisions["GOOD"], "BAD": decisions["BAD"], "total": len(lines)},
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def _check_checkpoint_root(
    checkpoint_root: Path, expected_steps: tuple[int, ...] = EXPECTED_STEPS
) -> None:
    expected_steps = normalize_expected_steps(expected_steps)
    for step in expected_steps:
        checkpoint = Path(checkpoint_root) / f"checkpoint-{step}"
        if not (checkpoint / "adapter_config.json").is_file():
            raise RunnerError(f"missing adapter_config.json: {checkpoint}")
        if not list(checkpoint.glob("adapter_model*.safetensors")):
            raise RunnerError(f"missing adapter weights: {checkpoint}")


def _check_gpus(gpus: tuple[int, ...]) -> dict[int, int]:
    free_by_gpu: dict[int, int] = {}
    for gpu in gpus:
        completed = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RunnerError(f"cannot query GPU {gpu}: {completed.stderr.strip()}")
        try:
            free = int(completed.stdout.strip())
        except ValueError as exc:
            raise RunnerError(f"invalid free-memory result for GPU {gpu}") from exc
        if free < 70000:
            raise RunnerError(f"GPU {gpu} has only {free} MiB free")
        free_by_gpu[gpu] = free
    return free_by_gpu


def preflight(
    model: Path,
    checkpoint_root: Path,
    dev: Path,
    output_root: Path,
    gpus: tuple[int, ...],
    expected_steps: tuple[int, ...] = EXPECTED_STEPS,
) -> dict:
    """Run every read-only gate required before dry-run or execution."""
    if shutil.which("swift") is None:
        raise RunnerError("swift executable is not available")
    try:
        import swift
        swift_version = str(swift.__version__)
    except (ImportError, AttributeError) as exc:
        raise RunnerError("cannot determine installed swift version") from exc
    if not (Path(model) / "config.json").is_file():
        raise RunnerError(f"model config does not exist: {model}")
    expected_steps = normalize_expected_steps(expected_steps)
    _check_checkpoint_root(checkpoint_root, expected_steps)
    if Path(output_root).exists():
        raise RunnerError(f"output root already exists: {output_root}")
    dev_summary = validate_dev_file(dev)
    free_by_gpu = _check_gpus(gpus)
    return {
        "dev": dev_summary,
        "gpu_free_mib": free_by_gpu,
        "swift_version": swift_version,
        "expected_steps": list(expected_steps),
    }


def _run_gpu_jobs(gpu: int, jobs: list[dict], evaluator: Path) -> list[int]:
    completed_steps: list[int] = []
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "IMAGE_MAX_TOKEN_NUM": "1024",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    for job in jobs:
        step = job["step"]
        checkpoint_dir = Path(job["result"]).parent
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
        log_path = Path(job["log"])
        status_path = checkpoint_dir / "job-status.json"
        status = {
            "step": step,
            "gpu": gpu,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "state": "running",
            "command": job["command"],
        }
        print(f"LAUNCH checkpoint-{step} on GPU {gpu}", flush=True)
        try:
            with log_path.open("x", encoding="utf-8", newline="\n") as log_stream:
                completed = subprocess.run(
                    job["command"],
                    check=False,
                    env=env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            status["infer_returncode"] = completed.returncode
            result_path = Path(job["result"])
            if (
                completed.returncode != 0
                or not result_path.is_file()
                or result_path.stat().st_size == 0
            ):
                raise RunnerError(
                    f"checkpoint-{step} inference failed on GPU {gpu}; see {log_path}"
                )

            evaluation_dir = Path(job["evaluation_dir"])
            eval_log = checkpoint_dir / "evaluate.log"
            eval_command = [
                sys.executable,
                str(evaluator),
                "--result", str(result_path),
                "--output-dir", str(evaluation_dir),
                "--expected-count", "200",
                "--checkpoint-step", str(step),
                "--expected-dev", job["dev"],
            ]
            with eval_log.open("x", encoding="utf-8", newline="\n") as log_stream:
                evaluated = subprocess.run(
                    eval_command,
                    check=False,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            status["evaluation_returncode"] = evaluated.returncode
            if evaluated.returncode != 0:
                raise RunnerError(
                    f"checkpoint-{step} evaluation failed; see {eval_log}"
                )
            status["state"] = "completed"
            completed_steps.append(step)
            print(f"COMPLETE checkpoint-{step} on GPU {gpu}", flush=True)
        except Exception as exc:
            status["state"] = "failed"
            status["error"] = str(exc)
            raise
        finally:
            status["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_manifest(status_path, status)
    return completed_steps


def run_batch(
    manifest: dict,
    output_root: Path,
    expected_steps: tuple[int, ...] = EXPECTED_STEPS,
) -> dict:
    """Execute one fixed two-checkpoint queue per GPU, then select by Dev metrics."""
    expected_steps = normalize_expected_steps(expected_steps)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    write_manifest(output_root / "run-manifest.json", manifest)
    evaluator = Path(__file__).with_name("evaluate_e1_dev.py").resolve()
    if not evaluator.is_file():
        raise RunnerError(f"evaluator script does not exist: {evaluator}")

    by_gpu: dict[int, list[dict]] = defaultdict(list)
    for job in manifest["jobs"]:
        by_gpu[job["gpu"]].append(job)

    errors: list[str] = []
    completed_steps: list[int] = []
    with ThreadPoolExecutor(max_workers=len(by_gpu)) as executor:
        futures = {
            executor.submit(_run_gpu_jobs, gpu, jobs, evaluator): gpu
            for gpu, jobs in sorted(by_gpu.items())
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                completed_steps.extend(future.result())
            except Exception as exc:
                errors.append(f"GPU {gpu}: {exc}")
    if errors:
        raise RunnerError("; ".join(errors))
    if sorted(completed_steps) != list(expected_steps):
        raise RunnerError(f"incomplete checkpoint set: {sorted(completed_steps)}")
    return run_selection(
        output_root,
        output_root / "checkpoint-summary.json",
        expected_steps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(DEFAULT_GPUS))
    parser.add_argument("--steps", nargs="+", type=int, default=list(EXPECTED_STEPS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpus = tuple(args.gpus)
    steps = tuple(args.steps)
    try:
        checked = preflight(
            args.model,
            args.checkpoint_root,
            args.dev,
            args.output_root,
            gpus,
            steps,
        )
        manifest = build_manifest(
            args.model,
            args.checkpoint_root,
            args.dev,
            args.output_root,
            steps,
            gpus,
            checked["dev"]["sha256"],
        )
        manifest["preflight"] = checked
        manifest["prepared_utc"] = datetime.now(timezone.utc).isoformat()
        if args.dry_run:
            manifest_path = args.manifest_path or args.output_root.with_name(
                args.output_root.name + "-dry-run-manifest.json"
            )
            write_manifest(manifest_path, manifest)
            print(f"DRY_RUN_CHECK: PASS\nManifest: {manifest_path}")
            return 0
        summary = run_batch(manifest, args.output_root, steps)
    except (RunnerError, SelectionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["test_unlocked"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
