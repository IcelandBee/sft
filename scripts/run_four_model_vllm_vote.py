#!/usr/bin/env python3
"""Run four merged Qwen3.5 models with vLLM and emit five GOOD/BAD results.

Input may be JSONL (``{"id": "x", "image": "/path/a.jpg"}``), ms-swift
JSONL with one item in ``images``, or a plain text file containing one image path
per line. The output is JSONL and intentionally stores no generated free text.

Examples:
  # One accelerator, four models loaded sequentially
  python run_four_model_vllm_vote.py --input images.jsonl --output votes.jsonl

  # Four GPUs, one model per GPU
  python run_four_model_vllm_vote.py --input images.jsonl --output votes.jsonl \
      --devices 4,5,6,7
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterable


MODEL_ROOT = Path(
    "/home/data/h30082292/DATA_71/h30082292/models/"
    "qwen35_27b_four_merged_models_v1"
)
DEFAULT_MODELS = (
    ("e1_1248", MODEL_ROOT / "qwen35-27b-e1-1248-merged-bf16"),
    ("e2_1248", MODEL_ROOT / "qwen35-27b-e2-1248-merged-bf16"),
    ("e5_780_recall", MODEL_ROOT / "qwen35-27b-e5-780-recall-merged-bf16"),
    ("e5_975_balanced", MODEL_ROOT / "qwen35-27b-e5-975-balanced-merged-bf16"),
)
MODEL_NAMES = tuple(name for name, _ in DEFAULT_MODELS)

SYSTEM_PROMPT = (
    "你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。"
    "严格只输出指定JSON，不要添加分析、解释或Markdown。"
)
USER_PROMPT = (
    "<image>\n检查这张图片。输出decision、categories和reasons。"
    "decision只能是GOOD或BAD。"
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
RESIZE_FACTOR = 28
MIN_IMAGE_PIXELS = 4 * RESIZE_FACTOR * RESIZE_FACTOR
MAX_IMAGE_PIXELS = 1024 * RESIZE_FACTOR * RESIZE_FACTOR


class BatchInferenceError(ValueError):
    """Raised when an input, prediction, or model contract is invalid."""


def parse_named_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("model must use NAME=/path/to/merged-model")
    return name, Path(path)


def build_prompt() -> str:
    user = USER_PROMPT.replace("<image>", IMAGE_PLACEHOLDER, 1)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{NON_THINKING_PREFIX}"
    )


def _input_image(row: dict, line_number: int) -> str:
    image = row.get("image")
    if isinstance(image, str) and image:
        return image
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1 and isinstance(images[0], str):
        return images[0]
    raise BatchInferenceError(
        f"input line {line_number}: expected 'image' or a one-item 'images' list"
    )


def load_input(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                source = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BatchInferenceError(
                    f"input line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(source, dict):
                raise BatchInferenceError(f"input line {line_number}: expected object")
            image_text = _input_image(source, line_number)
            item_id = str(source.get("id", line_number))
        else:
            image_text = stripped
            item_id = str(line_number)
        image = Path(image_text).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise BatchInferenceError(f"input line {line_number}: image not found: {image}")
        if item_id in seen_ids:
            raise BatchInferenceError(f"input line {line_number}: duplicate id: {item_id}")
        seen_ids.add(item_id)
        rows.append({"id": item_id, "image": str(image), "prompt": build_prompt()})
    if not rows:
        raise BatchInferenceError("input contains no images")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def parse_decision(text: str) -> str:
    stripped = text.strip()
    if stripped in {"GOOD", "BAD"}:
        return stripped
    decoder = json.JSONDecoder()
    for offset, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("decision") in {"GOOD", "BAD"}:
            return payload["decision"]
    raise BatchInferenceError("model response contains no valid GOOD/BAD decision")


def smart_resize(height: int, width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise BatchInferenceError(f"invalid image size: {width}x{height}")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise BatchInferenceError(f"image aspect ratio exceeds 200: {width}x{height}")
    resized_h = max(RESIZE_FACTOR, round(height / RESIZE_FACTOR) * RESIZE_FACTOR)
    resized_w = max(RESIZE_FACTOR, round(width / RESIZE_FACTOR) * RESIZE_FACTOR)
    area = resized_h * resized_w
    if area > MAX_IMAGE_PIXELS:
        beta = math.sqrt((height * width) / MAX_IMAGE_PIXELS)
        resized_h = max(
            RESIZE_FACTOR, math.floor(height / beta / RESIZE_FACTOR) * RESIZE_FACTOR
        )
        resized_w = max(
            RESIZE_FACTOR, math.floor(width / beta / RESIZE_FACTOR) * RESIZE_FACTOR
        )
    elif area < MIN_IMAGE_PIXELS:
        beta = math.sqrt(MIN_IMAGE_PIXELS / (height * width))
        resized_h = max(
            RESIZE_FACTOR, math.ceil(height * beta / RESIZE_FACTOR) * RESIZE_FACTOR
        )
        resized_w = max(
            RESIZE_FACTOR, math.ceil(width * beta / RESIZE_FACTOR) * RESIZE_FACTOR
        )
    return resized_h, resized_w


def load_resized_image(path: str):
    # Kept inside the worker-facing function so orchestration does not require Pillow.
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    target_h, target_w = smart_resize(image.height, image.width)
    if (image.width, image.height) != (target_w, target_h):
        image = image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return image


def run_worker(args: argparse.Namespace) -> None:
    # This is the only backend-specific section colleagues need to replace for NPU.
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.worker_requests)
    start = time.perf_counter()
    llm = LLM(
        model=str(args.worker_model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": MIN_IMAGE_PIXELS,
            "max_pixels": MAX_IMAGE_PIXELS,
        },
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=42,
    )
    inputs = [
        {
            "prompt": row["prompt"],
            "multi_modal_data": {"image": load_resized_image(row["image"])},
        }
        for row in rows
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=42)
    outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(rows):
        raise BatchInferenceError(
            f"{args.worker_name}: expected {len(rows)} outputs, got {len(outputs)}"
        )
    decisions = []
    for row, output in zip(rows, outputs):
        try:
            decision = parse_decision(output.outputs[0].text)
        except BatchInferenceError as exc:
            raise BatchInferenceError(
                f"{args.worker_name}: invalid response for id={row['id']} image={row['image']}"
            ) from exc
        decisions.append({"id": row["id"], "decision": decision})
    write_jsonl(args.worker_output, decisions)
    elapsed = time.perf_counter() - start
    print(
        f"MODEL_COMPLETE name={args.worker_name} rows={len(rows)} "
        f"seconds={elapsed:.3f} images_per_second={len(rows) / elapsed:.3f}",
        flush=True,
    )


def _run_subprocess(
    args: argparse.Namespace,
    name: str,
    model: Path,
    requests: Path,
    result: Path,
    device: str | None,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-name",
        name,
        "--worker-model",
        str(model),
        "--worker-requests",
        str(requests),
        "--worker-output",
        str(result),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    environment = os.environ.copy()
    environment["VLLM_USE_V1"] = "1"
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    print(f"MODEL_START name={name} model={model} device={device or 'inherited'}", flush=True)
    subprocess.run(command, check=True, env=environment)


def combine_results(requests: list[dict], model_results: dict[str, list[dict]]) -> list[dict]:
    if tuple(model_results) != MODEL_NAMES:
        raise BatchInferenceError(f"unexpected model order: {tuple(model_results)}")
    combined: list[dict] = []
    for index, request in enumerate(requests):
        votes: dict[str, str] = {}
        for name in MODEL_NAMES:
            rows = model_results[name]
            if len(rows) != len(requests) or rows[index].get("id") != request["id"]:
                raise BatchInferenceError(f"{name}: result alignment mismatch at index {index}")
            decision = rows[index].get("decision")
            if decision not in {"GOOD", "BAD"}:
                raise BatchInferenceError(f"{name}: invalid decision at index {index}")
            votes[name] = decision
        ensemble = "BAD" if sum(value == "BAD" for value in votes.values()) >= 2 else "GOOD"
        combined.append(
            {
                "id": request["id"],
                "image": request["image"],
                **votes,
                "ensemble_vote": ensemble,
            }
        )
    return combined


def run_orchestrator(args: argparse.Namespace) -> None:
    models = tuple(args.model or DEFAULT_MODELS)
    if tuple(name for name, _ in models) != MODEL_NAMES:
        raise BatchInferenceError(
            "models must be supplied once each and in this order: " + ",".join(MODEL_NAMES)
        )
    for name, model in models:
        if not (model / "config.json").is_file():
            raise BatchInferenceError(f"{name}: merged model is incomplete: {model}")
    if args.output.exists():
        raise BatchInferenceError(f"output already exists: {args.output}")
    requests = load_input(args.input)
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if devices and len(devices) != len(models):
        raise BatchInferenceError("--devices must contain exactly four device IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        requests_path = staging / "requests.jsonl"
        write_jsonl(requests_path, requests)
        jobs = [
            (
                args,
                name,
                model,
                requests_path,
                staging / f"{name}.jsonl",
                devices[index] if devices else None,
            )
            for index, (name, model) in enumerate(models)
        ]
        if devices:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_run_subprocess, *job) for job in jobs]
                for future in futures:
                    future.result()
        else:
            for job in jobs:
                _run_subprocess(*job)
        model_results = {
            name: load_jsonl(staging / f"{name}.jsonl") for name in MODEL_NAMES
        }
        combined = combine_results(requests, model_results)
        temporary_output = staging / "combined.jsonl"
        write_jsonl(temporary_output, combined)
        temporary_output.replace(args.output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"output={args.output} rows={len(requests)}", flush=True)
    print("FOUR_MODEL_VLLM_VOTE: PASS", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", action="append", type=parse_named_model)
    parser.add_argument(
        "--devices",
        default="",
        help="four comma-separated physical GPU IDs; omit for sequential execution",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--worker-name", choices=MODEL_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-requests", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    worker_values = (
        args.worker_name,
        args.worker_model,
        args.worker_requests,
        args.worker_output,
    )
    if any(value is not None for value in worker_values):
        if not all(value is not None for value in worker_values):
            parser.error("internal worker arguments must be provided together")
        run_worker(args)
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    run_orchestrator(args)


if __name__ == "__main__":
    main()
