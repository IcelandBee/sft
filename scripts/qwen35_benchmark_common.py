#!/usr/bin/env python3
"""Shared, framework-neutral helpers for the Qwen3.5 vLLM benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageOps


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
RESIZE_FACTOR = 28
MIN_IMAGE_PIXELS = 4 * RESIZE_FACTOR * RESIZE_FACTOR
MAX_IMAGE_PIXELS = 1024 * RESIZE_FACTOR * RESIZE_FACTOR


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Resize to Qwen's factor while keeping the configured pixel budget."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"image aspect ratio exceeds 200: {width}x{height}")
    resized_h = max(RESIZE_FACTOR, round(height / RESIZE_FACTOR) * RESIZE_FACTOR)
    resized_w = max(RESIZE_FACTOR, round(width / RESIZE_FACTOR) * RESIZE_FACTOR)
    area = resized_h * resized_w
    if area > MAX_IMAGE_PIXELS:
        beta = math.sqrt((height * width) / MAX_IMAGE_PIXELS)
        resized_h = max(RESIZE_FACTOR, math.floor(height / beta / RESIZE_FACTOR) * RESIZE_FACTOR)
        resized_w = max(RESIZE_FACTOR, math.floor(width / beta / RESIZE_FACTOR) * RESIZE_FACTOR)
    elif area < MIN_IMAGE_PIXELS:
        beta = math.sqrt(MIN_IMAGE_PIXELS / (height * width))
        resized_h = max(RESIZE_FACTOR, math.ceil(height * beta / RESIZE_FACTOR) * RESIZE_FACTOR)
        resized_w = max(RESIZE_FACTOR, math.ceil(width * beta / RESIZE_FACTOR) * RESIZE_FACTOR)
    return resized_h, resized_w


def load_resized_image(path: str) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    target_h, target_w = smart_resize(image.height, image.width)
    if (image.width, image.height) != (target_w, target_h):
        image = image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return image


def load_requests(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def make_raw_result(request: dict, generated_text: str) -> dict:
    response = NON_THINKING_PREFIX + generated_text.strip()
    messages = list(request["messages"])
    messages.append({"role": "assistant", "content": response})
    return {
        "response": response,
        "labels": request["label"],
        "logprobs": None,
        "images": [{"bytes": None, "path": request["image"]}],
        "messages": messages,
        "dataset": request["dataset"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")
