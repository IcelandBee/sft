#!/usr/bin/env python3
"""Freeze one canonical Qwen3.5 prompt per readjudicated Test image."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from qwen35_benchmark_common import IMAGE_PLACEHOLDER, NON_THINKING_PREFIX, write_jsonl
except ModuleNotFoundError:  # Imported as scripts.build_vllm_benchmark_requests in tests.
    from scripts.qwen35_benchmark_common import (
        IMAGE_PLACEHOLDER,
        NON_THINKING_PREFIX,
        write_jsonl,
    )


EXPECTED_SHA256 = "c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f"


def build(dataset: Path, output: Path, manifest_path: Path) -> dict:
    source = dataset.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"dataset sha256 mismatch: {digest}")
    source_rows = [
        json.loads(line)
        for line in source.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(source_rows) != 241:
        raise ValueError(f"expected 241 rows, got {len(source_rows)}")

    requests = []
    labels = Counter()
    seen_images = set()
    for index, row in enumerate(source_rows):
        images = row.get("images")
        messages = row.get("messages")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"row {index}: expected one image")
        image = images[0]
        if image in seen_images:
            raise ValueError(f"row {index}: duplicate image {image}")
        seen_images.add(image)
        if not Path(image).is_file():
            raise ValueError(f"row {index}: missing image {image}")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"row {index}: invalid message roles")
        user = messages[1]["content"]
        if user.count("<image>") != 1:
            raise ValueError(f"row {index}: expected one image placeholder")
        gold = json.loads(messages[2]["content"])
        labels[gold["decision"]] += 1
        user_with_placeholder = user.replace("<image>", IMAGE_PLACEHOLDER, 1)
        prompt = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{user_with_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n{NON_THINKING_PREFIX}"
        )
        requests.append(
            {
                "row_index": index,
                "image": image,
                "prompt": prompt,
                "label": messages[2]["content"],
                "messages": messages[:2],
                "dataset": str(dataset),
            }
        )
    if labels != {"BAD": 66, "GOOD": 175}:
        raise ValueError(f"unexpected label counts: {dict(labels)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, requests)
    request_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "protocol_version": "qwen35_vllm_native_prompt_v1",
        "dataset": str(dataset),
        "dataset_sha256": digest,
        "requests": str(output),
        "requests_sha256": request_sha,
        "rows": len(requests),
        "labels": dict(labels),
        "image_placeholder": IMAGE_PLACEHOLDER,
        "non_thinking_prefix": NON_THINKING_PREFIX,
        "max_new_tokens": 128,
        "temperature": 0.0,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.output, args.manifest), ensure_ascii=False, sort_keys=True))
    print("VLLM_BENCHMARK_REQUEST_BUILD: PASS")


if __name__ == "__main__":
    main()
