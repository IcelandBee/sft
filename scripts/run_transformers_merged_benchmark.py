#!/usr/bin/env python3
"""Benchmark merged Qwen3.5 with native Transformers, one image at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from qwen35_benchmark_common import load_requests, load_resized_image, make_raw_result, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()

    process_start = time.perf_counter()
    requests = load_requests(args.requests)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    model_load_seconds = time.perf_counter() - load_start

    raw_rows = []
    prompt_tokens = 0
    completion_tokens = 0
    request_prepare_seconds = 0.0
    generation_seconds = 0.0
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for index, request in enumerate(requests, start=1):
            prepare_start = time.perf_counter()
            image = load_resized_image(request["image"])
            inputs = processor(
                text=[request["prompt"]],
                images=[image],
                return_tensors="pt",
                padding=False,
            ).to(model.device)
            request_prepare_seconds += time.perf_counter() - prepare_start
            generation_start = time.perf_counter()
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
            generation_seconds += time.perf_counter() - generation_start
            input_length = inputs["input_ids"].shape[-1]
            output_ids = generated[0, input_length:]
            text = processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            prompt_tokens += input_length
            completion_tokens += output_ids.numel()
            raw_rows.append(make_raw_result(request, text))
            if index % 10 == 0 or index == len(requests):
                print(f"TRANSFORMERS_PROGRESS {index}/{len(requests)}", flush=True)
    inference_seconds = time.perf_counter() - inference_start
    total_seconds = time.perf_counter() - process_start

    write_jsonl(args.result, raw_rows)
    stats = {
        "backend": "transformers",
        "model": str(args.model),
        "num_samples": len(requests),
        "model_load_seconds": model_load_seconds,
        "request_prepare_seconds": request_prepare_seconds,
        "generation_seconds": generation_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "seconds_per_image": inference_seconds / len(requests),
        "samples_per_second": len(requests) / inference_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print("TRANSFORMERS_MERGED_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
