#!/usr/bin/env python3
"""Benchmark merged Qwen3.5 with the untouched native vLLM environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from vllm import LLM, SamplingParams

from qwen35_benchmark_common import (
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_PIXELS,
    load_requests,
    load_resized_image,
    make_raw_result,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    args = parser.parse_args()

    process_start = time.perf_counter()
    requests = load_requests(args.requests)

    load_start = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
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
    model_load_seconds = time.perf_counter() - load_start

    inference_start = time.perf_counter()
    prepare_start = inference_start
    inputs = [
        {
            "prompt": request["prompt"],
            "multi_modal_data": {"image": load_resized_image(request["image"])},
        }
        for request in requests
    ]
    request_prepare_seconds = time.perf_counter() - prepare_start

    sampling = SamplingParams(temperature=0.0, max_tokens=128, seed=42)
    generation_start = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=True)
    generation_seconds = time.perf_counter() - generation_start
    if len(outputs) != len(requests):
        raise RuntimeError(f"expected {len(requests)} outputs, got {len(outputs)}")

    raw_rows = []
    prompt_tokens = 0
    completion_tokens = 0
    for request, output in zip(requests, outputs):
        candidate = output.outputs[0]
        raw_rows.append(make_raw_result(request, candidate.text))
        prompt_tokens += len(output.prompt_token_ids)
        completion_tokens += len(candidate.token_ids)
    inference_seconds = time.perf_counter() - inference_start
    total_seconds = time.perf_counter() - process_start

    write_jsonl(args.result, raw_rows)
    stats = {
        "backend": "vllm",
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
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print("VLLM_MERGED_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
