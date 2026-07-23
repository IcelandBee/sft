#!/usr/bin/env python3
"""Encode an E4 two-image PoC with ms-swift and report true token lengths."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


class TokenLengthError(ValueError):
    """Raised when PoC rows, manifest, or encoded outputs are invalid."""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise TokenLengthError(f"cannot read PoC: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TokenLengthError(f"invalid JSON at PoC row {line_number}") from exc
        if not isinstance(value, dict):
            raise TokenLengthError(f"PoC row {line_number} must be an object")
        rows.append(value)
    return rows


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_token_lengths(records: list[dict]) -> dict:
    if not records:
        raise TokenLengthError("no encoded records")
    totals = [record["total_tokens"] for record in records]
    by_type: dict[str, list[int]] = {}
    for record in records:
        by_type.setdefault(record["sample_type"], []).append(record["total_tokens"])
    max_total = max(totals)
    candidates = [2048, 2560, 3072, 3584, 4096, 6144, 8192]
    recommended = next((value for value in candidates if value >= max_total + 64), None)
    return {
        "rows": len(records),
        "total_token_quantiles": {
            "min": min(totals),
            "p50": _quantile(totals, 0.50),
            "p90": _quantile(totals, 0.90),
            "p95": _quantile(totals, 0.95),
            "max": max_total,
        },
        "by_sample_type": {
            name: {
                "rows": len(values),
                "min": min(values),
                "p50": _quantile(values, 0.50),
                "max": max(values),
            }
            for name, values in sorted(by_type.items())
        },
        "rows_exceeding": {
            str(limit): sum(value > limit for value in totals)
            for limit in (2048, 2560, 3072, 4096)
        },
        "poc_recommended_max_length_with_64_token_margin": recommended,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--poc", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        rows = _load_jsonl(args.poc)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != len(rows):
            raise TokenLengthError("PoC and manifest row counts differ")
        try:
            import swift
            from swift import get_processor, get_template
        except ImportError:
            import swift
            from swift.llm import get_processor, get_template

        processor = get_processor(str(args.model))
        template = get_template(
            processor,
            max_length=262144,
            truncation_strategy="raise",
            loss_scale="default+ignore_empty_think",
            add_non_thinking_prefix=True,
        )
        template.set_mode("train")
        encoded_records: list[dict] = []
        for index, (row, sample) in enumerate(zip(rows, samples)):
            encoded = template.encode(row)
            input_ids = encoded.get("input_ids")
            labels = encoded.get("labels")
            if input_ids is None or labels is None or len(input_ids) != len(labels):
                raise TokenLengthError(f"invalid encoded output at row {index}")
            supervised = sum(int(label != -100) for label in labels)
            record = {
                "index": index,
                "sample_type": sample["sample_type"],
                "selection_reason": sample["selection_reason"],
                "total_tokens": len(input_ids),
                "supervised_tokens": supervised,
            }
            encoded_records.append(record)
            print(
                f"row={index} type={record['sample_type']} "
                f"tokens={record['total_tokens']} supervised={supervised} "
                f"reason={record['selection_reason']}"
            )
        summary = summarize_token_lengths(encoded_records)
        summary.update(
            {
                "protocol_version": "e4_two_image_token_length_v1",
                "model": str(args.model),
                "swift_version": swift.__version__,
                "image_max_token_num": int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024")),
                "test_untouched": True,
                "dev_untouched": True,
                "records": encoded_records,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TokenLengthError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== E4 TWO-IMAGE TOKEN LENGTHS ===")
    print(f"quantiles={summary['total_token_quantiles']}")
    print(f"by_type={summary['by_sample_type']}")
    print(f"rows_exceeding={summary['rows_exceeding']}")
    print(
        "poc_recommended_max_length="
        f"{summary['poc_recommended_max_length_with_64_token_margin']}"
    )
    print(f"summary={args.output}")
    print("E4_TOKEN_LENGTH_CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
