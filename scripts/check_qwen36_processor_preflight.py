#!/usr/bin/env python3
"""Run an offline Qwen3.6 processor/template preflight on 20 E5 rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


EXPECTED_STRATA = {
    "T1_GOOD": 5,
    "T1_BAD": 5,
    "T2_BAD": 5,
    "T3_GOOD": 5,
}
NON_THINKING_PREFIX = "<think>\n\n</think>"


class ProcessorPreflightError(ValueError):
    """Raised when the E5 data or Qwen3.6 template violates the protocol."""


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProcessorPreflightError(
                f"invalid JSON at row {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ProcessorPreflightError(f"row {line_number} is not an object")
        rows.append(row)
    return rows


def assistant_payload(row: dict) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProcessorPreflightError("row has no messages")
    assistant = messages[-1]
    if assistant.get("role") != "assistant":
        raise ProcessorPreflightError("last message is not assistant")
    try:
        payload = json.loads(assistant["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProcessorPreflightError("assistant content is not strict JSON") from exc
    if set(payload) != {"decision", "categories", "reasons"}:
        raise ProcessorPreflightError(f"unexpected assistant keys: {sorted(payload)}")
    if payload["decision"] not in {"GOOD", "BAD"}:
        raise ProcessorPreflightError(f"invalid decision: {payload['decision']}")
    if not isinstance(payload["categories"], list) or not isinstance(
        payload["reasons"], list
    ):
        raise ProcessorPreflightError("categories/reasons must be lists")
    return payload


def classify_stratum(row: dict) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) not in {1, 2}:
        raise ProcessorPreflightError("images must contain one or two paths")
    decision = assistant_payload(row)["decision"]
    if len(images) == 1:
        return f"T1_{decision}"
    return "T2_BAD" if decision == "BAD" else "T3_GOOD"


def _evenly_spaced(items: list[tuple[int, dict]], count: int) -> list[tuple[int, dict]]:
    if len(items) < count:
        raise ProcessorPreflightError(
            f"need {count} rows in stratum, found {len(items)}"
        )
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))]
        for index in range(count)
    ]


def select_poc_rows(
    rows: list[dict], counts: dict[str, int] | None = None
) -> list[tuple[int, str, dict]]:
    requested = EXPECTED_STRATA if counts is None else counts
    strata: dict[str, list[tuple[int, dict]]] = {name: [] for name in requested}
    for index, row in enumerate(rows):
        stratum = classify_stratum(row)
        if stratum in strata:
            strata[stratum].append((index, row))
    selected: list[tuple[int, str, dict]] = []
    for stratum, count in requested.items():
        selected.extend(
            (index, stratum, row)
            for index, row in _evenly_spaced(strata[stratum], count)
        )
    return selected


def validate_e5_distribution(rows: list[dict]) -> Counter:
    if len(rows) != 12472:
        raise ProcessorPreflightError(f"expected 12472 rows, got {len(rows)}")
    distribution = Counter(classify_stratum(row) for row in rows)
    expected = Counter(
        {
            "T1_GOOD": 6074,
            "T1_BAD": 3904,
            "T2_BAD": 1247,
            "T3_GOOD": 1247,
        }
    )
    if distribution != expected:
        raise ProcessorPreflightError(
            f"unexpected E5 distribution: {dict(distribution)}"
        )
    return distribution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=3072)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        rows = load_jsonl(args.train)
        distribution = validate_e5_distribution(rows)
        selected = select_poc_rows(rows)

        from transformers import AutoConfig

        try:
            import swift
            from swift import get_processor, get_template
        except ImportError:
            import swift
            from swift.llm import get_processor, get_template

        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        processor = get_processor(str(args.model))
        template = get_template(
            processor,
            max_length=262144,
            truncation_strategy="raise",
            loss_scale="default+ignore_empty_think",
            add_non_thinking_prefix=True,
        )
        template.set_mode("train")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        records: list[dict] = []
        missing_images: list[str] = []
        for source_index, stratum, row in selected:
            for image in row["images"]:
                if not Path(image).is_file():
                    missing_images.append(image)
            encoded = template.encode(row)
            input_ids = encoded.get("input_ids")
            labels = encoded.get("labels")
            if input_ids is None or labels is None or len(input_ids) != len(labels):
                raise ProcessorPreflightError(
                    f"invalid encoded result for source row {source_index}"
                )
            supervised_tokens = sum(label != -100 for label in labels)
            if supervised_tokens <= 0:
                raise ProcessorPreflightError(
                    f"no supervised tokens for source row {source_index}"
                )
            decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
            if NON_THINKING_PREFIX not in decoded:
                raise ProcessorPreflightError(
                    f"missing non-thinking prefix at source row {source_index}"
                )
            record = {
                "source_index": source_index,
                "stratum": stratum,
                "decision": assistant_payload(row)["decision"],
                "image_count": len(row["images"]),
                "total_tokens": len(input_ids),
                "image_tokens": sum(token == image_token_id for token in input_ids),
                "supervised_tokens": supervised_tokens,
                "within_max_length": len(input_ids) <= args.max_length,
            }
            records.append(record)
            print(
                f"row={source_index} stratum={stratum} images={record['image_count']} "
                f"tokens={record['total_tokens']} image_tokens={record['image_tokens']} "
                f"supervised={supervised_tokens}"
            )
        if missing_images:
            raise ProcessorPreflightError(
                f"missing selected images: {len(missing_images)}; first={missing_images[0]}"
            )

        summary = {
            "protocol_version": "qwen36_e5_processor_preflight_v1",
            "model": str(args.model),
            "model_type": config.model_type,
            "architectures": config.architectures,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "template_class": type(template).__name__,
            "swift_version": swift.__version__,
            "image_max_token_num": int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024")),
            "max_length": args.max_length,
            "train_distribution": dict(distribution),
            "selected_distribution": dict(Counter(r["stratum"] for r in records)),
            "selected_rows": len(records),
            "selected_image_inputs": sum(r["image_count"] for r in records),
            "max_observed_tokens": max(r["total_tokens"] for r in records),
            "rows_over_max_length": sum(not r["within_max_length"] for r in records),
            "non_thinking_prefix_verified": True,
            "strict_json_verified": True,
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if summary["rows_over_max_length"]:
            raise ProcessorPreflightError(
                f"{summary['rows_over_max_length']} selected rows exceed {args.max_length}"
            )
    except (OSError, ProcessorPreflightError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=== QWEN3.6 PROCESSOR PREFLIGHT ===")
    print(f"model_type={summary['model_type']}")
    print(f"architectures={summary['architectures']}")
    print(f"processor_class={summary['processor_class']}")
    print(f"template_class={summary['template_class']}")
    print(f"selected_distribution={summary['selected_distribution']}")
    print(f"selected_image_inputs={summary['selected_image_inputs']}")
    print(f"max_observed_tokens={summary['max_observed_tokens']}")
    print(f"summary={args.output}")
    print("QWEN36_PROCESSOR_PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
