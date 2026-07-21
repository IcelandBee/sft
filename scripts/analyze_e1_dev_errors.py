#!/usr/bin/env python3
"""Audit E1 Dev FN/FP records and summarize category-level recall."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

try:
    from .evaluate_e1_dev import validate_payload
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_e1_dev import validate_payload  # type: ignore


class ErrorAnalysisError(ValueError):
    """Raised when Dev, parsed predictions, or metrics do not align."""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ErrorAnalysisError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ErrorAnalysisError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _gold_payload(row: dict, index: int) -> tuple[str, dict]:
    images = row.get("images")
    messages = row.get("messages")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ErrorAnalysisError(f"invalid Dev image at row {index}")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ErrorAnalysisError(f"invalid Dev messages at row {index}")
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ErrorAnalysisError(f"missing Dev assistant gold at row {index}")
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ErrorAnalysisError(f"Dev gold at row {index} must be a JSON string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ErrorAnalysisError(f"invalid Dev gold JSON at row {index}") from exc
    schema_error = validate_payload(payload)
    if schema_error is not None:
        raise ErrorAnalysisError(
            f"Dev gold at row {index} violates schema: {schema_error}"
        )
    return images[0], payload


def _ranked_counts(counter: Counter[str]) -> list[dict]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def analyze_rows(
    dev_rows: list[dict],
    parsed_rows: list[dict],
    *,
    checkpoint_step: int,
    expected_count: int = 200,
) -> tuple[dict, list[dict], list[dict]]:
    """Return an auditable summary plus ordered FN and FP records."""
    if len(dev_rows) != expected_count:
        raise ErrorAnalysisError(
            f"expected {expected_count} Dev rows, got {len(dev_rows)}"
        )
    if len(parsed_rows) != expected_count:
        raise ErrorAnalysisError(
            f"expected {expected_count} parsed rows, got {len(parsed_rows)}"
        )

    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    fn_rows: list[dict] = []
    fp_rows: list[dict] = []
    fn_categories: Counter[str] = Counter()
    fn_reasons: Counter[str] = Counter()
    fp_categories: Counter[str] = Counter()
    fp_reasons: Counter[str] = Counter()
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"support": 0, "detected": 0, "missed": 0}
    )
    bad_other_only = {"support": 0, "detected": 0, "missed": 0}

    for offset, (dev, parsed) in enumerate(zip(dev_rows, parsed_rows), start=1):
        image_path, gold = _gold_payload(dev, offset)
        if parsed.get("index") != offset - 1:
            raise ErrorAnalysisError(f"parsed index mismatch at row {offset}")
        if parsed.get("image_path") != image_path:
            raise ErrorAnalysisError(f"image mismatch at row {offset}: {image_path}")

        schema_valid = parsed.get("schema_valid") is True
        predicted = parsed.get("predicted_decision")
        prediction_payload = parsed.get("payload")
        if schema_valid:
            if predicted not in {"GOOD", "BAD"}:
                raise ErrorAnalysisError(f"invalid parsed decision at row {offset}")
            if not isinstance(prediction_payload, dict):
                raise ErrorAnalysisError(f"missing parsed payload at row {offset}")
        else:
            predicted = None

        gold_bad = gold["decision"] == "BAD"
        detected = schema_valid and predicted == "BAD"
        is_fn = False
        is_fp = False
        if not schema_valid and gold_bad:
            confusion["fn"] += 1
            is_fn = True
        elif not schema_valid:
            confusion["fp"] += 1
            is_fp = True
        elif gold_bad and detected:
            confusion["tp"] += 1
        elif gold_bad:
            confusion["fn"] += 1
            is_fn = True
        elif detected:
            confusion["fp"] += 1
            is_fp = True
        else:
            confusion["tn"] += 1

        if gold_bad:
            for category in gold["categories"]:
                category_stats[category]["support"] += 1
                category_stats[category]["detected" if detected else "missed"] += 1
            if set(gold["categories"]) == {"其他"}:
                bad_other_only["support"] += 1
                bad_other_only["detected" if detected else "missed"] += 1

        record = {
            "row": offset,
            "image_path": image_path,
            "gold": gold,
            "prediction": prediction_payload,
            "predicted_decision": predicted,
            "schema_valid": schema_valid,
            "error_code": parsed.get("error_code"),
        }
        if is_fn:
            fn_rows.append(record)
            fn_categories.update(gold["categories"])
            fn_reasons.update(gold["reasons"])
        elif is_fp:
            fp_rows.append(record)
            if isinstance(prediction_payload, dict):
                fp_categories.update(prediction_payload["categories"])
                fp_reasons.update(prediction_payload["reasons"])

    category_performance = []
    for category, counts in category_stats.items():
        item = {"category": category, **counts}
        item["recall"] = counts["detected"] / counts["support"]
        category_performance.append(item)
    category_performance.sort(
        key=lambda item: (-item["missed"], -item["support"], item["category"])
    )

    summary = {
        "protocol_version": "e1_dev_error_analysis_v1",
        "checkpoint_step": checkpoint_step,
        "total": expected_count,
        **confusion,
        "bad_other_only": {
            **bad_other_only,
            "recall": (
                bad_other_only["detected"] / bad_other_only["support"]
                if bad_other_only["support"]
                else None
            ),
        },
        "bad_category_performance": category_performance,
        "fn_gold_category_counts": _ranked_counts(fn_categories),
        "fn_gold_reason_counts": _ranked_counts(fn_reasons),
        "fp_predicted_category_counts": _ranked_counts(fp_categories),
        "fp_predicted_reason_counts": _ranked_counts(fp_reasons),
    }
    return summary, fn_rows, fp_rows


def _verify_metrics(summary: dict, metrics: dict) -> None:
    for field in ("checkpoint_step", "total", "tp", "fn", "fp", "tn"):
        if summary[field] != metrics.get(field):
            raise ErrorAnalysisError(
                f"metrics mismatch for {field}: {summary[field]} != {metrics.get(field)}"
            )


def run_analysis(
    dev_path: Path,
    parsed_path: Path,
    metrics_path: Path,
    output_dir: Path,
    *,
    checkpoint_step: int,
    expected_count: int = 200,
) -> dict:
    """Validate inputs and atomically publish summary, FN, and FP artifacts."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ErrorAnalysisError(f"output directory already exists: {output_dir}")
    dev_rows = _load_jsonl(dev_path)
    parsed_rows = _load_jsonl(parsed_path)
    try:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ErrorAnalysisError(f"invalid metrics JSON: {metrics_path}") from exc
    if not isinstance(metrics, dict):
        raise ErrorAnalysisError("metrics must be a JSON object")

    summary, fn_rows, fp_rows = analyze_rows(
        dev_rows,
        parsed_rows,
        checkpoint_step=checkpoint_step,
        expected_count=expected_count,
    )
    _verify_metrics(summary, metrics)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "fn.jsonl").write_text(
            _jsonl_text(fn_rows), encoding="utf-8", newline="\n"
        )
        (staging / "fp.jsonl").write_text(
            _jsonl_text(fp_rows), encoding="utf-8", newline="\n"
        )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--parsed", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--expected-count", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.dev,
            args.parsed,
            args.metrics,
            args.output_dir,
            checkpoint_step=args.checkpoint_step,
            expected_count=args.expected_count,
        )
    except (ErrorAnalysisError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
