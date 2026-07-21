#!/usr/bin/env python3
"""Analyze E2 residual FN/FP cases on the frozen adjudicated Dev."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable


class ResidualAnalysisError(ValueError):
    """Raised when frozen Dev, predictions, or audit data do not align."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ResidualAnalysisError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResidualAnalysisError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ResidualAnalysisError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _load_annotations(path: Path) -> dict[int, dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualAnalysisError(f"cannot read annotations: {path}") from exc
    if not isinstance(value, dict):
        raise ResidualAnalysisError("annotations must be an object")
    result: dict[int, dict] = {}
    for key, annotation in value.items():
        try:
            row = int(key)
        except ValueError as exc:
            raise ResidualAnalysisError(f"invalid annotation row: {key}") from exc
        if not isinstance(annotation, dict):
            raise ResidualAnalysisError(f"annotation {row} must be an object")
        result[row] = annotation
    return result


def _payload(row: dict, row_number: int, name: str) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ResidualAnalysisError(f"invalid {name} messages at row {row_number}")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ResidualAnalysisError(f"invalid {name} gold at row {row_number}") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in {"GOOD", "BAD"}:
        raise ResidualAnalysisError(f"invalid {name} decision at row {row_number}")
    return payload


def _image(row: dict, row_number: int, name: str) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ResidualAnalysisError(f"invalid {name} image at row {row_number}")
    image = images[0]
    if not isinstance(image, str):
        raise ResidualAnalysisError(f"invalid {name} image path at row {row_number}")
    return image


def _ranked(counter: Counter[str]) -> list[dict]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _train_coverage(rows: list[dict]) -> tuple[int, Counter[str]]:
    bad_by_image: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=1):
        image = _image(row, row_number, "Train")
        payload = _payload(row, row_number, "Train")
        if payload["decision"] != "BAD":
            continue
        categories = payload.get("categories")
        if not isinstance(categories, list):
            raise ResidualAnalysisError(f"invalid Train categories at row {row_number}")
        values = {item for item in categories if isinstance(item, str) and item}
        if image in bad_by_image and bad_by_image[image] != values:
            raise ResidualAnalysisError(f"inconsistent duplicate Train labels: {image}")
        bad_by_image[image] = values
    counts: Counter[str] = Counter()
    for categories in bad_by_image.values():
        counts.update(categories)
    return len(bad_by_image), counts


def analyze_residuals(
    dev_rows: list[dict],
    original_dev_rows: list[dict],
    e2_rows: list[dict],
    annotations: dict[int, dict],
    train_rows: list[dict],
    *,
    expected_count: int = 200,
    expected_fn: int = 15,
    expected_fp: int = 5,
) -> tuple[dict, list[dict], list[dict]]:
    """Return residual summary and ordered FN/FP records."""
    for name, rows in (
        ("adjudicated Dev", dev_rows),
        ("original Dev", original_dev_rows),
        ("E2", e2_rows),
    ):
        if len(rows) != expected_count:
            raise ResidualAnalysisError(
                f"expected {expected_count} {name} rows, got {len(rows)}"
            )

    fn_rows: list[dict] = []
    fp_rows: list[dict] = []
    fn_severity: Counter[str] = Counter()
    fp_severity: Counter[str] = Counter()
    fn_categories: Counter[str] = Counter()
    fp_model_categories: Counter[str] = Counter()
    error_label_origin: Counter[str] = Counter()
    corrected_counts: Counter[str] = Counter()
    tp = tn = 0

    for row_number, (dev, original_dev, prediction) in enumerate(
        zip(dev_rows, original_dev_rows, e2_rows), start=1
    ):
        image = _image(dev, row_number, "adjudicated Dev")
        if _image(original_dev, row_number, "original Dev") != image:
            raise ResidualAnalysisError(f"Dev image drift at row {row_number}")
        if prediction.get("index") != row_number - 1 or prediction.get("image_path") != image:
            raise ResidualAnalysisError(f"E2 alignment mismatch at row {row_number}")
        gold = _payload(dev, row_number, "adjudicated Dev")
        original = _payload(original_dev, row_number, "original Dev")
        corrected_counts[gold["decision"]] += 1
        predicted = (
            prediction.get("predicted_decision")
            if prediction.get("schema_valid") is True
            else None
        )
        if predicted not in {"GOOD", "BAD"}:
            raise ResidualAnalysisError(f"E2 prediction is invalid at row {row_number}")

        if gold["decision"] == "BAD" and predicted == "BAD":
            tp += 1
            continue
        if gold["decision"] == "GOOD" and predicted == "GOOD":
            tn += 1
            continue

        error_type = "FN" if gold["decision"] == "BAD" else "FP"
        annotation = annotations.get(row_number, {})
        severity = annotation.get("visible_severity") or "unreviewed"
        label_origin = (
            "corrected_gold"
            if gold["decision"] != original["decision"]
            else "original_gold_retained"
        )
        primary_category = annotation.get("primary_category") or ""
        if error_type == "FN" and not primary_category:
            categories = gold.get("categories")
            primary_category = (
                categories[0]
                if isinstance(categories, list) and categories
                else "未填写"
            )
        record = {
            "row": row_number,
            "image_path": image,
            "error_type": error_type,
            "gold": gold,
            "original_gold": original,
            "predicted_decision": predicted,
            "prediction": prediction.get("payload"),
            "label_origin": label_origin,
            "visible_severity": severity,
            "primary_category": primary_category or "未填写",
            "review_notes": annotation.get("notes") or "",
        }
        error_label_origin[f"{error_type}:{label_origin}"] += 1
        if error_type == "FN":
            fn_rows.append(record)
            fn_severity[severity] += 1
            fn_categories[record["primary_category"]] += 1
        else:
            fp_rows.append(record)
            fp_severity[severity] += 1
            payload = prediction.get("payload")
            categories = payload.get("categories") if isinstance(payload, dict) else []
            fp_model_categories.update(categories or ["未填写"])

    if len(fn_rows) != expected_fn or len(fp_rows) != expected_fp:
        raise ResidualAnalysisError(
            f"expected FN/FP={expected_fn}/{expected_fp}, got {len(fn_rows)}/{len(fp_rows)}"
        )
    unique_train_bad, train_category_counts = _train_coverage(train_rows)
    bad_total = corrected_counts["BAD"]
    good_total = corrected_counts["GOOD"]
    challenge_tp = math.ceil(0.78 * bad_total)
    challenge_max_fp = math.ceil(0.20 * good_total) - 1
    summary = {
        "protocol_version": "e2_adjudicated_dev_residual_analysis_v1",
        "total": expected_count,
        "corrected_label_counts": dict(sorted(corrected_counts.items())),
        "confusion": {
            "tp": tp,
            "fn": len(fn_rows),
            "fp": len(fp_rows),
            "tn": tn,
        },
        "fn_visible_severity": _ranked(fn_severity),
        "fp_visible_severity": _ranked(fp_severity),
        "fn_primary_categories": _ranked(fn_categories),
        "fp_predicted_categories": _ranked(fp_model_categories),
        "error_label_origin": _ranked(error_label_origin),
        "train_unique_bad": unique_train_bad,
        "train_bad_category_counts": _ranked(train_category_counts),
        "challenge_gap": {
            "bad_total": bad_total,
            "current_tp": tp,
            "tp_required_for_recall_at_least_78pct": challenge_tp,
            "minimum_additional_tp_if_no_regressions": max(0, challenge_tp - tp),
            "good_total": good_total,
            "current_fp": len(fp_rows),
            "max_fp_for_fpr_below_20pct": challenge_max_fp,
        },
        "dev_use_restriction": "diagnosis_only; never_train_on_dev",
    }
    return summary, fn_rows, fp_rows


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def run_analysis(
    dev_path: Path,
    original_dev_path: Path,
    e2_path: Path,
    annotations_path: Path,
    train_path: Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ResidualAnalysisError(f"output directory already exists: {output_dir}")
    summary, fn_rows, fp_rows = analyze_residuals(
        _load_jsonl(dev_path),
        _load_jsonl(original_dev_path),
        _load_jsonl(e2_path),
        _load_annotations(annotations_path),
        _load_jsonl(train_path),
    )
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
    parser.add_argument("--original-dev", required=True, type=Path)
    parser.add_argument("--e2-parsed", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.dev,
            args.original_dev,
            args.e2_parsed,
            args.annotations,
            args.train,
            args.output_dir,
        )
    except (ResidualAnalysisError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== E2 ADJUDICATED RESIDUALS ===")
    print(f"confusion={summary['confusion']}")
    print(f"fn_severity={summary['fn_visible_severity']}")
    print(f"fp_severity={summary['fp_visible_severity']}")
    print(f"fn_categories={summary['fn_primary_categories']}")
    print(f"fp_predicted_categories={summary['fp_predicted_categories']}")
    print(f"error_label_origin={summary['error_label_origin']}")
    print(f"train_unique_bad={summary['train_unique_bad']}")
    print(f"train_bad_category_counts={summary['train_bad_category_counts']}")
    print(f"challenge_gap={summary['challenge_gap']}")
    print("E2_RESIDUAL_ANALYSIS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
