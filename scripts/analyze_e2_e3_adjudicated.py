#!/usr/bin/env python3
"""Compare E2 and E3 checkpoint-1248 on the frozen adjudicated Dev."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable


class ComparisonError(ValueError):
    """Raised when the frozen data or paired predictions do not align."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ComparisonError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ComparisonError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ComparisonError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _load_annotations(path: Path) -> dict[int, dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read annotations: {path}") from exc
    if not isinstance(value, dict):
        raise ComparisonError("annotations must be an object")
    result: dict[int, dict] = {}
    for key, annotation in value.items():
        try:
            row_number = int(key)
        except ValueError as exc:
            raise ComparisonError(f"invalid annotation row: {key}") from exc
        if not isinstance(annotation, dict):
            raise ComparisonError(f"annotation {key} must be an object")
        result[row_number] = annotation
    return result


def _gold(row: dict, row_number: int, name: str) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ComparisonError(f"invalid {name} messages at row {row_number}")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid {name} gold at row {row_number}") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in {"GOOD", "BAD"}:
        raise ComparisonError(f"invalid {name} decision at row {row_number}")
    return payload


def _image(row: dict, row_number: int, name: str) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ComparisonError(f"invalid {name} image at row {row_number}")
    return images[0]


def _prediction(row: dict, row_number: int, name: str) -> str | None:
    if row.get("index") != row_number - 1:
        raise ComparisonError(f"invalid {name} index at row {row_number}")
    if row.get("schema_valid") is not True:
        return None
    decision = row.get("predicted_decision")
    if decision not in {"GOOD", "BAD"}:
        raise ComparisonError(f"invalid {name} prediction at row {row_number}")
    return decision


def _confusion(golds: list[str], predictions: list[str | None]) -> dict[str, int]:
    result = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for gold, prediction in zip(golds, predictions):
        if gold == "BAD" and prediction == "BAD":
            result["tp"] += 1
        elif gold == "BAD":
            result["fn"] += 1
        elif prediction == "GOOD":
            result["tn"] += 1
        else:
            result["fp"] += 1
    return result


def _ranked(counter: Counter[str]) -> list[dict]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _rates(confusion: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = (confusion[key] for key in ("tp", "fn", "fp", "tn"))
    total = tp + fn + fp + tn
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return {
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, total),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def analyze_comparison(
    dev_rows: list[dict],
    original_dev_rows: list[dict],
    e2_rows: list[dict],
    e3_rows: list[dict],
    annotations: dict[int, dict],
    *,
    expected_count: int = 200,
    expected_e2: dict[str, int] | None = None,
    expected_e3: dict[str, int] | None = None,
) -> tuple[dict, dict[str, list[dict]]]:
    """Return paired E2/E3 attribution summary and auditable row lists."""
    for name, rows in (
        ("adjudicated Dev", dev_rows),
        ("original Dev", original_dev_rows),
        ("E2", e2_rows),
        ("E3", e3_rows),
    ):
        if len(rows) != expected_count:
            raise ComparisonError(f"expected {expected_count} {name} rows, got {len(rows)}")

    records: dict[str, list[dict]] = {
        "decision_differences": [],
        "e2_only_correct": [],
        "e3_only_correct": [],
        "e3_fn": [],
        "e3_fp": [],
    }
    outcome_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    e2_only_categories: Counter[str] = Counter()
    e3_only_categories: Counter[str] = Counter()
    e3_fn_categories: Counter[str] = Counter()
    e3_fp_categories: Counter[str] = Counter()
    delta_label_origin: Counter[str] = Counter()
    delta_severity: Counter[str] = Counter()
    golds: list[str] = []
    e2_predictions: list[str | None] = []
    e3_predictions: list[str | None] = []

    for row_number, (dev, original_dev, e2, e3) in enumerate(
        zip(dev_rows, original_dev_rows, e2_rows, e3_rows), start=1
    ):
        image = _image(dev, row_number, "adjudicated Dev")
        if _image(original_dev, row_number, "original Dev") != image:
            raise ComparisonError(f"Dev image drift at row {row_number}")
        if e2.get("image_path") != image or e3.get("image_path") != image:
            raise ComparisonError(f"prediction alignment mismatch at row {row_number}")
        gold_payload = _gold(dev, row_number, "adjudicated Dev")
        original_payload = _gold(original_dev, row_number, "original Dev")
        gold = gold_payload["decision"]
        e2_prediction = _prediction(e2, row_number, "E2")
        e3_prediction = _prediction(e3, row_number, "E3")
        golds.append(gold)
        e2_predictions.append(e2_prediction)
        e3_predictions.append(e3_prediction)

        e2_correct = e2_prediction == gold
        e3_correct = e3_prediction == gold
        if e2_correct and e3_correct:
            outcome = "both_correct"
        elif e2_correct:
            outcome = "e2_only_correct"
        elif e3_correct:
            outcome = "e3_only_correct"
        else:
            outcome = "both_wrong"
        outcome_counts[outcome] += 1

        annotation = annotations.get(row_number, {})
        if gold != original_payload["decision"]:
            label_origin = "corrected_gold"
        elif row_number in annotations:
            label_origin = "reviewed_gold_retained"
        else:
            label_origin = "unreviewed_gold_retained"
        severity = annotation.get("visible_severity") or "unreviewed"
        record = {
            "row": row_number,
            "image_path": image,
            "gold": gold_payload,
            "original_gold": original_payload,
            "e2_decision": e2_prediction,
            "e3_decision": e3_prediction,
            "e2_payload": e2.get("payload"),
            "e3_payload": e3.get("payload"),
            "outcome": outcome,
            "label_origin": label_origin,
            "visible_severity": severity,
            "review_notes": annotation.get("notes") or "",
        }

        if e2_prediction != e3_prediction:
            records["decision_differences"].append(record)
            transition_counts[f"{e2_prediction}->{e3_prediction}"] += 1
        if outcome in {"e2_only_correct", "e3_only_correct"}:
            records[outcome].append(record)
            delta_label_origin[f"{outcome}:{label_origin}"] += 1
            delta_severity[f"{outcome}:{severity}"] += 1
            categories = gold_payload.get("categories") or ["GOOD"]
            target = e2_only_categories if outcome == "e2_only_correct" else e3_only_categories
            target.update(categories)
        if gold == "BAD" and not e3_correct:
            records["e3_fn"].append(record)
            e3_fn_categories.update(gold_payload.get("categories") or ["未填写"])
        if gold == "GOOD" and not e3_correct:
            records["e3_fp"].append(record)
            payload = e3.get("payload")
            categories = payload.get("categories") if isinstance(payload, dict) else None
            e3_fp_categories.update(categories or ["未填写"])

    e2_confusion = _confusion(golds, e2_predictions)
    e3_confusion = _confusion(golds, e3_predictions)
    if expected_e2 is not None and e2_confusion != expected_e2:
        raise ComparisonError(f"E2 confusion mismatch: {e2_confusion} != {expected_e2}")
    if expected_e3 is not None and e3_confusion != expected_e3:
        raise ComparisonError(f"E3 confusion mismatch: {e3_confusion} != {expected_e3}")

    summary = {
        "protocol_version": "e2_e3_adjudicated_attribution_v1",
        "total": expected_count,
        "e2": {"confusion": e2_confusion, **_rates(e2_confusion)},
        "e3": {"confusion": e3_confusion, **_rates(e3_confusion)},
        "e3_minus_e2": {
            key: e3_confusion[key] - e2_confusion[key] for key in e2_confusion
        },
        "paired_outcomes": dict(sorted(outcome_counts.items())),
        "decision_agreement": expected_count - len(records["decision_differences"]),
        "decision_differences": len(records["decision_differences"]),
        "decision_transitions": _ranked(transition_counts),
        "e2_only_correct_gold_categories": _ranked(e2_only_categories),
        "e3_only_correct_gold_categories": _ranked(e3_only_categories),
        "e3_fn_gold_categories": _ranked(e3_fn_categories),
        "e3_fp_predicted_categories": _ranked(e3_fp_categories),
        "delta_label_origin": _ranked(delta_label_origin),
        "delta_visible_severity": _ranked(delta_severity),
        "dev_use_restriction": "diagnosis_only; never_train_on_dev",
    }
    return summary, records


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def run_analysis(
    dev_path: Path,
    original_dev_path: Path,
    e2_path: Path,
    e3_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ComparisonError(f"output directory already exists: {output_dir}")
    summary, records = analyze_comparison(
        _load_jsonl(dev_path),
        _load_jsonl(original_dev_path),
        _load_jsonl(e2_path),
        _load_jsonl(e3_path),
        _load_annotations(annotations_path),
        expected_e2={"tp": 43, "fn": 15, "fp": 5, "tn": 137},
        expected_e3={"tp": 36, "fn": 22, "fp": 9, "tn": 133},
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for name, rows in records.items():
            (staging / f"{name.replace('_', '-')}.jsonl").write_text(
                _jsonl_text(rows), encoding="utf-8", newline="\n"
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
    parser.add_argument("--e3-parsed", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.dev,
            args.original_dev,
            args.e2_parsed,
            args.e3_parsed,
            args.annotations,
            args.output_dir,
        )
    except (ComparisonError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== E2 VS E3 ADJUDICATED ATTRIBUTION ===")
    print(f"e2={summary['e2']}")
    print(f"e3={summary['e3']}")
    print(f"e3_minus_e2={summary['e3_minus_e2']}")
    print(f"paired_outcomes={summary['paired_outcomes']}")
    print(f"decision_agreement={summary['decision_agreement']}/{summary['total']}")
    print(f"transitions={summary['decision_transitions']}")
    print(f"e2_only_categories={summary['e2_only_correct_gold_categories']}")
    print(f"e3_only_categories={summary['e3_only_correct_gold_categories']}")
    print(f"e3_fn_categories={summary['e3_fn_gold_categories']}")
    print(f"e3_fp_categories={summary['e3_fp_predicted_categories']}")
    print(f"delta_label_origin={summary['delta_label_origin']}")
    print(f"delta_severity={summary['delta_visible_severity']}")
    print("E2_E3_ATTRIBUTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
