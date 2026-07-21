#!/usr/bin/env python3
"""Validate completed Dev adjudication and recompute E1/E2 metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

try:
    from .build_e1_e2_dev_audit import compare_rows
    from .dev_audit_web import load_review_rows
except ImportError:  # pragma: no cover - direct script execution
    from build_e1_e2_dev_audit import compare_rows  # type: ignore
    from dev_audit_web import load_review_rows  # type: ignore


class AdjudicationError(ValueError):
    """Raised when adjudication inputs are incomplete or misaligned."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise AdjudicationError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdjudicationError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise AdjudicationError(f"row at {path}:{line_number} must be an object")
        rows.append(value)
    return rows


def _load_annotations(path: Path) -> tuple[dict[int, dict], str]:
    try:
        source = Path(path).read_bytes()
        value = json.loads(source.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"cannot read annotations: {path}") from exc
    if not isinstance(value, dict):
        raise AdjudicationError("annotations must be a JSON object")
    result: dict[int, dict] = {}
    for key, annotation in value.items():
        try:
            row = int(key)
        except ValueError as exc:
            raise AdjudicationError(f"invalid annotation row: {key}") from exc
        if not isinstance(annotation, dict):
            raise AdjudicationError(f"annotation {row} must be an object")
        result[row] = annotation
    return result, hashlib.sha256(source).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(gold: dict[int, str], predictions: dict[int, str | None]) -> dict:
    if set(gold) != set(predictions):
        raise AdjudicationError("metric gold/prediction row sets do not match")
    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for row, decision in gold.items():
        predicted = predictions[row]
        if decision == "BAD" and predicted == "BAD":
            confusion["tp"] += 1
        elif decision == "BAD":
            confusion["fn"] += 1
        elif predicted == "GOOD":
            confusion["tn"] += 1
        else:
            # INVALID is conservatively wrong, matching the generation protocol.
            confusion["fp"] += 1
    tp, fn, fp, tn = (confusion[key] for key in ("tp", "fn", "fp", "tn"))
    return {
        "total": len(gold),
        **confusion,
        "recall": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, len(gold)),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _ranked(counter: Counter[str]) -> list[dict]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _issue(row: int, code: str, detail: str) -> dict:
    return {"row": row, "code": code, "detail": detail}


def analyze_adjudication(
    dev_rows: list[dict],
    e1_rows: list[dict],
    e2_rows: list[dict],
    review_rows: list[dict],
    annotations: dict[int, dict],
    *,
    expected_count: int = 200,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Return summary, merged review records, label changes, and uncertain rows."""
    _, expected_review, _, _ = compare_rows(
        dev_rows, e1_rows, e2_rows, expected_count=expected_count
    )
    expected_by_row = {record["row"]: record for record in expected_review}
    review_by_row = {record["row"]: record for record in review_rows}
    if set(review_by_row) != set(expected_by_row):
        raise AdjudicationError("review manifest does not match reconstructed boundary rows")
    for row, record in review_by_row.items():
        expected = expected_by_row[row]
        for field in (
            "review_group",
            "decision_disagreement",
            "image_path",
            "gold",
            "e1",
            "e2",
        ):
            if record.get(field) != expected.get(field):
                raise AdjudicationError(f"review manifest drift at row {row}: {field}")
    if set(annotations) != set(review_by_row):
        missing = sorted(set(review_by_row) - set(annotations))
        extra = sorted(set(annotations) - set(review_by_row))
        raise AdjudicationError(
            f"annotation row set mismatch; missing={missing} extra={extra}"
        )

    label_status_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    bad_category_counts: Counter[str] = Counter()
    issues: list[dict] = []
    merged: list[dict] = []
    changes: list[dict] = []
    uncertain: list[dict] = []

    for row in sorted(review_by_row):
        record = review_by_row[row]
        annotation = annotations[row]
        if annotation.get("completed") is not True:
            raise AdjudicationError(f"annotation row {row} is not completed")
        label_status = annotation.get("label_status")
        severity = annotation.get("visible_severity")
        decision = annotation.get("review_decision")
        if label_status not in {"gold_correct", "gold_incorrect", "uncertain"}:
            raise AdjudicationError(f"invalid label_status at row {row}")
        if severity not in {"obvious", "borderline", "none", "uncertain"}:
            raise AdjudicationError(f"invalid visible_severity at row {row}")
        if decision not in {"GOOD", "BAD", "UNSURE"}:
            raise AdjudicationError(f"invalid review_decision at row {row}")

        original = record["gold"]["decision"]
        label_status_counts[label_status] += 1
        severity_counts[severity] += 1
        decision_counts[decision] += 1
        transition_counts[f"{original}->{decision}"] += 1
        if decision == "BAD":
            bad_category_counts[annotation.get("primary_category") or "未填写"] += 1

        if label_status == "gold_correct" and decision != original:
            issues.append(_issue(row, "gold_correct_decision_mismatch", f"{original}->{decision}"))
        if label_status == "gold_incorrect" and decision == original:
            issues.append(_issue(row, "gold_incorrect_but_unchanged", original))
        if label_status == "gold_incorrect" and decision == "UNSURE":
            issues.append(_issue(row, "gold_incorrect_without_binary_replacement", decision))
        if severity == "none" and decision == "BAD":
            issues.append(_issue(row, "no_visible_anomaly_but_bad", decision))
        if severity == "obvious" and decision == "GOOD":
            issues.append(_issue(row, "obvious_anomaly_but_good", decision))

        item = {**record, "annotation": annotation}
        merged.append(item)
        if decision in {"GOOD", "BAD"} and decision != original:
            changes.append(item)
        if decision == "UNSURE":
            uncertain.append(item)

    original_gold: dict[int, str] = {}
    e1_predictions: dict[int, str | None] = {}
    e2_predictions: dict[int, str | None] = {}
    for row_number, (dev, e1, e2) in enumerate(zip(dev_rows, e1_rows, e2_rows), start=1):
        try:
            original_gold[row_number] = json.loads(dev["messages"][-1]["content"])["decision"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AdjudicationError(f"cannot read Dev gold at row {row_number}") from exc
        e1_predictions[row_number] = (
            e1.get("predicted_decision") if e1.get("schema_valid") is True else None
        )
        e2_predictions[row_number] = (
            e2.get("predicted_decision") if e2.get("schema_valid") is True else None
        )

    adjusted_gold: dict[int, str] = {}
    excluded_unsure: list[int] = []
    for row, original in original_gold.items():
        annotation = annotations.get(row)
        if annotation is None:
            adjusted_gold[row] = original
            continue
        decision = annotation["review_decision"]
        if decision == "UNSURE":
            excluded_unsure.append(row)
        else:
            adjusted_gold[row] = decision
    adjusted_e1 = {row: e1_predictions[row] for row in adjusted_gold}
    adjusted_e2 = {row: e2_predictions[row] for row in adjusted_gold}

    review_binary_rows = {
        row: annotations[row]["review_decision"]
        for row in review_by_row
        if annotations[row]["review_decision"] in {"GOOD", "BAD"}
    }
    review_e1 = {row: e1_predictions[row] for row in review_binary_rows}
    review_e2 = {row: e2_predictions[row] for row in review_binary_rows}

    summary = {
        "protocol_version": "e1_e2_dev_adjudication_analysis_v1",
        "dev_total": expected_count,
        "review_total": len(review_rows),
        "completed": len(annotations),
        "label_status_counts": dict(sorted(label_status_counts.items())),
        "visible_severity_counts": dict(sorted(severity_counts.items())),
        "review_decision_counts": dict(sorted(decision_counts.items())),
        "original_to_review_counts": dict(sorted(transition_counts.items())),
        "reviewed_bad_primary_categories": _ranked(bad_category_counts),
        "binary_label_changes": len(changes),
        "excluded_unsure_rows": excluded_unsure,
        "consistency_issue_count": len(issues),
        "consistency_issues": issues,
        "analysis_ready": not issues,
        "metrics": {
            "original_gold_full_dev": {
                "e1": _metrics(original_gold, e1_predictions),
                "e2": _metrics(original_gold, e2_predictions),
            },
            "adjudicated_review_subset_excluding_unsure": {
                "e1": _metrics(review_binary_rows, review_e1),
                "e2": _metrics(review_binary_rows, review_e2),
            },
            "conditionally_adjusted_full_dev_excluding_unsure": {
                "assumption": "unreviewed 143 rows retain original Gold",
                "e1": _metrics(adjusted_gold, adjusted_e1),
                "e2": _metrics(adjusted_gold, adjusted_e2),
            },
        },
        "dev_use_restriction": "diagnosis_and_model_selection_only; never_train_on_dev",
    }
    return summary, merged, changes, uncertain


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def run_analysis(
    dev_path: Path,
    e1_path: Path,
    e2_path: Path,
    review_path: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    expected_count: int = 200,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise AdjudicationError(f"output directory already exists: {output_dir}")
    annotations, annotations_sha256 = _load_annotations(annotations_path)
    summary, merged, changes, uncertain = analyze_adjudication(
        _load_jsonl(dev_path),
        _load_jsonl(e1_path),
        _load_jsonl(e2_path),
        load_review_rows(review_path),
        annotations,
        expected_count=expected_count,
    )
    summary["annotations_sha256"] = annotations_sha256

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
        for name, rows in (
            ("reviewed-merged.jsonl", merged),
            ("binary-label-changes.jsonl", changes),
            ("uncertain.jsonl", uncertain),
        ):
            (staging / name).write_text(
                _jsonl_text(rows), encoding="utf-8", newline="\n"
            )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(summary: dict) -> None:
    print("=== ADJUDICATION COMPLETION ===")
    print(f"reviewed={summary['completed']}/{summary['review_total']}")
    print(f"label_status={summary['label_status_counts']}")
    print(f"severity={summary['visible_severity_counts']}")
    print(f"review_decision={summary['review_decision_counts']}")
    print(f"transitions={summary['original_to_review_counts']}")
    print(f"binary_label_changes={summary['binary_label_changes']}")
    print(f"excluded_unsure_rows={summary['excluded_unsure_rows']}")
    print(f"consistency_issues={summary['consistency_issue_count']}")
    for issue in summary["consistency_issues"]:
        print(f"  row={issue['row']} code={issue['code']} detail={issue['detail']}")

    print("\n=== E1 / E2 METRICS ===")
    for scope, scoped in summary["metrics"].items():
        print(scope)
        for model in ("e1", "e2"):
            metrics = scoped[model]
            print(
                f"  {model.upper()} N={metrics['total']} "
                f"TP={metrics['tp']} FN={metrics['fn']} FP={metrics['fp']} TN={metrics['tn']} "
                f"Recall={_percent(metrics['recall'])} FPR={_percent(metrics['fpr'])} "
                f"Accuracy={_percent(metrics['accuracy'])} F1={_percent(metrics['f1'])}"
            )
    print(
        "\nADJUDICATION_ANALYSIS: "
        + ("PASS" if summary["analysis_ready"] else "NEEDS_REVIEW")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--e1-parsed", required=True, type=Path)
    parser.add_argument("--e2-parsed", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.dev,
            args.e1_parsed,
            args.e2_parsed,
            args.review,
            args.annotations,
            args.output_dir,
            expected_count=args.expected_count,
        )
    except (AdjudicationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_report(summary)
    return 0 if summary["analysis_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
