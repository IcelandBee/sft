#!/usr/bin/env python3
"""Build a human-review package for E1/E2 Dev boundary cases."""

from __future__ import annotations

import argparse
import csv
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


class AuditError(ValueError):
    """Raised when Dev or prediction artifacts do not align."""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise AuditError(f"cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _gold(row: dict, row_number: int) -> tuple[str, dict]:
    images = row.get("images")
    messages = row.get("messages")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise AuditError(f"invalid Dev image at row {row_number}")
    if not isinstance(messages, list) or len(messages) != 3:
        raise AuditError(f"invalid Dev messages at row {row_number}")
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if not isinstance(content, str):
        raise AuditError(f"invalid Dev gold at row {row_number}")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid Dev gold JSON at row {row_number}") from exc
    schema_error = validate_payload(payload)
    if schema_error is not None:
        raise AuditError(
            f"Dev gold at row {row_number} violates schema: {schema_error}"
        )
    return images[0], payload


def _prediction(row: dict, image_path: str, row_number: int, name: str) -> dict:
    if row.get("index") != row_number - 1:
        raise AuditError(f"{name} index mismatch at row {row_number}")
    if row.get("image_path") != image_path:
        raise AuditError(f"{name} image mismatch at row {row_number}")
    schema_valid = row.get("schema_valid") is True
    decision = row.get("predicted_decision") if schema_valid else None
    payload = row.get("payload") if schema_valid else None
    if schema_valid:
        if decision not in {"GOOD", "BAD"} or not isinstance(payload, dict):
            raise AuditError(f"invalid {name} prediction at row {row_number}")
        schema_error = validate_payload(payload)
        if schema_error is not None or payload.get("decision") != decision:
            raise AuditError(f"invalid {name} payload at row {row_number}")
    return {
        "schema_valid": schema_valid,
        "decision": decision,
        "payload": payload,
        "error_code": row.get("error_code"),
    }


def compare_rows(
    dev_rows: list[dict],
    e1_rows: list[dict],
    e2_rows: list[dict],
    *,
    expected_count: int = 200,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Compare aligned E1/E2 predictions and return review-ready records."""
    for name, rows in (("Dev", dev_rows), ("E1", e1_rows), ("E2", e2_rows)):
        if len(rows) != expected_count:
            raise AuditError(f"expected {expected_count} {name} rows, got {len(rows)}")

    counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "e1_only_correct": 0,
        "e2_only_correct": 0,
        "decision_disagreements": 0,
    }
    all_records: list[dict] = []
    both_wrong: list[dict] = []
    disagreements: list[dict] = []

    for row_number, (dev, e1_row, e2_row) in enumerate(
        zip(dev_rows, e1_rows, e2_rows), start=1
    ):
        image_path, gold = _gold(dev, row_number)
        e1 = _prediction(e1_row, image_path, row_number, "E1")
        e2 = _prediction(e2_row, image_path, row_number, "E2")
        e1_correct = e1["schema_valid"] and e1["decision"] == gold["decision"]
        e2_correct = e2["schema_valid"] and e2["decision"] == gold["decision"]

        if e1_correct and e2_correct:
            group = "both_correct"
        elif not e1_correct and not e2_correct:
            group = "both_wrong"
        elif e1_correct:
            group = "e1_only_correct"
        else:
            group = "e2_only_correct"
        counts[group] += 1

        decision_disagreement = (
            e1["schema_valid"]
            and e2["schema_valid"]
            and e1["decision"] != e2["decision"]
        )
        if decision_disagreement:
            counts["decision_disagreements"] += 1

        record = {
            "row": row_number,
            "review_group": group,
            "decision_disagreement": decision_disagreement,
            "image_path": image_path,
            "gold": gold,
            "e1": e1,
            "e2": e2,
        }
        all_records.append(record)
        if group == "both_wrong":
            both_wrong.append(record)
        if decision_disagreement:
            disagreements.append(record)

    review_records = sorted(
        both_wrong + disagreements,
        key=lambda item: (0 if item["decision_disagreement"] else 1, item["row"]),
    )
    summary = {
        "protocol_version": "e1_e2_dev_boundary_audit_v1",
        "total": expected_count,
        **counts,
        "review_total": len(review_records),
        "review_scope": "decision_disagreements_then_both_wrong",
        "dev_use_restriction": "diagnosis_and_model_selection_only; never_train_on_dev",
    }
    return summary, review_records, both_wrong, disagreements


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


CSV_FIELDS = [
    "review_order",
    "row",
    "review_group",
    "image_path",
    "gold_decision",
    "gold_categories",
    "gold_reasons",
    "e1_decision",
    "e1_categories",
    "e1_reasons",
    "e2_decision",
    "e2_categories",
    "e2_reasons",
    "review_label_status",
    "review_visible_severity",
    "review_decision",
    "review_primary_category",
    "review_notes",
]


def _join_payload(payload: dict | None, field: str) -> str:
    if not isinstance(payload, dict):
        return ""
    values = payload.get(field)
    return " | ".join(values) if isinstance(values, list) else ""


def _write_review_csv(path: Path, records: list[dict]) -> None:
    # utf-8-sig keeps Chinese text readable when opened directly in Excel.
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for order, record in enumerate(records, start=1):
            gold = record["gold"]
            e1_payload = record["e1"]["payload"]
            e2_payload = record["e2"]["payload"]
            writer.writerow(
                {
                    "review_order": order,
                    "row": record["row"],
                    "review_group": record["review_group"],
                    "image_path": record["image_path"],
                    "gold_decision": gold["decision"],
                    "gold_categories": _join_payload(gold, "categories"),
                    "gold_reasons": _join_payload(gold, "reasons"),
                    "e1_decision": record["e1"]["decision"] or "INVALID",
                    "e1_categories": _join_payload(e1_payload, "categories"),
                    "e1_reasons": _join_payload(e1_payload, "reasons"),
                    "e2_decision": record["e2"]["decision"] or "INVALID",
                    "e2_categories": _join_payload(e2_payload, "categories"),
                    "e2_reasons": _join_payload(e2_payload, "reasons"),
                    "review_label_status": "",
                    "review_visible_severity": "",
                    "review_decision": "",
                    "review_primary_category": "",
                    "review_notes": "",
                }
            )


def run_audit(
    dev_path: Path,
    e1_parsed_path: Path,
    e2_parsed_path: Path,
    output_dir: Path,
    *,
    expected_count: int = 200,
) -> dict:
    """Validate inputs and atomically publish the E1/E2 review package."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise AuditError(f"output directory already exists: {output_dir}")
    summary, review, both_wrong, disagreements = compare_rows(
        _load_jsonl(dev_path),
        _load_jsonl(e1_parsed_path),
        _load_jsonl(e2_parsed_path),
        expected_count=expected_count,
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
        (staging / "review.jsonl").write_text(
            _jsonl_text(review), encoding="utf-8", newline="\n"
        )
        (staging / "both-wrong.jsonl").write_text(
            _jsonl_text(both_wrong), encoding="utf-8", newline="\n"
        )
        (staging / "decision-disagreements.jsonl").write_text(
            _jsonl_text(disagreements), encoding="utf-8", newline="\n"
        )
        _write_review_csv(staging / "review.csv", review)
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--e1-parsed", required=True, type=Path)
    parser.add_argument("--e2-parsed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_audit(
            args.dev,
            args.e1_parsed,
            args.e2_parsed,
            args.output_dir,
            expected_count=args.expected_count,
        )
    except (AuditError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
