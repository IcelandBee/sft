#!/usr/bin/env python3
"""Build an immutable corrected Dev JSONL from completed adjudication."""

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
    from .dev_audit_web import load_review_rows
    from .evaluate_e1_dev import validate_payload
except ImportError:  # pragma: no cover - direct script execution
    from dev_audit_web import load_review_rows  # type: ignore
    from evaluate_e1_dev import validate_payload  # type: ignore


class AdjudicatedDevError(ValueError):
    """Raised when adjudication cannot safely produce a corrected Dev."""


def _load_jsonl(path: Path) -> tuple[list[dict], bytes]:
    try:
        source = Path(path).read_bytes()
        lines = source.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AdjudicatedDevError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdjudicatedDevError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise AdjudicatedDevError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows, source


def _load_annotations(path: Path) -> tuple[dict[int, dict], str]:
    try:
        source = Path(path).read_bytes()
        value = json.loads(source.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicatedDevError(f"cannot read annotations: {path}") from exc
    if not isinstance(value, dict):
        raise AdjudicatedDevError("annotations must be an object")
    result: dict[int, dict] = {}
    for key, annotation in value.items():
        try:
            row = int(key)
        except ValueError as exc:
            raise AdjudicatedDevError(f"invalid annotation row: {key}") from exc
        if not isinstance(annotation, dict):
            raise AdjudicatedDevError(f"annotation {row} must be an object")
        result[row] = annotation
    return result, hashlib.sha256(source).hexdigest()


def _gold(row: dict, row_number: int) -> dict:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
    ):
        raise AdjudicatedDevError(f"invalid Dev messages at row {row_number}")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AdjudicatedDevError(f"invalid Dev gold at row {row_number}") from exc
    schema_error = validate_payload(payload)
    if schema_error is not None:
        raise AdjudicatedDevError(
            f"Dev gold at row {row_number} violates schema: {schema_error}"
        )
    return payload


def _bad_payload(annotation: dict, row: int) -> dict:
    category = annotation.get("primary_category")
    reason = annotation.get("notes")
    if not isinstance(category, str) or not category.strip():
        raise AdjudicatedDevError(
            f"GOOD->BAD row {row} requires primary_category"
        )
    category = category.strip()
    if category in {"无可见异常", "类别不确定"}:
        raise AdjudicatedDevError(
            f"GOOD->BAD row {row} has unusable primary_category: {category}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise AdjudicatedDevError(f"GOOD->BAD row {row} requires review notes")
    payload = {
        "decision": "BAD",
        "categories": [category],
        "reasons": [reason.strip()],
    }
    schema_error = validate_payload(payload)
    if schema_error is not None:
        raise AdjudicatedDevError(
            f"generated BAD payload at row {row} violates schema: {schema_error}"
        )
    return payload


def build_rows(
    dev_rows: list[dict],
    review_rows: list[dict],
    annotations: dict[int, dict],
    *,
    expected_count: int = 200,
    expected_review: int = 57,
    expected_changes: int = 27,
) -> tuple[list[dict], list[dict], dict]:
    """Return corrected Dev rows, decision changes, and count summary."""
    if len(dev_rows) != expected_count:
        raise AdjudicatedDevError(
            f"expected {expected_count} Dev rows, got {len(dev_rows)}"
        )
    if len(review_rows) != expected_review:
        raise AdjudicatedDevError(
            f"expected {expected_review} review rows, got {len(review_rows)}"
        )
    review_by_row = {row["row"]: row for row in review_rows}
    if len(review_by_row) != len(review_rows):
        raise AdjudicatedDevError("review rows contain duplicate row numbers")
    if set(annotations) != set(review_by_row):
        raise AdjudicatedDevError("annotation rows do not match review rows")

    corrected: list[dict] = []
    changes: list[dict] = []
    original_counts: Counter[str] = Counter()
    corrected_counts: Counter[str] = Counter()

    for row_number, source_row in enumerate(dev_rows, start=1):
        original = _gold(source_row, row_number)
        original_decision = original["decision"]
        original_counts[original_decision] += 1
        payload = original
        annotation = annotations.get(row_number)
        if annotation is not None:
            if annotation.get("completed") is not True:
                raise AdjudicatedDevError(f"annotation row {row_number} is incomplete")
            label_status = annotation.get("label_status")
            decision = annotation.get("review_decision")
            if decision not in {"GOOD", "BAD"}:
                raise AdjudicatedDevError(
                    f"annotation row {row_number} must have binary review_decision"
                )
            if label_status == "gold_correct" and decision != original_decision:
                raise AdjudicatedDevError(
                    f"gold_correct row {row_number} changes {original_decision}->{decision}"
                )
            if label_status == "gold_incorrect" and decision == original_decision:
                raise AdjudicatedDevError(
                    f"gold_incorrect row {row_number} does not change decision"
                )
            if label_status not in {"gold_correct", "gold_incorrect"}:
                raise AdjudicatedDevError(
                    f"annotation row {row_number} has unresolved label_status"
                )
            if decision != original_decision:
                payload = (
                    {"decision": "GOOD", "categories": [], "reasons": []}
                    if decision == "GOOD"
                    else _bad_payload(annotation, row_number)
                )
                changes.append(
                    {
                        "row": row_number,
                        "image_path": source_row.get("images", [None])[0],
                        "original": original,
                        "adjudicated": payload,
                        "annotation": annotation,
                    }
                )

        corrected_counts[payload["decision"]] += 1
        row = json.loads(json.dumps(source_row, ensure_ascii=False))
        row["messages"][-1]["content"] = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        corrected.append(row)

    if len(changes) != expected_changes:
        raise AdjudicatedDevError(
            f"expected {expected_changes} decision changes, got {len(changes)}"
        )
    summary = {
        "total": expected_count,
        "reviewed": expected_review,
        "decision_changes": len(changes),
        "original_counts": dict(sorted(original_counts.items())),
        "adjudicated_counts": dict(sorted(corrected_counts.items())),
    }
    return corrected, changes, summary


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def run_build(
    dev_path: Path,
    review_path: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    expected_count: int = 200,
    expected_review: int = 57,
    expected_changes: int = 27,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise AdjudicatedDevError(f"output directory already exists: {output_dir}")
    dev_rows, dev_source = _load_jsonl(dev_path)
    review_rows = load_review_rows(review_path)
    annotations, annotations_sha256 = _load_annotations(annotations_path)
    corrected, changes, summary = build_rows(
        dev_rows,
        review_rows,
        annotations,
        expected_count=expected_count,
        expected_review=expected_review,
        expected_changes=expected_changes,
    )
    dev_text = _jsonl_text(corrected)
    manifest = {
        "protocol_version": "adjudicated_dev_v1",
        **summary,
        "source_dev": str(dev_path),
        "source_dev_sha256": hashlib.sha256(dev_source).hexdigest(),
        "review": str(review_path),
        "annotations": str(annotations_path),
        "annotations_sha256": annotations_sha256,
        "dev_sha256": hashlib.sha256(dev_text.encode("utf-8")).hexdigest(),
        "usage": "validation_and_model_selection_only",
        "training_forbidden": True,
        "test_untouched": True,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        (staging / "dev.jsonl").write_text(
            dev_text, encoding="utf-8", newline="\n"
        )
        (staging / "decision-changes.jsonl").write_text(
            _jsonl_text(changes), encoding="utf-8", newline="\n"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-review", type=int, default=57)
    parser.add_argument("--expected-changes", type=int, default=27)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_build(
            args.dev,
            args.review,
            args.annotations,
            args.output_dir,
            expected_count=args.expected_count,
            expected_review=args.expected_review,
            expected_changes=args.expected_changes,
        )
    except (AdjudicatedDevError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("ADJUDICATED_DEV_BUILD: PASS")
    print(f"rows={manifest['total']} reviewed={manifest['reviewed']} changes={manifest['decision_changes']}")
    print(f"original_counts={manifest['original_counts']}")
    print(f"adjudicated_counts={manifest['adjudicated_counts']}")
    print(f"dev_sha256={manifest['dev_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
