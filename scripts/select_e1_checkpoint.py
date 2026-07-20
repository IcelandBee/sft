#!/usr/bin/env python3
"""Select one E1 checkpoint using only pre-registered Dev metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile


EXPECTED_STEPS = (312, 624, 936, 1248, 1560, 1872, 2184, 2496)
REQUIRED_RATES = ("recall", "fpr", "accuracy", "f1", "schema_valid_rate")


class SelectionError(ValueError):
    """Raised when checkpoint metrics violate the selection contract."""


def _validate_metrics(metrics: list[dict]) -> list[dict]:
    rows = [dict(row) for row in metrics]
    steps = [row.get("checkpoint_step") for row in rows]
    duplicates = sorted(step for step, count in Counter(steps).items() if count > 1)
    if duplicates:
        raise SelectionError(f"duplicate checkpoint step: {duplicates[0]}")
    missing = sorted(set(EXPECTED_STEPS) - set(steps))
    extra = sorted(set(steps) - set(EXPECTED_STEPS), key=str)
    if missing:
        raise SelectionError(f"missing checkpoint step: {missing[0]}")
    if extra:
        raise SelectionError(f"unexpected checkpoint step: {extra[0]}")
    if len(rows) != len(EXPECTED_STEPS):
        raise SelectionError(
            f"expected {len(EXPECTED_STEPS)} checkpoint metrics, got {len(rows)}"
        )

    for row in rows:
        step = row["checkpoint_step"]
        if row.get("total") != 200:
            raise SelectionError(f"checkpoint-{step} must have total=200")
        for field in REQUIRED_RATES:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SelectionError(f"checkpoint-{step} {field} must be numeric")
            if not 0 <= value <= 1:
                raise SelectionError(f"checkpoint-{step} {field} must be in [0, 1]")
    return rows


def _rank_key(row: dict) -> tuple[float, float, float, int]:
    return (-row["recall"], -row["accuracy"], -row["f1"], row["checkpoint_step"])


def select_checkpoint(metrics: list[dict]) -> dict:
    """Apply the fixed Dev gates and ranking rule."""
    rows = _validate_metrics(metrics)
    annotated: list[dict] = []
    for row in sorted(rows, key=lambda item: item["checkpoint_step"]):
        item = dict(row)
        item["schema_gate_pass"] = item["schema_valid_rate"] >= 0.995
        item["fpr_gate_pass"] = item["fpr"] <= 0.25
        item["eligible"] = item["schema_gate_pass"] and item["fpr_gate_pass"]
        annotated.append(item)

    eligible = sorted((row for row in annotated if row["eligible"]), key=_rank_key)
    selected = eligible[0]["checkpoint_step"] if eligible else None
    return {
        "protocol_version": "e1_dev_checkpoint_selection_v1",
        "selection_rule": {
            "schema_valid_rate_min": 0.995,
            "fpr_max": 0.25,
            "rank_order": ["recall_desc", "accuracy_desc", "f1_desc", "step_asc"],
        },
        "expected_steps": list(EXPECTED_STEPS),
        "eligible_steps": [row["checkpoint_step"] for row in eligible],
        "selected_step": selected,
        "test_unlocked": selected is not None,
        "checkpoints": annotated,
    }


def run_selection(root: Path, output_path: Path) -> dict:
    """Load all fixed checkpoint metrics and atomically write their selection summary."""
    root = Path(root)
    output_path = Path(output_path)
    if output_path.exists():
        raise SelectionError(f"output path already exists: {output_path}")
    rows: list[dict] = []
    for step in EXPECTED_STEPS:
        path = root / f"checkpoint-{step}" / "evaluation" / "metrics.json"
        if not path.is_file():
            raise SelectionError(f"missing checkpoint metrics: {path}")
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise SelectionError(f"invalid checkpoint metrics JSON: {path}") from exc
        if not isinstance(row, dict):
            raise SelectionError(f"checkpoint metrics must be an object: {path}")
        if row.get("checkpoint_step") != step:
            raise SelectionError(
                f"metric checkpoint_step={row.get('checkpoint_step')} does not match folder "
                f"checkpoint-{step}"
            )
        rows.append(row)

    summary = select_checkpoint(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_selection(args.root, args.output)
    except (SelectionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["test_unlocked"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
