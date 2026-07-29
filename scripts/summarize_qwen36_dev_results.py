#!/usr/bin/env python3
"""Summarize Qwen3.6 Base and E1-E5 selected fixed-Dev results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5")
METRIC_FIELDS = (
    "tp", "fn", "fp", "tn", "recall", "fpr", "accuracy", "f1",
    "schema_valid_rate",
)


class SummaryError(ValueError):
    """Raised when an evaluation artifact violates the comparison contract."""


def _read_object(path: Path) -> dict:
    if not path.is_file():
        raise SummaryError(f"missing result: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"result must be a JSON object: {path}")
    return value


def _metrics(row: dict, source: str) -> dict:
    missing = [field for field in METRIC_FIELDS if field not in row]
    if missing:
        raise SummaryError(f"{source} missing metric: {missing[0]}")
    if row.get("total") != 200:
        raise SummaryError(f"{source} must have total=200")
    return {field: row[field] for field in METRIC_FIELDS}


def build_comparison(root: Path, experiments: tuple[str, ...] = EXPERIMENTS) -> dict:
    root = Path(root)
    if not experiments or len(set(experiments)) != len(experiments):
        raise SummaryError("experiments must be a non-empty unique sequence")
    invalid = [name for name in experiments if name not in EXPERIMENTS]
    if invalid:
        raise SummaryError(f"invalid experiment: {invalid[0]}")

    base_source = root / "base_dev_natural_v1" / "evaluation" / "metrics.json"
    base_raw = _read_object(base_source)
    base = {
        "name": "BASE",
        "checkpoint_step": None,
        **_metrics(base_raw, "BASE"),
        "source": str(base_source),
    }

    rows: list[dict] = []
    for experiment in experiments:
        source = root / f"{experiment.lower()}_dev_8ckpt_v1" / "checkpoint-summary.json"
        summary = _read_object(source)
        step = summary.get("selected_step")
        checkpoints = summary.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 8:
            raise SummaryError(f"{experiment} must contain eight checkpoint metrics")
        selected = next(
            (row for row in checkpoints if row.get("checkpoint_step") == step), None
        )
        if step is None:
            if any(row.get("eligible") for row in checkpoints):
                raise SummaryError(
                    f"{experiment} has eligible checkpoints but selected_step is null"
                )
            rows.append({
                "name": experiment,
                "checkpoint_step": None,
                "eligible": False,
                "source": str(source),
            })
            continue
        if selected is None or not selected.get("eligible"):
            raise SummaryError(f"{experiment} selected checkpoint is not eligible")
        result = {
            "name": experiment,
            "checkpoint_step": step,
            "eligible": True,
            **_metrics(selected, experiment),
            "source": str(source),
        }
        result["delta_vs_base"] = {
            field: result[field] - base[field]
            for field in ("recall", "fpr", "accuracy", "f1", "schema_valid_rate")
        }
        rows.append(result)

    eligible = [row for row in rows if row.get("eligible")]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["recall"], -row["accuracy"], -row["f1"], row["name"]
        ),
    )
    recommended = None
    if ranked:
        recommended = {
            "experiment": ranked[0]["name"],
            "checkpoint_step": ranked[0]["checkpoint_step"],
        }
    return {
        "protocol_version": "qwen36_fixed_dev_comparison_v1",
        "selection_rule": {
            "per_experiment_gates": {"schema_valid_rate_min": 0.995, "fpr_max": 0.25},
            "overall_rank_order": ["recall_desc", "accuracy_desc", "f1_desc", "experiment_asc"],
        },
        "base": base,
        "experiments": rows,
        "recommended": recommended,
        "test_evaluated": False,
    }


def print_table(summary: dict) -> None:
    print("=== QWEN3.6 FIXED DEV COMPARISON ===")
    print("model\tstep\tTP\tFN\tFP\tTN\tRecall\tFPR\tAccuracy\tF1\tJSON")
    rows = [summary["base"], *summary["experiments"]]
    for row in rows:
        if row.get("checkpoint_step") is None and row["name"] != "BASE":
            print(f"{row['name']}\tNONE\t-\t-\t-\t-\t-\t-\t-\t-\t-")
            continue
        step = "BASE" if row["name"] == "BASE" else row["checkpoint_step"]
        print(
            f"{row['name']}\t{step}\t{row['tp']}\t{row['fn']}\t{row['fp']}\t{row['tn']}\t"
            f"{row['recall']:.2%}\t{row['fpr']:.2%}\t{row['accuracy']:.2%}\t"
            f"{row['f1']:.2%}\t{row['schema_valid_rate']:.2%}"
        )
    recommended = summary["recommended"]
    if recommended is None:
        print("RECOMMENDED: NONE")
    else:
        print(
            f"RECOMMENDED: {recommended['experiment']} "
            f"checkpoint-{recommended['checkpoint_step']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_comparison(args.root, tuple(args.experiments))
        if args.output.exists():
            raise SummaryError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (SummaryError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_table(summary)
    print(f"summary={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
