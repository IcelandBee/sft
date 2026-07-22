#!/usr/bin/env python3
"""Rescore an existing Base inference result against the frozen corrected Dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

try:
    from .evaluate_e1_dev import parse_prediction, validate_payload
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_e1_dev import parse_prediction, validate_payload  # type: ignore


class BaseRescoreError(ValueError):
    """Raised when Base results and Dev artifacts do not align."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BaseRescoreError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaseRescoreError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise BaseRescoreError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _dev_fields(row: dict, row_number: int, name: str) -> tuple[str, list[dict], dict]:
    images = row.get("images")
    messages = row.get("messages")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise BaseRescoreError(f"invalid {name} image at row {row_number}")
    if not isinstance(messages, list) or len(messages) != 3:
        raise BaseRescoreError(f"invalid {name} messages at row {row_number}")
    try:
        gold = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BaseRescoreError(f"invalid {name} gold at row {row_number}") from exc
    schema_error = validate_payload(gold)
    if schema_error is not None:
        raise BaseRescoreError(
            f"{name} gold at row {row_number} violates schema: {schema_error}"
        )
    return images[0], messages, gold


def _result_image(row: dict, row_number: int) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise BaseRescoreError(f"invalid Base image at row {row_number}")
    image = images[0]
    if isinstance(image, dict):
        image = image.get("path")
    if not isinstance(image, str) or not image:
        raise BaseRescoreError(f"invalid Base image path at row {row_number}")
    return image


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(gold: list[str], predicted: list[str | None]) -> dict:
    if len(gold) != len(predicted):
        raise BaseRescoreError("gold and prediction lengths do not match")
    tp = fn = fp = tn = 0
    for truth, decision in zip(gold, predicted):
        if truth == "BAD" and decision == "BAD":
            tp += 1
        elif truth == "BAD":
            fn += 1
        elif decision == "GOOD":
            tn += 1
        else:
            # Invalid output is conservatively counted as wrong.
            fp += 1
    return {
        "total": len(gold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, len(gold)),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "unusable": sum(decision is None for decision in predicted),
    }


def rescore_rows(
    original_dev: list[dict],
    corrected_dev: list[dict],
    result_rows: list[dict],
    *,
    expected_count: int = 200,
    enforce_original_contract: bool = True,
) -> tuple[dict, list[dict]]:
    """Validate alignment and return Base metrics under both label versions."""
    for name, rows in (
        ("original Dev", original_dev),
        ("corrected Dev", corrected_dev),
        ("Base result", result_rows),
    ):
        if len(rows) != expected_count:
            raise BaseRescoreError(
                f"expected {expected_count} {name} rows, got {len(rows)}"
            )

    original_gold: list[str] = []
    corrected_gold: list[str] = []
    strict_predictions: list[str | None] = []
    decision_predictions: list[str | None] = []
    parsed_records: list[dict] = []
    envelope_valid = payload_valid = schema_valid = 0

    for row_number, (original, corrected, result) in enumerate(
        zip(original_dev, corrected_dev, result_rows), start=1
    ):
        image, original_messages, old_gold = _dev_fields(
            original, row_number, "original Dev"
        )
        corrected_image, corrected_messages, new_gold = _dev_fields(
            corrected, row_number, "corrected Dev"
        )
        if corrected_image != image:
            raise BaseRescoreError(f"Dev image drift at row {row_number}")
        if original_messages[:2] != corrected_messages[:2]:
            raise BaseRescoreError(f"Dev prompt drift at row {row_number}")
        if _result_image(result, row_number) != image:
            raise BaseRescoreError(f"Base result image mismatch at row {row_number}")

        labels = result.get("labels")
        if not isinstance(labels, str):
            raise BaseRescoreError(f"missing Base labels at row {row_number}")
        try:
            result_gold = json.loads(labels)
        except json.JSONDecodeError as exc:
            raise BaseRescoreError(f"invalid Base labels at row {row_number}") from exc
        if result_gold != old_gold:
            raise BaseRescoreError(f"Base labels do not match original Dev at row {row_number}")

        parsed = parse_prediction(result.get("response"))
        envelope_valid += parsed["envelope_valid"] is True
        payload_valid += parsed["payload_json_valid"] is True
        schema_valid += parsed["schema_valid"] is True
        strict_decision = parsed["decision"] if parsed["schema_valid"] else None
        payload = parsed.get("payload")
        decision_only = (
            payload.get("decision")
            if isinstance(payload, dict) and payload.get("decision") in {"GOOD", "BAD"}
            else None
        )

        original_gold.append(old_gold["decision"])
        corrected_gold.append(new_gold["decision"])
        strict_predictions.append(strict_decision)
        decision_predictions.append(decision_only)
        parsed_records.append(
            {
                "row": row_number,
                "image_path": image,
                "original_gold": old_gold["decision"],
                "corrected_gold": new_gold["decision"],
                "strict_decision": strict_decision,
                "decision_only": decision_only,
                "schema_valid": parsed["schema_valid"],
                "error_code": parsed["error_code"],
            }
        )

    summary = {
        "protocol_version": "base_adjudicated_dev_rescore_v1",
        "total": expected_count,
        "validity": {
            "envelope_valid_rate": _ratio(envelope_valid, expected_count),
            "payload_json_valid_rate": _ratio(payload_valid, expected_count),
            "schema_valid_rate": _ratio(schema_valid, expected_count),
        },
        "original_dev": {
            "strict_schema": _metrics(original_gold, strict_predictions),
            "decision_only": _metrics(original_gold, decision_predictions),
        },
        "corrected_dev": {
            "strict_schema": _metrics(corrected_gold, strict_predictions),
            "decision_only": _metrics(corrected_gold, decision_predictions),
        },
        "note": "decision_only ignores categories/reasons schema but still counts unusable output as wrong",
    }

    expected_original_strict = {"tp": 15, "fn": 36, "fp": 93, "tn": 56}
    expected_original_decision = {
        "tp": 15,
        "fn": 36,
        "fp": 65,
        "tn": 84,
        "unusable": 35,
    }
    strict = summary["original_dev"]["strict_schema"]
    decision = summary["original_dev"]["decision_only"]
    if enforce_original_contract:
        if any(strict[key] != value for key, value in expected_original_strict.items()):
            raise BaseRescoreError(f"original strict metric contract drift: {strict}")
        if any(
            decision[key] != value
            for key, value in expected_original_decision.items()
        ):
            raise BaseRescoreError(
                f"original decision-only metric contract drift: {decision}"
            )
    return summary, parsed_records


def _jsonl_text(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def run_rescore(
    original_dev_path: Path,
    corrected_dev_path: Path,
    result_path: Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise BaseRescoreError(f"output directory already exists: {output_dir}")
    summary, rows = rescore_rows(
        _load_jsonl(original_dev_path),
        _load_jsonl(corrected_dev_path),
        _load_jsonl(result_path),
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
        (staging / "parsed.jsonl").write_text(
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


def _print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}: TP={metrics['tp']} FN={metrics['fn']} "
        f"FP={metrics['fp']} TN={metrics['tn']} unusable={metrics['unusable']} "
        f"Recall={_percent(metrics['recall'])} FPR={_percent(metrics['fpr'])} "
        f"Accuracy={_percent(metrics['accuracy'])} F1={_percent(metrics['f1'])}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dev", required=True, type=Path)
    parser.add_argument("--corrected-dev", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_rescore(
            args.original_dev, args.corrected_dev, args.result, args.output_dir
        )
    except (BaseRescoreError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== BASE VALIDITY ===")
    print(summary["validity"])
    print("=== ORIGINAL DEV CONTRACT CHECK ===")
    _print_metrics("strict_schema", summary["original_dev"]["strict_schema"])
    _print_metrics("decision_only", summary["original_dev"]["decision_only"])
    print("=== CORRECTED DEV ===")
    _print_metrics("strict_schema", summary["corrected_dev"]["strict_schema"])
    _print_metrics("decision_only", summary["corrected_dev"]["decision_only"])
    print("BASE_ADJUDICATED_RESCORE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
