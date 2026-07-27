#!/usr/bin/env python3
"""Merge the 73-row blind review and rescore frozen Test250 predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


DECISIONS = {"GOOD", "BAD"}


class ReadjudicationError(ValueError):
    """Raised when review or prediction artifacts do not align exactly."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ReadjudicationError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReadjudicationError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ReadjudicationError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _load_annotations(path: Path) -> dict[int, dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadjudicationError(f"cannot read annotations: {path}") from exc
    if not isinstance(value, dict):
        raise ReadjudicationError("annotations must be an object")
    result: dict[int, dict] = {}
    for key, annotation in value.items():
        try:
            row = int(key)
        except ValueError as exc:
            raise ReadjudicationError(f"invalid annotation row: {key}") from exc
        if not isinstance(annotation, dict):
            raise ReadjudicationError(f"annotation {key} must be an object")
        result[row] = annotation
    return result


def _gold(row: dict, row_number: int) -> tuple[str, dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ReadjudicationError(f"invalid Test messages at row {row_number}")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReadjudicationError(f"invalid Test label at row {row_number}") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in DECISIONS:
        raise ReadjudicationError(f"invalid Test decision at row {row_number}")
    return payload["decision"], payload


def _image(row: dict, row_number: int) -> str:
    images = row.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0]
    ):
        raise ReadjudicationError(f"invalid Test image at row {row_number}")
    return images[0]


def _prediction(row: dict, row_number: int, model: str, image: str) -> str:
    if row.get("index") != row_number - 1 or row.get("image_path") != image:
        raise ReadjudicationError(f"{model} alignment mismatch at row {row_number}")
    if row.get("schema_valid") is not True:
        raise ReadjudicationError(
            f"{model} prediction is not schema-valid at row {row_number}"
        )
    decision = row.get("predicted_decision")
    if decision not in DECISIONS:
        raise ReadjudicationError(f"invalid {model} prediction at row {row_number}")
    return decision


def _metrics(golds: list[str], predictions: list[str]) -> dict:
    if len(golds) != len(predictions):
        raise ReadjudicationError("gold and prediction lengths differ")
    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for gold, prediction in zip(golds, predictions):
        if gold == "BAD" and prediction == "BAD":
            confusion["tp"] += 1
        elif gold == "BAD":
            confusion["fn"] += 1
        elif prediction == "BAD":
            confusion["fp"] += 1
        else:
            confusion["tn"] += 1
    tp, fn, fp, tn = (confusion[key] for key in ("tp", "fn", "fp", "tn"))
    total = len(golds)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return {
        "total": total,
        **confusion,
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, total),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def _delta(after: dict, before: dict) -> dict:
    return {
        key: after[key] - before[key]
        for key in ("tp", "fn", "fp", "tn", "recall", "fpr", "accuracy", "f1")
    }


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def analyze_readjudication(
    test_rows: list[dict],
    review_rows: list[dict],
    consensus_rows: list[dict],
    annotations: dict[int, dict],
    predictions_by_model: dict[str, list[dict]],
    *,
    expected_test_count: int = 250,
    expected_review_count: int = 73,
    expected_original_confusions: dict[str, dict[str, int]] | None = None,
) -> tuple[dict, dict[str, list[dict]], list[dict]]:
    if len(test_rows) != expected_test_count:
        raise ReadjudicationError(
            f"expected {expected_test_count} Test rows, got {len(test_rows)}"
        )
    if len(review_rows) != expected_review_count:
        raise ReadjudicationError(
            f"expected {expected_review_count} review rows, got {len(review_rows)}"
        )
    if len(consensus_rows) != expected_review_count:
        raise ReadjudicationError(
            f"expected {expected_review_count} consensus rows, got {len(consensus_rows)}"
        )
    review_by_row = {row.get("row"): row for row in review_rows}
    consensus_by_row = {row.get("row"): row for row in consensus_rows}
    if len(review_by_row) != expected_review_count or set(review_by_row) != set(consensus_by_row):
        raise ReadjudicationError("review and consensus row sets do not match")
    if set(annotations) != set(review_by_row):
        missing = sorted(set(review_by_row) - set(annotations))
        extra = sorted(set(annotations) - set(review_by_row))
        raise ReadjudicationError(
            f"annotations do not cover all review rows: missing={missing} extra={extra}"
        )
    for model, rows in predictions_by_model.items():
        if len(rows) != expected_test_count:
            raise ReadjudicationError(
                f"expected {expected_test_count} {model} rows, got {len(rows)}"
            )

    original_golds: list[str] = []
    adjusted_golds: list[str | None] = []
    images: list[str] = []
    adjusted_test_rows: list[dict] = []
    records: dict[str, list[dict]] = {
        "reviewed_records": [],
        "label_changes": [],
        "excluded_unsure": [],
        "consistency_issues": [],
    }
    review_decisions: Counter[str] = Counter()
    severities: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    stratum_outcomes: Counter[str] = Counter()

    for row_number, test_row in enumerate(test_rows, start=1):
        image = _image(test_row, row_number)
        original, original_payload = _gold(test_row, row_number)
        images.append(image)
        original_golds.append(original)
        adjusted: str | None = original
        annotation = annotations.get(row_number)
        review_record = review_by_row.get(row_number)
        consensus = consensus_by_row.get(row_number)
        if annotation is not None:
            if annotation.get("completed") is not True:
                raise ReadjudicationError(f"review row {row_number} is not completed")
            decision = annotation.get("review_decision")
            severity = annotation.get("visible_severity")
            if decision not in {"GOOD", "BAD", "UNSURE"}:
                raise ReadjudicationError(f"invalid review decision at row {row_number}")
            if severity not in {"obvious", "borderline", "none", "uncertain"}:
                raise ReadjudicationError(f"invalid severity at row {row_number}")
            if review_record.get("image_path") != image:
                raise ReadjudicationError(f"review image mismatch at row {row_number}")
            review_decisions[decision] += 1
            severities[severity] += 1
            if decision == "UNSURE":
                adjusted = None
                if severity != "uncertain":
                    records["consistency_issues"].append(
                        {
                            "row": row_number,
                            "code": "unsure_with_non_uncertain_severity",
                            "severity": severity,
                        }
                    )
            elif severity == "uncertain":
                records["consistency_issues"].append(
                    {
                        "row": row_number,
                        "code": "binary_decision_with_uncertain_severity",
                        "review_decision": decision,
                    }
                )
                adjusted = decision
            else:
                adjusted = decision
            transition = f"{original}->{adjusted or 'EXCLUDED'}"
            transitions[transition] += 1
            stratum = consensus.get("stratum")
            stratum_outcomes[f"{stratum}:{transition}"] += 1
            record = {
                "row": row_number,
                "review_order": review_record.get("review_order"),
                "image_path": image,
                "original_decision": original,
                "review_decision": decision,
                "adjusted_decision": adjusted,
                "visible_severity": severity,
                "categories": annotation.get("categories") or [],
                "notes": annotation.get("notes") or "",
                "consensus_stratum": stratum,
                "wrong_count_under_original_gold": consensus.get("wrong_count"),
            }
            records["reviewed_records"].append(record)
            if adjusted is None:
                records["excluded_unsure"].append(record)
            elif adjusted != original:
                records["label_changes"].append(record)
        adjusted_golds.append(adjusted)

        if adjusted is not None:
            adjusted_row = copy.deepcopy(test_row)
            if adjusted == "GOOD":
                payload = {"decision": "GOOD", "categories": [], "reasons": []}
            elif annotation is not None:
                categories = (annotation.get("categories") or ["其他"])[:3]
                reason = (annotation.get("notes") or "人工复核判定为有异常").strip()
                payload = {
                    "decision": "BAD",
                    "categories": categories,
                    "reasons": [reason],
                }
            else:
                payload = original_payload
            adjusted_row["messages"][-1]["content"] = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            adjusted_test_rows.append(adjusted_row)

    predictions: dict[str, list[str]] = {}
    for model, parsed_rows in predictions_by_model.items():
        predictions[model] = [
            _prediction(parsed, row_number, model, images[row_number - 1])
            for row_number, parsed in enumerate(parsed_rows, start=1)
        ]

    included_indices = [
        index for index, decision in enumerate(adjusted_golds) if decision is not None
    ]
    reviewed_indices = sorted(row - 1 for row in review_by_row)
    reviewed_included_indices = [
        index for index in reviewed_indices if adjusted_golds[index] is not None
    ]
    original_same_population = [original_golds[index] for index in included_indices]
    adjusted_population = [adjusted_golds[index] for index in included_indices]
    model_results = {}
    paired_changes: dict[str, dict[str, int]] = {}
    for model, model_predictions in predictions.items():
        original_full = _metrics(original_golds, model_predictions)
        expected = (expected_original_confusions or {}).get(model)
        if expected is not None and {
            key: original_full[key] for key in ("tp", "fn", "fp", "tn")
        } != expected:
            raise ReadjudicationError(
                f"{model} original confusion mismatch: {original_full} != {expected}"
            )
        same_predictions = [model_predictions[index] for index in included_indices]
        original_same = _metrics(original_same_population, same_predictions)
        adjusted = _metrics(adjusted_population, same_predictions)
        reviewed_original = _metrics(
            [original_golds[index] for index in reviewed_included_indices],
            [model_predictions[index] for index in reviewed_included_indices],
        )
        reviewed_adjusted = _metrics(
            [adjusted_golds[index] for index in reviewed_included_indices],
            [model_predictions[index] for index in reviewed_included_indices],
        )
        paired = Counter()
        for index in included_indices:
            before = model_predictions[index] == original_golds[index]
            after = model_predictions[index] == adjusted_golds[index]
            paired[
                "both_correct"
                if before and after
                else "became_wrong"
                if before
                else "became_correct"
                if after
                else "both_wrong"
            ] += 1
        paired_changes[model] = dict(sorted(paired.items()))
        model_results[model] = {
            "original_full_250": original_full,
            "original_same_population_excluding_unsure": original_same,
            "adjusted_excluding_unsure": adjusted,
            "adjusted_minus_original_same_population": _delta(adjusted, original_same),
            "reviewed_subset_original_excluding_unsure": reviewed_original,
            "reviewed_subset_adjusted_excluding_unsure": reviewed_adjusted,
            "paired_correctness_change": paired_changes[model],
        }

    summary = {
        "protocol_version": "final_test250_priority73_conditional_readjudication_v1",
        "original_total": expected_test_count,
        "reviewed": expected_review_count,
        "unreviewed_original_labels_retained": expected_test_count - expected_review_count,
        "review_decision_counts": dict(sorted(review_decisions.items())),
        "severity_counts": dict(sorted(severities.items())),
        "transitions": dict(sorted(transitions.items())),
        "label_changes": len(records["label_changes"]),
        "excluded_unsure": len(records["excluded_unsure"]),
        "included_total": len(included_indices),
        "original_included_gold_counts": dict(
            sorted(Counter(original_same_population).items())
        ),
        "adjusted_gold_counts": dict(sorted(Counter(adjusted_population).items())),
        "stratum_transition_counts": dict(sorted(stratum_outcomes.items())),
        "consistency_issues": records["consistency_issues"],
        "model_results": model_results,
        "labels_modified_in_source_v1": False,
        "selection_is_model_informed": True,
        "checkpoint_selection_forbidden": True,
        "interpretation": (
            "conditional diagnostic rescore; 177 unreviewed rows retain original labels; "
            "excluded UNSURE rows are outside every adjusted metric denominator"
        ),
    }
    return summary, records, adjusted_test_rows


def run_analysis(
    test_path: Path,
    review_path: Path,
    consensus_path: Path,
    annotations_path: Path,
    parsed_paths: dict[str, Path],
    output_dir: Path,
    *,
    expected_test_count: int = 250,
    expected_review_count: int = 73,
    expected_original_confusions: dict[str, dict[str, int]] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ReadjudicationError(f"output directory already exists: {output_dir}")
    summary, records, adjusted_test_rows = analyze_readjudication(
        _load_jsonl(test_path),
        _load_jsonl(review_path),
        _load_jsonl(consensus_path),
        _load_annotations(annotations_path),
        {model: _load_jsonl(path) for model, path in parsed_paths.items()},
        expected_test_count=expected_test_count,
        expected_review_count=expected_review_count,
        expected_original_confusions=expected_original_confusions,
    )
    adjusted_text = _jsonl_text(adjusted_test_rows)
    summary["inputs"] = {
        "test": str(test_path),
        "review": str(review_path),
        "consensus": str(consensus_path),
        "annotations": str(annotations_path),
        "parsed": {model: str(path) for model, path in parsed_paths.items()},
    }
    summary["adjusted_test_sha256"] = _sha256_bytes(adjusted_text.encode("utf-8"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name, rows in records.items():
            (staging / f"{name}.jsonl").write_text(
                _jsonl_text(rows), encoding="utf-8"
            )
        (staging / "test_conditionally_readjudicated.jsonl").write_text(
            adjusted_text, encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("prediction must use NAME=/path/to/parsed.jsonl")
    return name, Path(path)


def _expected_confusion(value: str) -> tuple[str, dict[str, int]]:
    name, separator, numbers = value.partition("=")
    try:
        tp, fn, fp, tn = (int(item) for item in numbers.split(","))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected confusion must use NAME=TP,FN,FP,TN"
        ) from exc
    if not separator or not name:
        raise argparse.ArgumentTypeError(
            "expected confusion must use NAME=TP,FN,FP,TN"
        )
    return name, {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=_named_path, required=True)
    parser.add_argument(
        "--expected-confusion", action="append", type=_expected_confusion, default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parsed_paths = dict(args.prediction)
    expected = dict(args.expected_confusion)
    if len(parsed_paths) != len(args.prediction):
        raise ReadjudicationError("prediction model names must be unique")
    summary = run_analysis(
        args.test,
        args.review,
        args.consensus,
        args.annotations,
        parsed_paths,
        args.output_dir,
        expected_original_confusions=expected,
    )
    print("=== FINAL TEST250 PRIORITY73 READJUDICATION ===")
    print(
        f"reviewed={summary['reviewed']} label_changes={summary['label_changes']} "
        f"excluded_unsure={summary['excluded_unsure']} included={summary['included_total']}"
    )
    print(f"review_decisions={summary['review_decision_counts']}")
    print(f"severity={summary['severity_counts']}")
    print(f"transitions={summary['transitions']}")
    print(f"adjusted_gold_counts={summary['adjusted_gold_counts']}")
    if summary["consistency_issues"]:
        print(f"consistency_issues={summary['consistency_issues']}")
    print("=== RESCORED CHECKPOINTS (EXCLUDING UNSURE) ===")
    for model, result in summary["model_results"].items():
        metrics = result["adjusted_excluding_unsure"]
        delta = result["adjusted_minus_original_same_population"]
        print(
            f"{model} N={metrics['total']} TP={metrics['tp']} FN={metrics['fn']} "
            f"FP={metrics['fp']} TN={metrics['tn']} Recall={metrics['recall']:.2%} "
            f"FPR={metrics['fpr']:.2%} Accuracy={metrics['accuracy']:.2%} "
            f"F1={metrics['f1']:.2%} dAcc={delta['accuracy']:+.2%} "
            f"dF1={delta['f1']:+.2%}"
        )
    print(f"adjusted_test_sha256={summary['adjusted_test_sha256']}")
    print(f"summary={args.output_dir / 'summary.json'}")
    print("FINAL_TEST250_PRIORITY_READJUDICATION: PASS")


if __name__ == "__main__":
    main()
