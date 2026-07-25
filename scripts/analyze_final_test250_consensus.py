#!/usr/bin/env python3
"""Analyze four frozen Test250 predictions without changing labels or checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


DECISIONS = {"GOOD", "BAD"}


class ConsensusAnalysisError(ValueError):
    """Raised when Test250 or prediction artifacts violate the frozen contract."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConsensusAnalysisError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsensusAnalysisError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ConsensusAnalysisError(
                f"row at {path}:{line_number} must be an object"
            )
        rows.append(row)
    return rows


def _gold(row: dict, row_number: int) -> tuple[str, dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ConsensusAnalysisError(f"invalid Test messages at row {row_number}")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ConsensusAnalysisError(f"invalid Test label at row {row_number}") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in DECISIONS:
        raise ConsensusAnalysisError(f"invalid Test decision at row {row_number}")
    return payload["decision"], payload


def _image(row: dict, row_number: int) -> str:
    images = row.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0]
    ):
        raise ConsensusAnalysisError(f"invalid Test image at row {row_number}")
    return images[0]


def _prediction(row: dict, row_number: int, model: str, image: str) -> str:
    if row.get("index") != row_number - 1:
        raise ConsensusAnalysisError(
            f"invalid {model} index at row {row_number}: {row.get('index')}"
        )
    if row.get("image_path") != image:
        raise ConsensusAnalysisError(
            f"{model} image alignment mismatch at row {row_number}"
        )
    if row.get("schema_valid") is not True:
        raise ConsensusAnalysisError(
            f"{model} prediction is not schema-valid at row {row_number}"
        )
    decision = row.get("predicted_decision")
    if decision not in DECISIONS:
        raise ConsensusAnalysisError(
            f"invalid {model} prediction at row {row_number}"
        )
    return decision


def _confusion(golds: list[str], predictions: list[str]) -> dict[str, int]:
    result = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for gold, prediction in zip(golds, predictions):
        if gold == "BAD" and prediction == "BAD":
            result["tp"] += 1
        elif gold == "BAD":
            result["fn"] += 1
        elif prediction == "BAD":
            result["fp"] += 1
        else:
            result["tn"] += 1
    return result


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_consensus(
    test_rows: list[dict],
    predictions_by_model: dict[str, list[dict]],
    *,
    expected_count: int = 250,
) -> tuple[dict, dict[str, list[dict]]]:
    """Return reproducible vote/error statistics and auditable record lists."""
    if len(test_rows) != expected_count:
        raise ConsensusAnalysisError(
            f"expected {expected_count} Test rows, got {len(test_rows)}"
        )
    if len(predictions_by_model) != 4:
        raise ConsensusAnalysisError(
            f"expected four models, got {len(predictions_by_model)}"
        )
    for model, rows in predictions_by_model.items():
        if len(rows) != expected_count:
            raise ConsensusAnalysisError(
                f"expected {expected_count} {model} rows, got {len(rows)}"
            )

    model_names = list(predictions_by_model)
    predictions: dict[str, list[str]] = {model: [] for model in model_names}
    golds: list[str] = []
    records: dict[str, list[dict]] = {
        "all_records": [],
        "review_priority": [],
        "unanimous_against_gold": [],
        "unanimous_fn": [],
        "unanimous_fp": [],
        "three_of_four_against_gold": [],
        "split_two_two": [],
        "shared_errors_two_plus": [],
    }
    for model in model_names:
        records[f"{model}_unique_errors"] = []

    wrong_count_distribution: Counter[int] = Counter()
    bad_vote_distribution: Counter[int] = Counter()
    bad_vote_by_gold: dict[str, Counter[int]] = {
        "BAD": Counter(),
        "GOOD": Counter(),
    }
    stratum_counts: Counter[str] = Counter()
    pairwise_agreement: dict[str, dict[str, int]] = {
        model: {} for model in model_names
    }

    for row_number, test_row in enumerate(test_rows, start=1):
        image = _image(test_row, row_number)
        gold, gold_payload = _gold(test_row, row_number)
        golds.append(gold)
        row_predictions: dict[str, str] = {}
        payloads: dict[str, object] = {}
        for model in model_names:
            parsed = predictions_by_model[model][row_number - 1]
            decision = _prediction(parsed, row_number, model, image)
            predictions[model].append(decision)
            row_predictions[model] = decision
            payloads[model] = parsed.get("payload")

        wrong_models = [
            model for model, decision in row_predictions.items() if decision != gold
        ]
        correct_models = [
            model for model, decision in row_predictions.items() if decision == gold
        ]
        wrong_count = len(wrong_models)
        bad_votes = sum(decision == "BAD" for decision in row_predictions.values())
        wrong_count_distribution[wrong_count] += 1
        bad_vote_distribution[bad_votes] += 1
        bad_vote_by_gold[gold][bad_votes] += 1

        if wrong_count == 4:
            stratum = "unanimous_against_gold"
        elif wrong_count == 3:
            stratum = "three_of_four_against_gold"
        elif wrong_count == 2:
            stratum = "split_two_two"
        elif wrong_count == 1:
            stratum = "one_of_four_against_gold"
        else:
            stratum = "unanimous_with_gold"
        stratum_counts[stratum] += 1

        if bad_votes >= 3:
            consensus_decision = "BAD"
        elif bad_votes <= 1:
            consensus_decision = "GOOD"
        else:
            consensus_decision = None
        record = {
            "row": row_number,
            "index": row_number - 1,
            "image_path": image,
            "gold_decision": gold,
            "gold_payload": gold_payload,
            "predictions": row_predictions,
            "prediction_payloads": payloads,
            "bad_votes": bad_votes,
            "good_votes": 4 - bad_votes,
            "wrong_count": wrong_count,
            "wrong_models": wrong_models,
            "correct_models": correct_models,
            "consensus_decision": consensus_decision,
            "stratum": stratum,
        }
        records["all_records"].append(record)
        if wrong_count >= 2:
            records["shared_errors_two_plus"].append(record)
        if wrong_count in {2, 3, 4}:
            records["review_priority"].append(record)
        if wrong_count == 4:
            records["unanimous_against_gold"].append(record)
            records["unanimous_fn" if gold == "BAD" else "unanimous_fp"].append(
                record
            )
        elif wrong_count == 3:
            records["three_of_four_against_gold"].append(record)
        elif wrong_count == 2:
            records["split_two_two"].append(record)
        elif wrong_count == 1:
            records[f"{wrong_models[0]}_unique_errors"].append(record)

    for left in model_names:
        for right in model_names:
            pairwise_agreement[left][right] = sum(
                left_prediction == right_prediction
                for left_prediction, right_prediction in zip(
                    predictions[left], predictions[right]
                )
            )

    summary = {
        "protocol_version": "final_test250_four_model_consensus_v1",
        "total": expected_count,
        "models": model_names,
        "gold_counts": dict(sorted(Counter(golds).items())),
        "model_results": {
            model: {
                "confusion": _confusion(golds, predictions[model]),
                "predicted_counts": dict(
                    sorted(Counter(predictions[model]).items())
                ),
                "unique_errors": len(records[f"{model}_unique_errors"]),
                "unique_error_breakdown": {
                    "fn": sum(
                        row["gold_decision"] == "BAD"
                        for row in records[f"{model}_unique_errors"]
                    ),
                    "fp": sum(
                        row["gold_decision"] == "GOOD"
                        for row in records[f"{model}_unique_errors"]
                    ),
                },
            }
            for model in model_names
        },
        "strata": dict(sorted(stratum_counts.items())),
        "wrong_model_count_distribution": {
            str(value): wrong_count_distribution[value] for value in range(5)
        },
        "bad_vote_distribution": {
            str(value): bad_vote_distribution[value] for value in range(5)
        },
        "bad_vote_distribution_by_gold": {
            gold: {
                str(value): bad_vote_by_gold[gold][value] for value in range(5)
            }
            for gold in ("BAD", "GOOD")
        },
        "pairwise_decision_agreement": pairwise_agreement,
        "review_priority_rows": len(records["review_priority"]),
        "unanimous_fn": len(records["unanimous_fn"]),
        "unanimous_fp": len(records["unanimous_fp"]),
        "shared_errors_two_plus": {
            "total": len(records["shared_errors_two_plus"]),
            "fn": sum(
                row["gold_decision"] == "BAD"
                for row in records["shared_errors_two_plus"]
            ),
            "fp": sum(
                row["gold_decision"] == "GOOD"
                for row in records["shared_errors_two_plus"]
            ),
        },
        "analysis_only": True,
        "labels_modified": False,
        "checkpoint_selection_forbidden": True,
        "consensus_is_not_ground_truth": True,
        "gpu_required": False,
    }
    return summary, records


def run_analysis(
    test_path: Path,
    parsed_paths: dict[str, Path],
    output_dir: Path,
    *,
    expected_count: int = 250,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConsensusAnalysisError(
            f"output directory already exists: {output_dir}"
        )
    test_rows = _load_jsonl(test_path)
    prediction_rows = {
        model: _load_jsonl(path) for model, path in parsed_paths.items()
    }
    summary, records = analyze_consensus(
        test_rows,
        prediction_rows,
        expected_count=expected_count,
    )
    summary["test_path"] = str(test_path)
    summary["test_sha256"] = _sha256(test_path)
    summary["parsed_inputs"] = {
        model: {"path": str(path), "sha256": _sha256(path)}
        for model, path in parsed_paths.items()
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name, rows in records.items():
            (temporary / f"{name}.jsonl").write_text(
                _jsonl_text(rows), encoding="utf-8"
            )
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("prediction must use NAME=/path/to/parsed.jsonl")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=_parse_named_path,
        required=True,
        help="Repeat four times as NAME=/path/to/parsed.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=250)
    args = parser.parse_args()
    parsed_paths = dict(args.prediction)
    if len(parsed_paths) != len(args.prediction):
        raise ConsensusAnalysisError("prediction model names must be unique")

    summary = run_analysis(
        args.test,
        parsed_paths,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print("=== FINAL TEST250 FOUR-MODEL CONSENSUS ===")
    print(f"gold_counts={summary['gold_counts']}")
    for model, result in summary["model_results"].items():
        print(
            f"{model} confusion={result['confusion']} "
            f"predicted_counts={result['predicted_counts']} "
            f"unique_errors={result['unique_errors']}"
        )
    print(f"strata={summary['strata']}")
    print(
        f"review_priority_rows={summary['review_priority_rows']} "
        f"unanimous_fn={summary['unanimous_fn']} "
        f"unanimous_fp={summary['unanimous_fp']}"
    )
    print(f"pairwise_agreement={summary['pairwise_decision_agreement']}")
    print(f"summary={args.output_dir / 'summary.json'}")
    print("FINAL_TEST250_CONSENSUS_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
