#!/usr/bin/env python3
"""Select simple voting rules on Dev, then apply one frozen rule to Test."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable


MODEL_NAMES = ("e1-1248", "e2-1248", "e5-780-recall", "e5-975-balanced")


class EnsembleStudyError(ValueError):
    """Raised when datasets, predictions, or rule selection are invalid."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EnsembleStudyError(f"cannot read JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnsembleStudyError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise EnsembleStudyError(f"row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset(rows: list[dict], name: str, expected_count: int) -> list[dict]:
    if len(rows) != expected_count:
        raise EnsembleStudyError(
            f"expected {expected_count} {name} rows, got {len(rows)}"
        )
    result: list[dict] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        images = row.get("images")
        messages = row.get("messages")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
            or images[0] in seen
        ):
            raise EnsembleStudyError(f"invalid {name} image at row {row_number}")
        try:
            gold = json.loads(messages[-1]["content"])["decision"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise EnsembleStudyError(f"invalid {name} gold at row {row_number}") from exc
        if gold not in {"GOOD", "BAD"}:
            raise EnsembleStudyError(f"invalid {name} decision at row {row_number}")
        seen.add(images[0])
        result.append({"row": row_number, "image_path": images[0], "gold": gold})
    return result


def _prediction_map(rows: list[dict], model: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        image = row.get("image_path")
        decision = row.get("predicted_decision")
        if not isinstance(image, str) or not image or image in result:
            raise EnsembleStudyError(f"invalid {model} image at parsed row {row_number}")
        if row.get("schema_valid") is not True or decision not in {"GOOD", "BAD"}:
            raise EnsembleStudyError(
                f"invalid {model} prediction at parsed row {row_number}"
            )
        result[image] = decision
    return result


def _aligned_predictions(
    dataset: list[dict], predictions_by_model: dict[str, list[dict]]
) -> dict[str, list[str]]:
    if set(predictions_by_model) != set(MODEL_NAMES):
        raise EnsembleStudyError(
            f"model set mismatch: {sorted(predictions_by_model)}"
        )
    aligned: dict[str, list[str]] = {}
    for model in MODEL_NAMES:
        by_image = _prediction_map(predictions_by_model[model], model)
        missing = [row["image_path"] for row in dataset if row["image_path"] not in by_image]
        if missing:
            raise EnsembleStudyError(
                f"{model} lacks {len(missing)} dataset predictions"
            )
        aligned[model] = [by_image[row["image_path"]] for row in dataset]
    return aligned


def _metrics(golds: list[str], predictions: list[str]) -> dict:
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
    recall = ratio(tp, tp + fn)
    fpr = ratio(fp, fp + tn)
    accuracy = ratio(tp + tn, total)
    return {
        "total": total,
        **confusion,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "go_pass": recall >= 0.72 and fpr < 0.25 and accuracy >= 0.72,
        "challenge_pass": recall >= 0.78 and fpr < 0.20 and accuracy >= 0.76,
    }


def _bad(votes: dict[str, str], model: str) -> int:
    return int(votes[model] == "BAD")


def _rules() -> dict[str, tuple[str, tuple[str, ...], Callable[[dict[str, str]], bool]]]:
    e1, e2, e5r, e5b = MODEL_NAMES
    return {
        "three_majority": (
            "BAD if at least 2 of E1/E5-780/E5-975 vote BAD",
            (e1, e5r, e5b),
            lambda vote: sum(_bad(vote, model) for model in (e1, e5r, e5b)) >= 2,
        ),
        "four_at_least_two": (
            "BAD if at least 2 of all 4 models vote BAD",
            MODEL_NAMES,
            lambda vote: sum(_bad(vote, model) for model in MODEL_NAMES) >= 2,
        ),
        "four_at_least_three": (
            "BAD if at least 3 of all 4 models vote BAD",
            MODEL_NAMES,
            lambda vote: sum(_bad(vote, model) for model in MODEL_NAMES) >= 3,
        ),
        "e5_recall_confirmed": (
            "BAD if E5-780 is BAD and E1 or E5-975 confirms BAD",
            (e1, e5r, e5b),
            lambda vote: _bad(vote, e5r) == 1
            and (_bad(vote, e1) == 1 or _bad(vote, e5b) == 1),
        ),
        "balanced_or_double_confirmed": (
            "BAD if E5-975 is BAD, or both E1 and E5-780 are BAD",
            (e1, e5r, e5b),
            lambda vote: _bad(vote, e5b) == 1
            or (_bad(vote, e1) == 1 and _bad(vote, e5r) == 1),
        ),
        "weighted_balanced": (
            "weights E1=1,E2=1,E5-780=1,E5-975=2; BAD if score>=3",
            MODEL_NAMES,
            lambda vote: _bad(vote, e1)
            + _bad(vote, e2)
            + _bad(vote, e5r)
            + 2 * _bad(vote, e5b)
            >= 3,
        ),
        "weighted_recall": (
            "weights E1=1,E2=1,E5-780=2,E5-975=2; BAD if score>=3",
            MODEL_NAMES,
            lambda vote: _bad(vote, e1)
            + _bad(vote, e2)
            + 2 * _bad(vote, e5r)
            + 2 * _bad(vote, e5b)
            >= 3,
        ),
    }


def _apply_rule(
    aligned: dict[str, list[str]], rule: Callable[[dict[str, str]], bool]
) -> list[str]:
    count = len(next(iter(aligned.values())))
    return [
        "BAD"
        if rule({model: aligned[model][index] for model in MODEL_NAMES})
        else "GOOD"
        for index in range(count)
    ]


def _selection_key(result: dict) -> tuple:
    return (
        result["challenge_pass"],
        result["go_pass"],
        result["accuracy"],
        result["f1"],
        result["recall"],
        -result["fpr"],
        -len(result["required_models"]),
        result["rule"],
    )


def study_ensemble(
    dev_dataset: list[dict],
    dev_predictions: dict[str, list[dict]],
    test_dataset: list[dict],
    test_predictions: dict[str, list[dict]],
) -> tuple[dict, list[dict]]:
    dev_aligned = _aligned_predictions(dev_dataset, dev_predictions)
    test_aligned = _aligned_predictions(test_dataset, test_predictions)
    dev_golds = [row["gold"] for row in dev_dataset]
    test_golds = [row["gold"] for row in test_dataset]
    rules = _rules()
    dev_results = []
    for name, (expression, required_models, rule) in rules.items():
        metrics = _metrics(dev_golds, _apply_rule(dev_aligned, rule))
        dev_results.append(
            {
                "rule": name,
                "expression": expression,
                "required_models": list(required_models),
                **metrics,
            }
        )
    selected = max(dev_results, key=_selection_key)
    expression, required_models, rule = rules[selected["rule"]]
    test_ensemble_predictions = _apply_rule(test_aligned, rule)
    test_metrics = _metrics(test_golds, test_ensemble_predictions)

    single_dev = {
        model: _metrics(dev_golds, dev_aligned[model]) for model in MODEL_NAMES
    }
    single_test = {
        model: _metrics(test_golds, test_aligned[model]) for model in MODEL_NAMES
    }
    records = []
    for index, row in enumerate(test_dataset):
        votes = {model: test_aligned[model][index] for model in MODEL_NAMES}
        prediction = test_ensemble_predictions[index]
        records.append(
            {
                **row,
                "model_votes": votes,
                "selected_rule": selected["rule"],
                "ensemble_prediction": prediction,
                "is_error": prediction != row["gold"],
            }
        )
    summary = {
        "protocol_version": "dev_selected_simple_ensemble_v1",
        "selection_dataset_total": len(dev_dataset),
        "test_dataset_total": len(test_dataset),
        "selection_policy": (
            "challenge_pass, then go_pass, then accuracy, F1, recall, lower FPR, "
            "fewer required models; Dev only"
        ),
        "dev_rule_results": sorted(dev_results, key=_selection_key, reverse=True),
        "selected_rule": selected,
        "dev_single_models": single_dev,
        "test_selected_ensemble": test_metrics,
        "test_single_models": single_test,
        "selected_expression": expression,
        "selected_required_models": list(required_models),
        "selected_on_dev_only": True,
        "test_previously_observed": True,
        "test_result_status": "exploratory; requires a new untouched holdout for confirmation",
        "checkpoint_selection_forbidden": True,
        "gpu_required": False,
    }
    return summary, records


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def run_study(
    dev_path: Path,
    dev_parsed: dict[str, Path],
    test_path: Path,
    test_parsed: dict[str, Path],
    output_dir: Path,
    *,
    expected_dev_sha256: str,
    expected_test_sha256: str,
    expected_dev_count: int = 200,
    expected_test_count: int = 241,
) -> dict:
    if _sha256(dev_path) != expected_dev_sha256:
        raise EnsembleStudyError("Dev sha256 mismatch")
    if _sha256(test_path) != expected_test_sha256:
        raise EnsembleStudyError("Test sha256 mismatch")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise EnsembleStudyError(f"output directory already exists: {output_dir}")
    summary, records = study_ensemble(
        _dataset(_load_jsonl(dev_path), "Dev", expected_dev_count),
        {model: _load_jsonl(path) for model, path in dev_parsed.items()},
        _dataset(_load_jsonl(test_path), "Test", expected_test_count),
        {model: _load_jsonl(path) for model, path in test_parsed.items()},
    )
    summary["inputs"] = {
        "dev": str(dev_path),
        "dev_sha256": expected_dev_sha256,
        "dev_predictions": {model: str(path) for model, path in dev_parsed.items()},
        "test": str(test_path),
        "test_sha256": expected_test_sha256,
        "test_predictions": {model: str(path) for model, path in test_parsed.items()},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "test_predictions.jsonl").write_text(
            _jsonl_text(records), encoding="utf-8"
        )
        (staging / "test_errors.jsonl").write_text(
            _jsonl_text(row for row in records if row["is_error"]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dev-prediction", action="append", type=_named_path, required=True)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-prediction", action="append", type=_named_path, required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_study(
        args.dev,
        dict(args.dev_prediction),
        args.test,
        dict(args.test_prediction),
        args.output_dir,
        expected_dev_sha256=args.expected_dev_sha256,
        expected_test_sha256=args.expected_test_sha256,
    )
    print("=== DEV-SELECTED SIMPLE ENSEMBLE ===")
    for result in summary["dev_rule_results"]:
        print(
            f"DEV {result['rule']} models={len(result['required_models'])} "
            f"R={result['recall']:.2%} FPR={result['fpr']:.2%} "
            f"Acc={result['accuracy']:.2%} F1={result['f1']:.2%} "
            f"Go={result['go_pass']} Challenge={result['challenge_pass']}"
        )
    selected = summary["selected_rule"]
    test = summary["test_selected_ensemble"]
    print("=== FROZEN RULE ===")
    print(f"rule={selected['rule']} expression={selected['expression']}")
    print(f"required_models={selected['required_models']}")
    print("=== EXPLORATORY TEST N=241 ===")
    print(
        f"TP={test['tp']} FN={test['fn']} FP={test['fp']} TN={test['tn']} "
        f"Recall={test['recall']:.2%} FPR={test['fpr']:.2%} "
        f"Accuracy={test['accuracy']:.2%} F1={test['f1']:.2%} "
        f"Go={test['go_pass']} Challenge={test['challenge_pass']}"
    )
    print("NOTE: Test result is exploratory because Test had already been observed.")
    print(f"summary={args.output_dir / 'summary.json'}")
    print("DEV_SELECTED_ENSEMBLE_STUDY: PASS")


if __name__ == "__main__":
    main()
