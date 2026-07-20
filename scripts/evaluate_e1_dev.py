#!/usr/bin/env python3
"""Strictly evaluate one ms-swift E1 Dev inference result JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
VALID_DECISIONS = {"GOOD", "BAD"}
PAYLOAD_FIELDS = {"decision", "categories", "reasons"}
SYSTEM_PROMPT = (
    "你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。"
    "严格只输出指定JSON，不要添加分析、解释或Markdown。"
)
USER_PROMPT = (
    "<image>\n检查这张图片。输出decision、categories和reasons。"
    "decision只能是GOOD或BAD。"
)


class EvaluationError(ValueError):
    """Raised when an inference result violates the evaluation contract."""


def validate_payload(value: object) -> str | None:
    """Return a stable error code, or ``None`` for a valid E1 payload."""
    if not isinstance(value, dict):
        return "schema_not_object"
    if set(value) != PAYLOAD_FIELDS:
        return "schema_fields"
    if value.get("decision") not in VALID_DECISIONS:
        return "schema_decision"
    categories = value.get("categories")
    reasons = value.get("reasons")
    if not isinstance(categories, list):
        return "schema_categories_type"
    if not isinstance(reasons, list):
        return "schema_reasons_type"
    if any(not isinstance(item, str) or not item.strip() for item in categories):
        return "schema_categories_item"
    if any(not isinstance(item, str) or not item.strip() for item in reasons):
        return "schema_reasons_item"
    if value["decision"] == "GOOD":
        if categories or reasons:
            return "schema_good_aux"
    elif not (1 <= len(categories) <= 3 and 1 <= len(reasons) <= 3):
        return "schema_bad_aux"
    return None


def parse_prediction(raw: object) -> dict:
    """Strictly parse one response without extraction or repair."""
    result = {
        "envelope_valid": False,
        "raw_direct_json_valid": False,
        "payload_json_valid": False,
        "schema_valid": False,
        "payload_text": None,
        "payload": None,
        "decision": None,
        "error_code": None,
    }
    if not isinstance(raw, str):
        result["error_code"] = "response_not_string"
        return result

    text = raw.strip()
    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        result["raw_direct_json_valid"] = True

    if not text.startswith(NON_THINKING_PREFIX):
        result["error_code"] = (
            "envelope_noncanonical" if "<think>" in text or "</think>" in text
            else "envelope_missing"
        )
        return result
    if (
        text.count(NON_THINKING_PREFIX) != 1
        or text.count("<think>") != 1
        or text.count("</think>") != 1
    ):
        result["error_code"] = "envelope_duplicate"
        return result

    result["envelope_valid"] = True
    payload_text = text[len(NON_THINKING_PREFIX):].strip()
    result["payload_text"] = payload_text
    try:
        value = json.loads(payload_text)
    except json.JSONDecodeError:
        result["error_code"] = "payload_json_invalid"
        return result

    result["payload_json_valid"] = True
    result["payload"] = value
    schema_error = validate_payload(value)
    if schema_error is not None:
        result["error_code"] = schema_error
        return result

    result["schema_valid"] = True
    result["decision"] = value["decision"]
    return result


def _load_gold(value: object, location: str) -> dict:
    if not isinstance(value, str):
        raise EvaluationError(f"gold labels at {location} must be a JSON string")
    try:
        gold = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"gold labels at {location} are invalid JSON") from exc
    schema_error = validate_payload(gold)
    if schema_error is not None:
        raise EvaluationError(
            f"gold labels at {location} violate schema: {schema_error}"
        )
    return gold


def _image_path(row: dict, location: str) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise EvaluationError(f"images at {location} must contain exactly one item")
    image = images[0]
    if not isinstance(image, dict):
        raise EvaluationError(f"image at {location} must be an object")
    path = image.get("path")
    if not isinstance(path, str) or not path:
        raise EvaluationError(f"image path at {location} must be a non-empty string")
    return path


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_rows(
    rows: list[dict], expected_count: int
) -> tuple[list[dict], dict]:
    """Evaluate rows in order, conservatively treating INVALID as incorrect."""
    if len(rows) != expected_count:
        raise EvaluationError(f"expected {expected_count} result rows, got {len(rows)}")

    parsed_rows: list[dict] = []
    seen_images: dict[str, int] = {}
    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    invalid_by_gold = {"GOOD": 0, "BAD": 0}

    for index, row in enumerate(rows):
        location = f"row {index + 1}"
        if not isinstance(row, dict):
            raise EvaluationError(f"{location} must be an object")
        image_path = _image_path(row, location)
        if image_path in seen_images:
            raise EvaluationError(
                f"duplicate image path at rows {seen_images[image_path]} and {index + 1}: "
                f"{image_path}"
            )
        seen_images[image_path] = index + 1

        raw_response = row.get("response")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or messages[-1] != {"role": "assistant", "content": raw_response}
        ):
            raise EvaluationError(
                f"last assistant message at {location} must equal response"
            )

        gold_payload = _load_gold(row.get("labels"), location)
        gold = gold_payload["decision"]
        prediction = parse_prediction(raw_response)
        predicted = prediction["decision"]

        if not prediction["schema_valid"]:
            invalid_by_gold[gold] += 1
            if gold == "BAD":
                confusion["fn"] += 1
            else:
                confusion["fp"] += 1
            is_error = True
        elif gold == "BAD" and predicted == "BAD":
            confusion["tp"] += 1
            is_error = False
        elif gold == "BAD":
            confusion["fn"] += 1
            is_error = True
        elif predicted == "BAD":
            confusion["fp"] += 1
            is_error = True
        else:
            confusion["tn"] += 1
            is_error = False

        parsed_rows.append(
            {
                "index": index,
                "image_path": image_path,
                "gold_decision": gold,
                "predicted_decision": predicted,
                "is_error": is_error,
                "raw_response": raw_response,
                **prediction,
            }
        )

    tp, fn, fp, tn = (
        confusion["tp"], confusion["fn"], confusion["fp"], confusion["tn"]
    )
    metrics = {
        "protocol_version": "e1_dev_generation_v1",
        "total": expected_count,
        **confusion,
        "recall": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, expected_count),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "envelope_valid_rate": _ratio(
            sum(row["envelope_valid"] for row in parsed_rows), expected_count
        ),
        "payload_json_valid_rate": _ratio(
            sum(row["payload_json_valid"] for row in parsed_rows), expected_count
        ),
        "raw_direct_json_valid_rate": _ratio(
            sum(row["raw_direct_json_valid"] for row in parsed_rows), expected_count
        ),
        "schema_valid_rate": _ratio(
            sum(row["schema_valid"] for row in parsed_rows), expected_count
        ),
        "invalid_by_gold": invalid_by_gold,
    }
    if tp + fn + fp + tn != expected_count:
        raise AssertionError("confusion matrix does not cover every result row")
    return parsed_rows, metrics


def _load_result_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"result file does not exist: {path}")
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"invalid result JSON at {path}:{line_number}: {exc.msg}"
            ) from exc
        rows.append(row)
    return rows


def _load_expected_dev(path: Path, expected_count: int) -> tuple[list[dict], str]:
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"expected Dev file does not exist: {path}")
    source = path.read_bytes()
    rows: list[dict] = []
    for line_number, line in enumerate(
        source.decode("utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"invalid expected Dev JSON at {path}:{line_number}"
            ) from exc
        rows.append(row)
    if len(rows) != expected_count:
        raise EvaluationError(
            f"expected Dev must contain {expected_count} rows, got {len(rows)}"
        )
    return rows, hashlib.sha256(source).hexdigest()


def _validate_against_dev(result_rows: list[dict], dev_rows: list[dict]) -> None:
    for index, (result, dev) in enumerate(zip(result_rows, dev_rows), start=1):
        dev_images = dev.get("images") if isinstance(dev, dict) else None
        if (
            not isinstance(dev_images, list)
            or len(dev_images) != 1
            or not isinstance(dev_images[0], str)
        ):
            raise EvaluationError(f"invalid expected Dev image at row {index}")
        result_image = _image_path(result, f"result row {index}")
        if result_image != dev_images[0]:
            raise EvaluationError(
                f"Dev order/image mismatch at row {index}: "
                f"expected {dev_images[0]}, got {result_image}"
            )

        dev_messages = dev.get("messages")
        result_messages = result.get("messages") if isinstance(result, dict) else None
        if (
            not isinstance(dev_messages, list)
            or len(dev_messages) != 3
            or not isinstance(result_messages, list)
            or len(result_messages) != 3
        ):
            raise EvaluationError(f"invalid Dev/result messages at row {index}")
        if result_messages[:2] != dev_messages[:2]:
            raise EvaluationError(f"Dev prompt mismatch at row {index}")
        if result.get("labels") != dev_messages[2].get("content"):
            raise EvaluationError(f"Dev gold mismatch at row {index}")


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def run_evaluation(
    result_path: Path,
    output_dir: Path,
    expected_count: int = 200,
    checkpoint_step: int | None = None,
    expected_dev: Path | None = None,
) -> dict:
    """Evaluate a result file and atomically publish three deterministic artifacts."""
    result_path = Path(result_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise EvaluationError(f"output directory already exists: {output_dir}")
    rows = _load_result_rows(result_path)
    dev_sha256 = None
    if expected_dev is not None:
        dev_rows, dev_sha256 = _load_expected_dev(expected_dev, expected_count)
        _validate_against_dev(rows, dev_rows)
    parsed_rows, metrics = evaluate_rows(rows, expected_count)
    metrics["checkpoint_step"] = checkpoint_step
    metrics["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    metrics["dev_sha256"] = dev_sha256
    errors = [row for row in parsed_rows if row["is_error"]]

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        _write_text(staging / "parsed.jsonl", _jsonl_text(parsed_rows))
        _write_text(staging / "errors.jsonl", _jsonl_text(errors))
        _write_text(
            staging / "metrics.json",
            json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--expected-dev", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run_evaluation(
            args.result,
            args.output_dir,
            expected_count=args.expected_count,
            checkpoint_step=args.checkpoint_step,
            expected_dev=args.expected_dev,
        )
    except (EvaluationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
