#!/usr/bin/env python3
"""Load Qwen3.6 weights and run natural JSON inference on 20 mixed E5/Dev rows."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

try:
    from scripts.check_qwen36_processor_preflight import (
        ProcessorPreflightError,
        assistant_payload,
        load_jsonl,
        select_poc_rows,
    )
except ModuleNotFoundError:  # Support direct execution via an absolute script path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.check_qwen36_processor_preflight import (
        ProcessorPreflightError,
        assistant_payload,
        load_jsonl,
        select_poc_rows,
    )


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


class InferencePocError(ValueError):
    """Raised when the fixed PoC contract or inference output is invalid."""


def _evenly_spaced(items: list[tuple[int, dict]], count: int) -> list[tuple[int, dict]]:
    if len(items) < count:
        raise InferencePocError(f"need {count} rows, found {len(items)}")
    return [
        items[round(index * (len(items) - 1) / (count - 1))]
        for index in range(count)
    ]


def select_balanced_dev(rows: list[dict], per_decision: int = 5) -> list[tuple[int, str, dict]]:
    groups: dict[str, list[tuple[int, dict]]] = {"GOOD": [], "BAD": []}
    for index, row in enumerate(rows):
        decision = assistant_payload(row)["decision"]
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise InferencePocError(f"Dev row {index} is not single-image")
        groups[decision].append((index, row))
    selected: list[tuple[int, str, dict]] = []
    for decision in ("GOOD", "BAD"):
        selected.extend(
            (index, f"DEV_{decision}", row)
            for index, row in _evenly_spaced(groups[decision], per_decision)
        )
    return selected


def validate_generated_payload(text: str) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(payload, dict):
        return None, "payload_not_object"
    if set(payload) != {"decision", "categories", "reasons"}:
        return None, f"unexpected_keys:{sorted(payload)}"
    if payload["decision"] not in {"GOOD", "BAD"}:
        return None, f"invalid_decision:{payload['decision']!r}"
    if not isinstance(payload["categories"], list):
        return None, "categories_not_list"
    if not isinstance(payload["reasons"], list):
        return None, "reasons_not_list"
    if not all(isinstance(item, str) for item in payload["categories"]):
        return None, "category_not_string"
    if not all(isinstance(item, str) for item in payload["reasons"]):
        return None, "reason_not_string"
    return payload, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise InferencePocError(f"output directory already exists: {args.output_dir}")
    dev_bytes = args.dev.read_bytes()
    dev_sha256 = hashlib.sha256(dev_bytes).hexdigest()
    if dev_sha256 != args.expected_dev_sha256:
        raise InferencePocError(
            f"Dev sha256 mismatch: {dev_sha256} != {args.expected_dev_sha256}"
        )
    dev_rows = load_jsonl(args.dev)
    if len(dev_rows) != 200:
        raise InferencePocError(f"expected 200 Dev rows, got {len(dev_rows)}")
    dev_counts = Counter(assistant_payload(row)["decision"] for row in dev_rows)
    if dev_counts != {"GOOD": 142, "BAD": 58}:
        raise InferencePocError(f"unexpected Dev labels: {dict(dev_counts)}")

    train_rows = load_jsonl(args.train)
    dev_selected = select_balanced_dev(dev_rows)
    try:
        dual_selected = select_poc_rows(
            train_rows, counts={"T2_BAD": 5, "T3_GOOD": 5}
        )
    except ProcessorPreflightError as exc:
        raise InferencePocError(str(exc)) from exc
    selected = dev_selected + [
        (index, f"E5_{stratum}", row) for index, stratum, row in dual_selected
    ]
    if len(selected) != 20:
        raise InferencePocError(f"expected 20 selected rows, got {len(selected)}")
    missing = [
        image
        for _, _, row in selected
        for image in row["images"]
        if not Path(image).is_file()
    ]
    if missing:
        raise InferencePocError(f"missing images: {len(missing)}; first={missing[0]}")

    try:
        import torch
        from swift import TransformersEngine
        from swift.infer_engine import InferRequest, RequestConfig
    except ImportError as exc:
        raise InferencePocError(f"required runtime import failed: {exc}") from exc

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    load_started = time.monotonic()
    engine = TransformersEngine(
        str(args.model),
        max_batch_size=1,
        torch_dtype=torch.bfloat16,
        attn_impl="flash_attention_2",
    )
    model_load_seconds = time.monotonic() - load_started
    engine.strict = True
    engine.template.enable_thinking = False
    engine.template.response_prefix = None

    model_parameter = next(engine.model.parameters())
    model_device = str(model_parameter.device)
    model_dtype = str(model_parameter.dtype)
    load_allocated_bytes = torch.cuda.memory_allocated()
    load_reserved_bytes = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    requests = [
        InferRequest(messages=deepcopy(row["messages"][:-1]), images=list(row["images"]))
        for _, _, row in selected
    ]
    request_config = RequestConfig(
        max_tokens=128,
        temperature=0.0,
        num_beams=1,
        seed=42,
        stream=False,
        return_details=True,
    )
    infer_started = time.monotonic()
    responses = engine.infer(requests, request_config=request_config, use_tqdm=True)
    inference_seconds = time.monotonic() - infer_started
    if len(responses) != len(selected):
        raise InferencePocError(
            f"response count mismatch: {len(responses)} != {len(selected)}"
        )

    result_rows: list[dict] = []
    confusion = Counter()
    valid_predictions = Counter()
    for request_index, ((source_index, stratum, source), response) in enumerate(
        zip(selected, responses)
    ):
        if response is None or len(response.choices) != 1:
            raise InferencePocError(f"invalid engine response at request {request_index}")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = ""
        prefix_valid = content.startswith(NON_THINKING_PREFIX)
        payload_text = content[len(NON_THINKING_PREFIX) :] if prefix_valid else content
        payload, schema_error = validate_generated_payload(payload_text)
        gold = assistant_payload(source)["decision"]
        prediction = None if payload is None else payload["decision"]
        if prediction is not None:
            valid_predictions[prediction] += 1
            if gold == "BAD" and prediction == "BAD":
                confusion["tp"] += 1
            elif gold == "BAD" and prediction == "GOOD":
                confusion["fn"] += 1
            elif gold == "GOOD" and prediction == "BAD":
                confusion["fp"] += 1
            else:
                confusion["tn"] += 1
        result = {
            "request_index": request_index,
            "source_index": source_index,
            "stratum": stratum,
            "images": source["images"],
            "gold_decision": gold,
            "response": content,
            "non_thinking_prefix_valid": prefix_valid,
            "payload_json": payload,
            "schema_error": schema_error,
            "prediction": prediction,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        result_rows.append(result)
        print(
            f"request={request_index} stratum={stratum} gold={gold} "
            f"prediction={prediction} prefix_valid={prefix_valid} "
            f"schema_error={schema_error}"
        )

    schema_valid = sum(row["schema_error"] is None for row in result_rows)
    prefix_valid = sum(row["non_thinking_prefix_valid"] for row in result_rows)
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    tp, fn, fp, tn = (confusion[name] for name in ("tp", "fn", "fp", "tn"))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    accuracy = (tp + tn) / sum(confusion.values()) if confusion else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    summary = {
        "protocol_version": "qwen36_mixed_inference_poc20_v1",
        "model": str(args.model),
        "dev": str(args.dev),
        "dev_sha256": dev_sha256,
        "train": str(args.train),
        "selection": {
            "DEV_GOOD": 5,
            "DEV_BAD": 5,
            "E5_T2_BAD": 5,
            "E5_T3_GOOD": 5,
        },
        "rows": len(result_rows),
        "image_inputs": sum(len(row["images"]) for row in result_rows),
        "model_device": model_device,
        "model_dtype": model_dtype,
        "attention_implementation": "flash_attention_2",
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "gpu_memory": {
            "load_allocated_bytes": load_allocated_bytes,
            "load_reserved_bytes": load_reserved_bytes,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
        },
        "non_thinking_prefix_valid": prefix_valid,
        "schema_valid": schema_valid,
        "json_schema_valid_rate": schema_valid / len(result_rows),
        "prediction_counts": dict(valid_predictions),
        "invalid_predictions": len(result_rows) - schema_valid,
        "confusion_on_valid_predictions": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": recall,
            "fpr": fpr,
            "accuracy": accuracy,
            "f1": f1,
        },
        "test_untouched": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in result_rows
        ),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (InferencePocError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== QWEN3.6 MIXED INFERENCE POC20 ===")
    print(f"model_device={summary['model_device']} dtype={summary['model_dtype']}")
    print(f"model_load_seconds={summary['model_load_seconds']:.1f}")
    print(f"inference_seconds={summary['inference_seconds']:.1f}")
    print(
        f"non_thinking_prefix_valid={summary['non_thinking_prefix_valid']}/20 "
        f"schema_valid={summary['schema_valid']}/20"
    )
    print(f"prediction_counts={summary['prediction_counts']}")
    print(f"confusion={summary['confusion_on_valid_predictions']}")
    memory = summary["gpu_memory"]
    print(
        f"gpu_load_reserved_gib={memory['load_reserved_bytes'] / 2**30:.2f} "
        f"gpu_peak_reserved_gib={memory['peak_reserved_bytes'] / 2**30:.2f}"
    )
    print(f"output={args.output_dir}")
    if summary["schema_valid"] != 20 or summary["non_thinking_prefix_valid"] != 20:
        print("QWEN36_INFERENCE_POC20: FAIL")
        return 2
    print("QWEN36_INFERENCE_POC20: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
