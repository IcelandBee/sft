#!/usr/bin/env python3
"""Summarize speed, quality, and row-level parity for the two backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_metrics(rows: list[dict]) -> dict:
    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for row in rows:
        gold = row["gold_decision"]
        predicted = row["predicted_decision"]
        if gold == "BAD" and predicted == "BAD":
            confusion["tp"] += 1
        elif gold == "BAD":
            confusion["fn"] += 1
        elif predicted == "BAD":
            confusion["fp"] += 1
        else:
            confusion["tn"] += 1
    tp, fn, fp, tn = (confusion[key] for key in ("tp", "fn", "fp", "tn"))
    ratio = lambda a, b: a / b if b else 0.0
    return {
        **confusion,
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(rows)),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-parsed", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    hf_stats = load_json(root / "transformers" / "stats.json")
    vl_stats = load_json(root / "vllm" / "stats.json")
    hf_metrics = load_json(root / "transformers" / "evaluation" / "metrics.json")
    vl_metrics = load_json(root / "vllm" / "evaluation" / "metrics.json")
    hf_rows = load_jsonl(root / "transformers" / "evaluation" / "parsed.jsonl")
    vl_rows = load_jsonl(root / "vllm" / "evaluation" / "parsed.jsonl")
    if len(hf_rows) != len(vl_rows) or len(hf_rows) != 241:
        raise ValueError("backend row counts do not match the N=241 contract")

    reference_all = load_jsonl(args.reference_parsed)
    reference_by_image = {row["image_path"]: row for row in reference_all}
    reference_rows = []
    reference_disagreements = []
    reference_agreements = 0
    for hf in hf_rows:
        reference = reference_by_image.get(hf["image_path"])
        if reference is None:
            raise ValueError(f"reference prediction missing image: {hf['image_path']}")
        reference_row = {
            "image_path": hf["image_path"],
            "gold_decision": hf["gold_decision"],
            "predicted_decision": reference["predicted_decision"],
        }
        reference_rows.append(reference_row)
        if reference_row["predicted_decision"] == hf["predicted_decision"]:
            reference_agreements += 1
        else:
            reference_disagreements.append(
                {
                    "index": hf["index"],
                    "image_path": hf["image_path"],
                    "gold_decision": hf["gold_decision"],
                    "original_adapter": reference_row["predicted_decision"],
                    "merged_transformers": hf["predicted_decision"],
                }
            )
    reference_metrics = binary_metrics(reference_rows)

    disagreements = []
    agreements = 0
    for hf, vl in zip(hf_rows, vl_rows):
        if hf["image_path"] != vl["image_path"]:
            raise ValueError("backend image order mismatch")
        if hf["predicted_decision"] == vl["predicted_decision"]:
            agreements += 1
        else:
            disagreements.append(
                {
                    "index": hf["index"],
                    "image_path": hf["image_path"],
                    "gold_decision": hf["gold_decision"],
                    "transformers": hf["predicted_decision"],
                    "vllm": vl["predicted_decision"],
                }
            )

    summary = {
        "protocol_version": "e5_975_transformers_vs_vllm_v1",
        "num_samples": 241,
        "original_adapter_reference": {
            "parsed": str(args.reference_parsed),
            "metrics_on_n241": reference_metrics,
            "merged_transformers_decision_agreement": reference_agreements,
            "merged_transformers_decision_agreement_rate": reference_agreements / 241,
            "merged_transformers_decision_disagreements": len(reference_disagreements),
        },
        "transformers": {"timing": hf_stats, "metrics": hf_metrics},
        "vllm": {"timing": vl_stats, "metrics": vl_metrics},
        "speedup": {
            "inference": hf_stats["inference_seconds"] / vl_stats["inference_seconds"],
            "end_to_end": hf_stats["total_seconds"] / vl_stats["total_seconds"],
        },
        "decision_agreement": agreements,
        "decision_agreement_rate": agreements / 241,
        "decision_disagreements": len(disagreements),
        "quality_delta_vllm_minus_transformers": {
            key: vl_metrics[key] - hf_metrics[key]
            for key in ("recall", "fpr", "accuracy", "f1", "schema_valid_rate")
        },
        "comparison_scope": "same merged E5-975 model, same frozen prompts, same resized images, same GPU sequentially",
    }
    (root / "disagreements.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in disagreements),
        encoding="utf-8",
        newline="\n",
    )
    (root / "merge-reference-disagreements.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in reference_disagreements
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / "benchmark-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=== E5-975 TRANSFORMERS VS VLLM ===")
    print(
        f"Original adapter reference: R={reference_metrics['recall']:.2%} "
        f"FPR={reference_metrics['fpr']:.2%} Acc={reference_metrics['accuracy']:.2%} "
        f"F1={reference_metrics['f1']:.2%}"
    )
    for name, stats, metrics in (
        ("Transformers", hf_stats, hf_metrics),
        ("vLLM", vl_stats, vl_metrics),
    ):
        print(
            f"{name}: inference={stats['inference_seconds']:.2f}s "
            f"generation={stats['generation_seconds']:.2f}s "
            f"per_image={stats['seconds_per_image']:.3f}s "
            f"throughput={stats['samples_per_second']:.3f} img/s "
            f"total={stats['total_seconds']:.2f}s "
            f"R={metrics['recall']:.2%} FPR={metrics['fpr']:.2%} "
            f"Acc={metrics['accuracy']:.2%} F1={metrics['f1']:.2%} "
            f"Schema={metrics['schema_valid_rate']:.2%}"
        )
    print(
        f"speedup_inference={summary['speedup']['inference']:.2f}x "
        f"speedup_end_to_end={summary['speedup']['end_to_end']:.2f}x"
    )
    print(
        f"decision_agreement={agreements}/241 "
        f"({summary['decision_agreement_rate']:.2%}) disagreements={len(disagreements)}"
    )
    print(
        f"merge_reference_agreement={reference_agreements}/241 "
        f"({reference_agreements / 241:.2%}) disagreements={len(reference_disagreements)}"
    )
    print(f"summary={root / 'benchmark-summary.json'}")
    print("VLLM_LORA_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
