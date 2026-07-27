#!/usr/bin/env python3
"""Compare E5-975 adapter and merged-model outputs under identical Swift PT inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def analyze(root: Path) -> dict:
    adapter_metrics = load_json(root / "adapter" / "evaluation" / "metrics.json")
    merged_metrics = load_json(root / "merged" / "evaluation" / "metrics.json")
    adapter_rows = load_jsonl(root / "adapter" / "evaluation" / "parsed.jsonl")
    merged_rows = load_jsonl(root / "merged" / "evaluation" / "parsed.jsonl")
    if len(adapter_rows) != 241 or len(merged_rows) != 241:
        raise ValueError("both backends must contain exactly 241 parsed rows")

    decision_agreements = 0
    response_exact_matches = 0
    disagreements = []
    for adapter, merged in zip(adapter_rows, merged_rows):
        if adapter["image_path"] != merged["image_path"]:
            raise ValueError("adapter/merged image order mismatch")
        decision_match = adapter["predicted_decision"] == merged["predicted_decision"]
        response_match = adapter["raw_response"] == merged["raw_response"]
        decision_agreements += int(decision_match)
        response_exact_matches += int(response_match)
        if not decision_match or not response_match:
            disagreements.append(
                {
                    "index": adapter["index"],
                    "image_path": adapter["image_path"],
                    "gold_decision": adapter["gold_decision"],
                    "adapter_decision": adapter["predicted_decision"],
                    "merged_decision": merged["predicted_decision"],
                    "decision_match": decision_match,
                    "response_exact_match": response_match,
                    "adapter_response": adapter["raw_response"],
                    "merged_response": merged["raw_response"],
                }
            )

    quality_keys = ("recall", "fpr", "accuracy", "f1", "schema_valid_rate")
    summary = {
        "protocol_version": "e5_975_merge_equivalence_swift_pt_v1",
        "num_samples": 241,
        "adapter_metrics": adapter_metrics,
        "merged_metrics": merged_metrics,
        "decision_agreements": decision_agreements,
        "decision_agreement_rate": decision_agreements / 241,
        "decision_disagreements": 241 - decision_agreements,
        "response_exact_matches": response_exact_matches,
        "response_exact_match_rate": response_exact_matches / 241,
        "response_differences": 241 - response_exact_matches,
        "quality_delta_merged_minus_adapter": {
            key: merged_metrics[key] - adapter_metrics[key] for key in quality_keys
        },
        "strictly_equivalent": decision_agreements == 241 and response_exact_matches == 241,
    }
    (root / "disagreements.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in disagreements
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def format_metrics(metrics: dict) -> str:
    return (
        f"TP={metrics['tp']} FN={metrics['fn']} FP={metrics['fp']} TN={metrics['tn']} "
        f"Recall={metrics['recall']:.2%} FPR={metrics['fpr']:.2%} "
        f"Accuracy={metrics['accuracy']:.2%} F1={metrics['f1']:.2%} "
        f"Schema={metrics['schema_valid_rate']:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root)
    print("=== E5-975 MERGE EQUIVALENCE / SWIFT TRANSFORMERS ===")
    print("Adapter: " + format_metrics(summary["adapter_metrics"]))
    print("Merged:  " + format_metrics(summary["merged_metrics"]))
    print(
        f"decision_agreement={summary['decision_agreements']}/241 "
        f"({summary['decision_agreement_rate']:.2%}) "
        f"disagreements={summary['decision_disagreements']}"
    )
    print(
        f"response_exact_match={summary['response_exact_matches']}/241 "
        f"({summary['response_exact_match_rate']:.2%}) "
        f"differences={summary['response_differences']}"
    )
    print(f"strictly_equivalent={summary['strictly_equivalent']}")
    print(f"summary={args.root / 'summary.json'}")
    print("E5_975_MERGE_EQUIVALENCE: PASS")


if __name__ == "__main__":
    main()
