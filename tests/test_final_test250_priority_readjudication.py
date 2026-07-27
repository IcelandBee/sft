import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_final_test250_priority_readjudication import (
    ReadjudicationError,
    analyze_readjudication,
    run_analysis,
)


MODELS = ["e1-1248", "e2-1248", "e5-780-recall", "e5-975-balanced"]


def make_test_row(index: int, decision: str) -> dict:
    payload = (
        {"decision": "GOOD", "categories": [], "reasons": []}
        if decision == "GOOD"
        else {"decision": "BAD", "categories": ["其他"], "reasons": ["原标签"]}
    )
    return {
        "images": [f"/images/{index}.png"],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
    }


def parsed_row(index: int, decision: str) -> dict:
    return {
        "index": index,
        "image_path": f"/images/{index}.png",
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": {"decision": decision},
    }


class FinalTest250PriorityReadjudicationTests(unittest.TestCase):
    def setUp(self):
        self.golds = ["GOOD", "BAD", "GOOD", "BAD", "GOOD", "BAD"]
        self.test_rows = [
            make_test_row(index, decision)
            for index, decision in enumerate(self.golds)
        ]
        self.review_rows = [
            {
                "review_order": order,
                "row": row,
                "index": row - 1,
                "image_path": f"/images/{row - 1}.png",
            }
            for order, row in enumerate([3, 1, 4, 2], start=1)
        ]
        self.consensus_rows = [
            {
                "row": row,
                "index": row - 1,
                "image_path": f"/images/{row - 1}.png",
                "stratum": "unanimous_against_gold" if row < 3 else "split_two_two",
                "wrong_count": 4 if row < 3 else 2,
            }
            for row in range(1, 5)
        ]
        self.annotations = {
            1: {
                "review_decision": "BAD",
                "visible_severity": "obvious",
                "categories": ["手部异常"],
                "notes": "手指异常",
                "completed": True,
            },
            2: {
                "review_decision": "GOOD",
                "visible_severity": "none",
                "categories": [],
                "notes": "",
                "completed": True,
            },
            3: {
                "review_decision": "UNSURE",
                "visible_severity": "uncertain",
                "categories": [],
                "notes": "无法确定",
                "completed": True,
            },
            4: {
                "review_decision": "BAD",
                "visible_severity": "borderline",
                "categories": ["其他"],
                "notes": "边界异常",
                "completed": True,
            },
        }
        decisions = ["BAD", "GOOD", "BAD", "BAD", "GOOD", "GOOD"]
        self.predictions = {
            model: [parsed_row(index, value) for index, value in enumerate(decisions)]
            for model in MODELS
        }

    def test_excludes_unsure_and_rescores_existing_predictions(self):
        summary, records, adjusted_rows = analyze_readjudication(
            self.test_rows,
            self.review_rows,
            self.consensus_rows,
            self.annotations,
            self.predictions,
            expected_test_count=6,
            expected_review_count=4,
        )
        self.assertEqual(summary["label_changes"], 2)
        self.assertEqual(summary["excluded_unsure"], 1)
        self.assertEqual(summary["included_total"], 5)
        self.assertEqual(summary["adjusted_gold_counts"], {"BAD": 3, "GOOD": 2})
        self.assertEqual(len(records["excluded_unsure"]), 1)
        self.assertEqual(len(adjusted_rows), 5)
        metrics = summary["model_results"]["e1-1248"][
            "adjusted_excluding_unsure"
        ]
        self.assertEqual(
            {key: metrics[key] for key in ("tp", "fn", "fp", "tn")},
            {"tp": 2, "fn": 1, "fp": 0, "tn": 2},
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertEqual(
            summary["model_results"]["e1-1248"]["paired_correctness_change"],
            {"became_correct": 2, "both_correct": 2, "both_wrong": 1},
        )

    def test_requires_all_review_annotations(self):
        del self.annotations[4]
        with self.assertRaisesRegex(ReadjudicationError, "do not cover"):
            analyze_readjudication(
                self.test_rows,
                self.review_rows,
                self.consensus_rows,
                self.annotations,
                self.predictions,
                expected_test_count=6,
                expected_review_count=4,
            )

    def test_run_writes_adjusted_test_and_audit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, rows in {
                "test": self.test_rows,
                "review": self.review_rows,
                "consensus": self.consensus_rows,
            }.items():
                path = root / f"{name}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[name] = path
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(self.annotations), encoding="utf-8")
            parsed_paths = {}
            for model, rows in self.predictions.items():
                path = root / f"{model}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                parsed_paths[model] = path
            output = root / "output"
            summary = run_analysis(
                paths["test"],
                paths["review"],
                paths["consensus"],
                annotations,
                parsed_paths,
                output,
                expected_test_count=6,
                expected_review_count=4,
            )
            self.assertEqual(summary["included_total"], 5)
            self.assertEqual(
                len(
                    (output / "test_conditionally_readjudicated.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                5,
            )
            self.assertTrue((output / "label_changes.jsonl").is_file())
            self.assertTrue((output / "summary.json").is_file())

    def test_wrapper_is_gpu_free_and_locks_original_confusions(self):
        wrapper = Path(
            "scripts/run_final_test250_priority_readjudication.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("e1-1248=33,31,27,159", wrapper)
        self.assertIn("e5-780-recall=38,26,52,134", wrapper)
        self.assertNotIn("swift infer", wrapper)
        self.assertNotIn("nvidia-smi", wrapper)


if __name__ == "__main__":
    unittest.main()
