import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_final_test250_consensus import (
    ConsensusAnalysisError,
    analyze_consensus,
    run_analysis,
)


MODELS = ["e1-1248", "e2-1248", "e5-780-recall", "e5-975-balanced"]


def make_test_row(index: int, gold: str) -> dict:
    payload = (
        {"decision": "GOOD", "categories": [], "reasons": []}
        if gold == "GOOD"
        else {
            "decision": "BAD",
            "categories": ["其他"],
            "reasons": ["人工标注为有异常"],
        }
    )
    return {
        "images": [f"/images/{index}.png"],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
    }


def prediction_row(index: int, decision: str) -> dict:
    payload = (
        {"decision": "GOOD", "categories": [], "reasons": []}
        if decision == "GOOD"
        else {
            "decision": "BAD",
            "categories": ["手部异常"],
            "reasons": ["手指异常"],
        }
    )
    return {
        "index": index,
        "image_path": f"/images/{index}.png",
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": payload,
    }


class FinalTest250ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.golds = ["BAD", "GOOD", "BAD", "GOOD", "BAD", "GOOD"]
        by_row = [
            ["GOOD", "GOOD", "GOOD", "GOOD"],
            ["BAD", "BAD", "BAD", "BAD"],
            ["GOOD", "GOOD", "GOOD", "BAD"],
            ["BAD", "BAD", "GOOD", "GOOD"],
            ["GOOD", "BAD", "BAD", "BAD"],
            ["GOOD", "GOOD", "GOOD", "GOOD"],
        ]
        self.test_rows = [
            make_test_row(index, gold) for index, gold in enumerate(self.golds)
        ]
        self.predictions = {
            model: [
                prediction_row(index, by_row[index][model_index])
                for index in range(len(self.golds))
            ]
            for model_index, model in enumerate(MODELS)
        }

    def test_stratifies_consensus_and_unique_errors(self):
        summary, records = analyze_consensus(
            self.test_rows,
            self.predictions,
            expected_count=6,
        )
        self.assertEqual(
            summary["strata"],
            {
                "one_of_four_against_gold": 1,
                "split_two_two": 1,
                "three_of_four_against_gold": 1,
                "unanimous_against_gold": 2,
                "unanimous_with_gold": 1,
            },
        )
        self.assertEqual(summary["review_priority_rows"], 4)
        self.assertEqual(summary["unanimous_fn"], 1)
        self.assertEqual(summary["unanimous_fp"], 1)
        self.assertEqual(len(records["e1-1248_unique_errors"]), 1)
        self.assertEqual(len(records["shared_errors_two_plus"]), 4)
        self.assertEqual(
            summary["model_results"]["e1-1248"]["unique_error_breakdown"],
            {"fn": 1, "fp": 0},
        )
        self.assertEqual(
            summary["shared_errors_two_plus"],
            {"total": 4, "fn": 2, "fp": 2},
        )
        self.assertEqual(
            summary["model_results"]["e1-1248"]["confusion"],
            {"tp": 0, "fn": 3, "fp": 2, "tn": 1},
        )

    def test_rejects_image_misalignment(self):
        self.predictions["e2-1248"][0]["image_path"] = "/wrong.png"
        with self.assertRaisesRegex(
            ConsensusAnalysisError, "image alignment mismatch"
        ):
            analyze_consensus(
                self.test_rows,
                self.predictions,
                expected_count=6,
            )

    def test_writes_atomic_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.test_rows),
                encoding="utf-8",
            )
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
                test_path,
                parsed_paths,
                output,
                expected_count=6,
            )
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(
                len((output / "review_priority.jsonl").read_text().splitlines()),
                4,
            )
            self.assertFalse(summary["labels_modified"])
            with self.assertRaisesRegex(
                ConsensusAnalysisError, "output directory already exists"
            ):
                run_analysis(
                    test_path,
                    parsed_paths,
                    output,
                    expected_count=6,
                )

    def test_wrapper_is_read_only_and_gpu_free(self):
        wrapper = Path(
            "scripts/run_final_test250_consensus_analysis.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "860e62ee2326b3f96b524e3e982e912d9f41f35042735b39744fd7a08f85649f",
            wrapper,
        )
        self.assertIn("four_model_consensus_v1", wrapper)
        self.assertNotIn("swift infer", wrapper)
        self.assertNotIn("nvidia-smi", wrapper)


if __name__ == "__main__":
    unittest.main()
