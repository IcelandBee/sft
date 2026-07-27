import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_dev_selected_ensemble import (
    MODEL_NAMES,
    EnsembleStudyError,
    run_study,
    study_ensemble,
)


def dataset(prefix: str, golds: list[str]) -> list[dict]:
    return [
        {"row": index + 1, "image_path": f"/{prefix}/{index}.png", "gold": gold}
        for index, gold in enumerate(golds)
    ]


def parsed(prefix: str, decisions: list[str]) -> list[dict]:
    return [
        {
            "index": index,
            "image_path": f"/{prefix}/{index}.png",
            "schema_valid": True,
            "predicted_decision": decision,
        }
        for index, decision in enumerate(decisions)
    ]


def source_dataset(prefix: str, golds: list[str]) -> list[dict]:
    rows = []
    for index, gold in enumerate(golds):
        payload = (
            {"decision": "GOOD", "categories": [], "reasons": []}
            if gold == "GOOD"
            else {"decision": "BAD", "categories": ["其他"], "reasons": ["异常"]}
        )
        rows.append(
            {
                "images": [f"/{prefix}/{index}.png"],
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": json.dumps(payload)},
                ],
            }
        )
    return rows


class DevSelectedEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.dev_golds = ["BAD"] * 5 + ["GOOD"] * 5
        self.dev = dataset("dev", self.dev_golds)
        decisions = {
            "e1-1248": ["BAD"] * 4 + ["GOOD"] + ["GOOD"] * 5,
            "e2-1248": ["BAD"] * 3 + ["GOOD"] * 2 + ["GOOD"] * 5,
            "e5-780-recall": ["BAD"] * 5 + ["BAD", "GOOD", "GOOD", "GOOD", "GOOD"],
            "e5-975-balanced": ["BAD"] * 4 + ["GOOD"] + ["GOOD"] * 5,
        }
        self.dev_predictions = {
            model: parsed("dev", decisions[model]) for model in MODEL_NAMES
        }
        self.test_golds = ["BAD", "GOOD", "BAD", "GOOD"]
        self.test = dataset("test", self.test_golds)
        self.test_predictions = {
            model: parsed("test", decisions[model][:4]) for model in MODEL_NAMES
        }

    def test_selects_only_on_dev_and_reports_fixed_rule_set(self):
        summary, records = study_ensemble(
            self.dev,
            self.dev_predictions,
            self.test,
            self.test_predictions,
        )
        rule_names = {row["rule"] for row in summary["dev_rule_results"]}
        self.assertEqual(
            rule_names,
            {
                "three_majority",
                "four_at_least_two",
                "four_at_least_three",
                "e5_recall_confirmed",
                "balanced_or_double_confirmed",
                "weighted_balanced",
                "weighted_recall",
            },
        )
        self.assertTrue(summary["selected_on_dev_only"])
        self.assertTrue(summary["test_previously_observed"])
        selected = summary["selected_rule"]["rule"]

        inverted_test = {
            model: parsed(
                "test",
                ["GOOD" if row["predicted_decision"] == "BAD" else "BAD" for row in rows],
            )
            for model, rows in self.test_predictions.items()
        }
        changed, _ = study_ensemble(
            self.dev,
            self.dev_predictions,
            self.test,
            inverted_test,
        )
        self.assertEqual(changed["selected_rule"]["rule"], selected)
        self.assertEqual(len(records), 4)

    def test_rejects_missing_model(self):
        del self.dev_predictions["e2-1248"]
        with self.assertRaisesRegex(EnsembleStudyError, "model set mismatch"):
            study_ensemble(
                self.dev,
                self.dev_predictions,
                self.test,
                self.test_predictions,
            )

    def test_run_writes_reproducible_audit_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev_path = root / "dev.jsonl"
            test_path = root / "test.jsonl"
            dev_path.write_text(
                "".join(json.dumps(row) + "\n" for row in source_dataset("dev", self.dev_golds)),
                encoding="utf-8",
            )
            test_path.write_text(
                "".join(json.dumps(row) + "\n" for row in source_dataset("test", self.test_golds)),
                encoding="utf-8",
            )
            dev_paths = {}
            test_paths = {}
            for model in MODEL_NAMES:
                dev_file = root / f"dev-{model}.jsonl"
                test_file = root / f"test-{model}.jsonl"
                dev_file.write_text(
                    "".join(json.dumps(row) + "\n" for row in self.dev_predictions[model]),
                    encoding="utf-8",
                )
                test_file.write_text(
                    "".join(json.dumps(row) + "\n" for row in self.test_predictions[model]),
                    encoding="utf-8",
                )
                dev_paths[model] = dev_file
                test_paths[model] = test_file
            output = root / "output"
            summary = run_study(
                dev_path,
                dev_paths,
                test_path,
                test_paths,
                output,
                expected_dev_sha256=hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                expected_test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                expected_dev_count=10,
                expected_test_count=4,
            )
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "test_predictions.jsonl").is_file())
            self.assertIn("selected_rule", summary)

    def test_wrapper_is_gpu_free_and_uses_corrected_dev_only_for_selection(self):
        wrapper = Path("scripts/run_dev_selected_ensemble_study.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("dev_adjudicated_v1/dev.jsonl", wrapper)
        self.assertIn("checkpoint-780/evaluation/parsed.jsonl", wrapper)
        self.assertIn("checkpoint-975/evaluation/parsed.jsonl", wrapper)
        self.assertNotIn("swift infer", wrapper)
        self.assertNotIn("nvidia-smi", wrapper)


if __name__ == "__main__":
    unittest.main()
