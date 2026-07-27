import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_e5_975_merge_equivalence import analyze


def metrics(tp, fn, fp, tn):
    total = tp + fn + fp + tn
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "accuracy": (tp + tn) / total,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "schema_valid_rate": 1.0,
    }


class E5975MergeEquivalenceTests(unittest.TestCase):
    def test_runner_freezes_identical_transformers_protocol(self):
        script = Path("scripts/run_e5_975_merge_equivalence.sh").read_text(encoding="utf-8")
        self.assertIn("checkpoint-975", script)
        self.assertIn("merged-e5-975", script)
        self.assertIn("run_infer()", script)
        self.assertEqual(script.count("run_infer "), 2)
        self.assertIn("run_infer adapter --model \"$MODEL\" --adapters \"$ADAPTER\"", script)
        self.assertIn("run_infer merged --model \"$MERGED\"", script)
        for value in (
            "--infer_backend transformers",
            "--add_non_thinking_prefix true",
            "--max_new_tokens 128",
            "--temperature 0",
            "--load_args false",
            "IMAGE_MAX_TOKEN_NUM=1024",
        ):
            self.assertIn(value, script)

    def test_analyzer_reports_decision_and_response_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("adapter", "merged"):
                evaluation = root / name / "evaluation"
                evaluation.mkdir(parents=True)
                (evaluation / "metrics.json").write_text(
                    json.dumps(metrics(44, 22, 17, 158)), encoding="utf-8"
                )
                rows = [
                    {
                        "index": index,
                        "image_path": f"/{index}.jpg",
                        "gold_decision": "GOOD",
                        "predicted_decision": "GOOD",
                        "raw_response": "same",
                    }
                    for index in range(241)
                ]
                (evaluation / "parsed.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            summary = analyze(root)
            self.assertTrue(summary["strictly_equivalent"])
            self.assertEqual(summary["decision_agreement_rate"], 1.0)
            self.assertEqual(summary["response_exact_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
