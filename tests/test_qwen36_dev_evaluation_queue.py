import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_qwen36_dev_results import SummaryError, build_comparison


WRAPPER = Path("scripts/run_qwen36_dev_checkpoints.sh").read_text(encoding="utf-8")
QUEUE = Path("scripts/run_qwen36_dev_evaluation_queue.sh").read_text(encoding="utf-8")


class Qwen36DevCheckpointWrapperTests(unittest.TestCase):
    def test_locks_isolated_model_fixed_dev_and_all_run_directories(self):
        self.assertIn("/envs/qwen36_27b", WRAPPER)
        self.assertIn("Qwen3.6-27B", WRAPPER)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", WRAPPER)
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            WRAPPER,
        )
        for run_dir in (
            "v0-20260729-015806",
            "v0-20260728-233940",
            "v0-20260729-062123",
            "v0-20260729-084616",
            "v0-20260728-204136",
        ):
            self.assertIn(run_dir, WRAPPER)
        self.assertNotIn("test.jsonl", WRAPPER.lower())

    def test_locks_all_eight_steps_and_four_gpu_protocol(self):
        for steps in (
            "312 624 936 1248 1560 1872 2184 2496",
            "156 312 468 624 780 936 1092 1248",
            "260 520 780 1040 1300 1560 1820 2080",
            "195 390 585 780 975 1170 1365 1560",
        ):
            self.assertIn(f"STEPS=({steps})", WRAPPER)
        self.assertIn("--gpus 4 5 6 7", WRAPPER)
        self.assertIn("--expected-good 142", WRAPPER)
        self.assertIn("--expected-bad 58", WRAPPER)
        self.assertIn("run_e1_dev_checkpoints.py", WRAPPER)


class Qwen36DevQueueTests(unittest.TestCase):
    def test_runs_experiments_sequentially_then_summarizes(self):
        self.assertIn("DEFAULT_ORDER=(E5 E2 E1 E3 E4)", QUEUE)
        self.assertIn("status.tsv", QUEUE)
        self.assertIn("summarize_qwen36_dev_results.py", QUEUE)
        self.assertIn("dev_comparison_v1.json", QUEUE)
        self.assertNotIn(" &\n", QUEUE)


def metric(step, recall, accuracy, f1, fpr=0.1, schema=1.0):
    return {
        "checkpoint_step": step,
        "total": 200,
        "tp": round(recall * 50),
        "fn": 50 - round(recall * 50),
        "fp": round(fpr * 150),
        "tn": 150 - round(fpr * 150),
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
        "schema_valid_rate": schema,
        "eligible": schema >= 0.995 and fpr <= 0.25,
    }


class Qwen36DevSummaryTests(unittest.TestCase):
    def test_builds_base_deltas_and_overall_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base_dev_natural_v1" / "evaluation"
            base_dir.mkdir(parents=True)
            base = metric(0, 0.4, 0.7, 0.45, fpr=0.2)
            base_dir.joinpath("metrics.json").write_text(json.dumps(base), encoding="utf-8")

            for name, selected_step, recall in (("E1", 312, 0.7), ("E5", 195, 0.8)):
                output = root / f"{name.lower()}_dev_8ckpt_v1"
                output.mkdir()
                rows = [metric(index + 1, 0.5, 0.75, 0.55) for index in range(8)]
                rows[0] = metric(selected_step, recall, 0.85, 0.7)
                output.joinpath("checkpoint-summary.json").write_text(
                    json.dumps({"selected_step": selected_step, "checkpoints": rows}),
                    encoding="utf-8",
                )

            summary = build_comparison(root, ("E1", "E5"))

            self.assertEqual(summary["recommended"], {"experiment": "E5", "checkpoint_step": 195})
            self.assertAlmostEqual(summary["experiments"][0]["delta_vs_base"]["recall"], 0.3)
            self.assertFalse(summary["test_evaluated"])

    def test_preserves_no_eligible_experiment_without_aborting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base_dev_natural_v1" / "evaluation"
            base_dir.mkdir(parents=True)
            base_dir.joinpath("metrics.json").write_text(
                json.dumps(metric(0, 0.4, 0.7, 0.45)), encoding="utf-8"
            )
            output = root / "e2_dev_8ckpt_v1"
            output.mkdir()
            output.joinpath("checkpoint-summary.json").write_text(
                json.dumps({
                    "selected_step": None,
                    "checkpoints": [
                        metric(index + 1, 0.5, 0.7, 0.5, fpr=0.3)
                        for index in range(8)
                    ],
                }),
                encoding="utf-8",
            )

            summary = build_comparison(root, ("E2",))

            self.assertIsNone(summary["recommended"])
            self.assertFalse(summary["experiments"][0]["eligible"])

    def test_rejects_incomplete_checkpoint_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base_dev_natural_v1" / "evaluation"
            base_dir.mkdir(parents=True)
            base_dir.joinpath("metrics.json").write_text(
                json.dumps(metric(0, 0.4, 0.7, 0.45)), encoding="utf-8"
            )
            output = root / "e3_dev_8ckpt_v1"
            output.mkdir()
            output.joinpath("checkpoint-summary.json").write_text(
                json.dumps({"selected_step": None, "checkpoints": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(SummaryError, "eight checkpoint"):
                build_comparison(root, ("E3",))


if __name__ == "__main__":
    unittest.main()
