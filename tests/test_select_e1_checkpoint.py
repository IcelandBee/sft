import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_e1_checkpoint import (
    EXPECTED_STEPS,
    SelectionError,
    run_selection,
    select_checkpoint,
)


def metric(step, recall=0.75, fpr=0.20, accuracy=0.74, f1=0.70, schema=1.0):
    return {
        "checkpoint_step": step,
        "total": 200,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
        "schema_valid_rate": schema,
    }


def metrics_with(**overrides):
    rows = [metric(step) for step in EXPECTED_STEPS]
    for step, values in overrides.items():
        target = next(row for row in rows if row["checkpoint_step"] == int(step))
        target.update(values)
    return rows


class SelectionRuleTests(unittest.TestCase):
    def test_recall_is_primary_after_schema_and_fpr_gates(self):
        rows = metrics_with(
            **{
                "312": {"recall": 0.82, "accuracy": 0.72, "f1": 0.69},
                "624": {"recall": 0.80, "accuracy": 0.80, "f1": 0.78},
                "936": {"recall": 0.99, "fpr": 0.26},
                "1248": {"recall": 0.98, "schema_valid_rate": 0.99},
            }
        )

        summary = select_checkpoint(rows)

        self.assertEqual(summary["selected_step"], 312)
        self.assertTrue(summary["test_unlocked"])
        self.assertEqual(summary["eligible_steps"][0], 312)

    def test_ties_use_accuracy_then_f1_then_earlier_step(self):
        rows = metrics_with(
            **{
                "312": {"recall": 0.80, "accuracy": 0.75, "f1": 0.72},
                "624": {"recall": 0.80, "accuracy": 0.76, "f1": 0.70},
                "936": {"recall": 0.80, "accuracy": 0.76, "f1": 0.74},
                "1248": {"recall": 0.80, "accuracy": 0.76, "f1": 0.74},
            }
        )

        summary = select_checkpoint(rows)

        self.assertEqual(summary["selected_step"], 936)

    def test_no_eligible_checkpoint_keeps_test_locked(self):
        rows = [metric(step, fpr=0.30) for step in EXPECTED_STEPS]

        summary = select_checkpoint(rows)

        self.assertIsNone(summary["selected_step"])
        self.assertFalse(summary["test_unlocked"])
        self.assertEqual(summary["eligible_steps"], [])

    def test_rejects_missing_duplicate_and_wrong_sample_count(self):
        with self.assertRaisesRegex(SelectionError, "missing checkpoint"):
            select_checkpoint([metric(step) for step in EXPECTED_STEPS[:-1]])

        duplicate = [metric(step) for step in EXPECTED_STEPS]
        duplicate[-1]["checkpoint_step"] = EXPECTED_STEPS[0]
        with self.assertRaisesRegex(SelectionError, "duplicate checkpoint"):
            select_checkpoint(duplicate)

        wrong_count = [metric(step) for step in EXPECTED_STEPS]
        wrong_count[0]["total"] = 199
        with self.assertRaisesRegex(SelectionError, "total=200"):
            select_checkpoint(wrong_count)


class SelectionCliTests(unittest.TestCase):
    def test_run_selection_reads_all_metrics_and_refuses_overwrite(self):
        rows = metrics_with(**{"624": {"recall": 0.90}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for row in rows:
                path = root / f"checkpoint-{row['checkpoint_step']}" / "evaluation"
                path.mkdir(parents=True)
                (path / "metrics.json").write_text(
                    json.dumps(row), encoding="utf-8"
                )
            output = root / "checkpoint-summary.json"

            summary = run_selection(root, output)

            self.assertEqual(summary["selected_step"], 624)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary)
            with self.assertRaisesRegex(SelectionError, "already exists"):
                run_selection(root, output)

    def test_run_selection_rejects_metric_step_that_does_not_match_folder(self):
        rows = [metric(step) for step in EXPECTED_STEPS]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for row in rows:
                path = root / f"checkpoint-{row['checkpoint_step']}" / "evaluation"
                path.mkdir(parents=True)
                stored = dict(row)
                if row["checkpoint_step"] == 312:
                    stored["checkpoint_step"] = 624
                (path / "metrics.json").write_text(
                    json.dumps(stored), encoding="utf-8"
                )
            with self.assertRaisesRegex(SelectionError, "does not match folder"):
                run_selection(root, root / "summary.json")


if __name__ == "__main__":
    unittest.main()
