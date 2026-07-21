import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_e1_dev_errors import (
    ErrorAnalysisError,
    analyze_rows,
    run_analysis,
)


def dev_row(image, decision, categories=None, reasons=None):
    categories = categories if categories is not None else []
    reasons = reasons if reasons is not None else []
    gold = {"decision": decision, "categories": categories, "reasons": reasons}
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": json.dumps(gold, ensure_ascii=False),
            },
        ],
    }


def parsed_row(index, image, decision, categories=None, reasons=None, valid=True):
    payload = None
    if valid:
        payload = {
            "decision": decision,
            "categories": categories if categories is not None else [],
            "reasons": reasons if reasons is not None else [],
        }
    return {
        "index": index,
        "image_path": image,
        "predicted_decision": decision if valid else None,
        "schema_valid": valid,
        "payload": payload,
        "error_code": None if valid else "payload_json_invalid",
    }


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dev = [
            dev_row("/a.jpg", "BAD", ["手部异常"], ["手指畸形"]),
            dev_row("/b.jpg", "BAD", ["其他"], ["餐具结构异常"]),
            dev_row("/c.jpg", "GOOD"),
            dev_row("/d.jpg", "GOOD"),
        ]
        self.parsed = [
            parsed_row(0, "/a.jpg", "GOOD"),
            parsed_row(1, "/b.jpg", "BAD", ["其他"], ["餐具异常"]),
            parsed_row(2, "/c.jpg", "BAD", ["面部异常"], ["眼睛异常"]),
            parsed_row(3, "/d.jpg", "GOOD"),
        ]

    def test_summarizes_fn_fp_and_category_recall(self):
        summary, fn_rows, fp_rows = analyze_rows(
            self.dev, self.parsed, checkpoint_step=1248, expected_count=4
        )

        self.assertEqual(
            (summary["tp"], summary["fn"], summary["fp"], summary["tn"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(len(fn_rows), 1)
        self.assertEqual(len(fp_rows), 1)
        self.assertEqual(summary["bad_other_only"]["recall"], 1.0)
        self.assertEqual(
            summary["fn_gold_category_counts"],
            [{"value": "手部异常", "count": 1}],
        )
        by_category = {
            row["category"]: row for row in summary["bad_category_performance"]
        }
        self.assertEqual(by_category["手部异常"]["recall"], 0.0)
        self.assertEqual(by_category["其他"]["recall"], 1.0)

    def test_rejects_alignment_errors(self):
        bad = [dict(row) for row in self.parsed]
        bad[0] = {**bad[0], "image_path": "/wrong.jpg"}
        with self.assertRaisesRegex(ErrorAnalysisError, "image mismatch"):
            analyze_rows(self.dev, bad, checkpoint_step=1248, expected_count=4)

    def test_invalid_predictions_are_scored_conservatively(self):
        parsed = list(self.parsed)
        parsed[1] = parsed_row(1, "/b.jpg", None, valid=False)
        parsed[3] = parsed_row(3, "/d.jpg", None, valid=False)

        summary, fn_rows, fp_rows = analyze_rows(
            self.dev, parsed, checkpoint_step=1248, expected_count=4
        )

        self.assertEqual(
            (summary["tp"], summary["fn"], summary["fp"], summary["tn"]),
            (0, 2, 2, 0),
        )
        self.assertEqual(len(fn_rows), 2)
        self.assertEqual(len(fp_rows), 2)

    def test_writes_atomic_artifacts_and_verifies_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev_path = root / "dev.jsonl"
            parsed_path = root / "parsed.jsonl"
            metrics_path = root / "metrics.json"
            output = root / "analysis"
            dev_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.dev),
                encoding="utf-8",
            )
            parsed_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in self.parsed
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_step": 1248,
                        "total": 4,
                        "tp": 1,
                        "fn": 1,
                        "fp": 1,
                        "tn": 1,
                    }
                ),
                encoding="utf-8",
            )

            summary = run_analysis(
                dev_path,
                parsed_path,
                metrics_path,
                output,
                checkpoint_step=1248,
                expected_count=4,
            )

            self.assertEqual(summary["fn"], 1)
            self.assertEqual(len((output / "fn.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((output / "fp.jsonl").read_text().splitlines()), 1)
            self.assertTrue((output / "summary.json").is_file())
            with self.assertRaisesRegex(ErrorAnalysisError, "already exists"):
                run_analysis(
                    dev_path,
                    parsed_path,
                    metrics_path,
                    output,
                    checkpoint_step=1248,
                    expected_count=4,
                )

            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_step": 1248,
                        "total": 4,
                        "tp": 0,
                        "fn": 1,
                        "fp": 1,
                        "tn": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ErrorAnalysisError, "metrics mismatch for tp"):
                run_analysis(
                    dev_path,
                    parsed_path,
                    metrics_path,
                    root / "analysis-2",
                    checkpoint_step=1248,
                    expected_count=4,
                )


if __name__ == "__main__":
    unittest.main()
