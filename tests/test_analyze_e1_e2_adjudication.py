import json
import unittest

from scripts.analyze_e1_e2_adjudication import analyze_adjudication
from scripts.build_e1_e2_dev_audit import compare_rows


def payload(decision):
    return {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["手部异常"],
        "reasons": [] if decision == "GOOD" else ["手指异常"],
    }


def dev_row(image, decision):
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload(decision), ensure_ascii=False)},
        ],
    }


def parsed_row(index, image, decision):
    return {
        "index": index,
        "image_path": image,
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": payload(decision),
        "error_code": None,
    }


def annotation(label_status, severity, decision, category=""):
    return {
        "label_status": label_status,
        "visible_severity": severity,
        "review_decision": decision,
        "primary_category": category,
        "notes": "",
        "completed": True,
        "updated_at": "2026-07-21T00:00:00+00:00",
    }


class AdjudicationAnalysisTests(unittest.TestCase):
    def test_runner_allows_non_destructive_versioned_output(self):
        from pathlib import Path

        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "run_e1_e2_adjudication_analysis.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ADJUDICATION_OUT", script)

    def setUp(self):
        self.dev = [
            dev_row("/a.jpg", "GOOD"),
            dev_row("/b.jpg", "BAD"),
            dev_row("/c.jpg", "GOOD"),
            dev_row("/d.jpg", "BAD"),
        ]
        self.e1 = [
            parsed_row(0, "/a.jpg", "GOOD"),
            parsed_row(1, "/b.jpg", "GOOD"),
            parsed_row(2, "/c.jpg", "BAD"),
            parsed_row(3, "/d.jpg", "BAD"),
        ]
        self.e2 = [
            parsed_row(0, "/a.jpg", "GOOD"),
            parsed_row(1, "/b.jpg", "GOOD"),
            parsed_row(2, "/c.jpg", "GOOD"),
            parsed_row(3, "/d.jpg", "GOOD"),
        ]
        _, self.review, _, _ = compare_rows(
            self.dev, self.e1, self.e2, expected_count=4
        )

    def test_recomputes_metrics_after_binary_label_changes(self):
        annotations = {
            2: annotation("gold_incorrect", "none", "GOOD", "无可见异常"),
            3: annotation("gold_correct", "none", "GOOD", "无可见异常"),
            4: annotation("gold_correct", "obvious", "BAD", "手部异常"),
        }
        summary, merged, changes, uncertain = analyze_adjudication(
            self.dev,
            self.e1,
            self.e2,
            self.review,
            annotations,
            expected_count=4,
        )

        self.assertTrue(summary["analysis_ready"])
        self.assertEqual(summary["binary_label_changes"], 1)
        self.assertEqual(len(merged), 3)
        self.assertEqual(len(changes), 1)
        self.assertEqual(uncertain, [])
        adjusted = summary["metrics"]["conditionally_adjusted_full_dev_excluding_unsure"]
        self.assertEqual(adjusted["e1"]["accuracy"], 0.75)
        self.assertEqual(adjusted["e2"]["accuracy"], 0.75)

    def test_excludes_unsure_and_reports_consistency_issues(self):
        annotations = {
            2: annotation("uncertain", "uncertain", "UNSURE"),
            3: annotation("gold_correct", "obvious", "GOOD"),
            4: annotation("gold_incorrect", "obvious", "BAD"),
        }
        summary, _, _, uncertain = analyze_adjudication(
            self.dev,
            self.e1,
            self.e2,
            self.review,
            annotations,
            expected_count=4,
        )

        self.assertFalse(summary["analysis_ready"])
        self.assertEqual(summary["excluded_unsure_rows"], [2])
        self.assertEqual(len(uncertain), 1)
        codes = {issue["code"] for issue in summary["consistency_issues"]}
        self.assertIn("obvious_anomaly_but_good", codes)
        self.assertIn("gold_incorrect_but_unchanged", codes)
        adjusted = summary["metrics"]["conditionally_adjusted_full_dev_excluding_unsure"]
        self.assertEqual(adjusted["e1"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
