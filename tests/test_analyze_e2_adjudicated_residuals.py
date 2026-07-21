import json
import unittest

from scripts.analyze_e2_adjudicated_residuals import analyze_residuals


def payload(decision, category="手部异常"):
    return {
        "decision": decision,
        "categories": [] if decision == "GOOD" else [category],
        "reasons": [] if decision == "GOOD" else ["异常"],
    }


def data_row(image, decision, category="手部异常"):
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload(decision, category), ensure_ascii=False)},
        ],
    }


def prediction(index, image, decision, category="手部异常"):
    return {
        "index": index,
        "image_path": image,
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": payload(decision, category),
    }


class E2ResidualAnalysisTests(unittest.TestCase):
    def test_stratifies_residuals_and_computes_challenge_gap(self):
        dev = [
            data_row("/a.jpg", "BAD", "手部异常"),
            data_row("/b.jpg", "GOOD"),
            data_row("/c.jpg", "BAD", "文字/符号异常"),
            data_row("/d.jpg", "GOOD"),
        ]
        original = [
            data_row("/a.jpg", "GOOD"),
            data_row("/b.jpg", "GOOD"),
            data_row("/c.jpg", "BAD", "文字/符号异常"),
            data_row("/d.jpg", "GOOD"),
        ]
        e2 = [
            prediction(0, "/a.jpg", "GOOD"),
            prediction(1, "/b.jpg", "BAD", "面部/五官异常"),
            prediction(2, "/c.jpg", "BAD", "文字/符号异常"),
            prediction(3, "/d.jpg", "GOOD"),
        ]
        annotations = {
            1: {
                "visible_severity": "obvious",
                "primary_category": "手部异常",
                "notes": "六指",
            },
            2: {
                "visible_severity": "none",
                "primary_category": "无可见异常",
                "notes": "",
            },
        }
        train = [
            data_row("/t1.jpg", "BAD", "手部异常"),
            data_row("/t1.jpg", "BAD", "手部异常"),
            data_row("/t2.jpg", "BAD", "文字/符号异常"),
            data_row("/t3.jpg", "GOOD"),
        ]

        summary, fn_rows, fp_rows = analyze_residuals(
            dev,
            original,
            e2,
            annotations,
            train,
            expected_count=4,
            expected_fn=1,
            expected_fp=1,
        )

        self.assertEqual(summary["confusion"], {"tp": 1, "fn": 1, "fp": 1, "tn": 1})
        self.assertEqual(fn_rows[0]["label_origin"], "corrected_gold")
        self.assertEqual(fn_rows[0]["visible_severity"], "obvious")
        self.assertEqual(fp_rows[0]["visible_severity"], "none")
        self.assertEqual(summary["train_unique_bad"], 2)
        self.assertEqual(
            summary["train_bad_category_counts"],
            [
                {"value": "手部异常", "count": 1},
                {"value": "文字/符号异常", "count": 1},
            ],
        )
        self.assertEqual(
            summary["challenge_gap"]["minimum_additional_tp_if_no_regressions"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
