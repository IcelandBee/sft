import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_final_test250_priority_review import build_manifest, run_build
from scripts.final_test250_review_web import (
    AnnotationStore,
    BlindReviewError,
    ReviewApplication,
    load_review_rows,
    validate_annotation,
)


def source_row(row: int) -> dict:
    return {
        "row": row,
        "index": row - 1,
        "image_path": f"/images/{row}.png",
        "gold_decision": "GOOD",
        "predictions": {"e1": "BAD", "e2": "BAD"},
        "prediction_payloads": {"e1": {"decision": "BAD"}},
        "wrong_count": 4,
        "stratum": "unanimous_against_gold",
    }


class FinalTest250PriorityReviewTests(unittest.TestCase):
    def test_builder_strips_gold_predictions_and_vote_metadata(self):
        records = build_manifest(
            [source_row(row) for row in range(1, 5)],
            expected_count=4,
            seed=42,
        )
        self.assertEqual(
            set(records[0]), {"review_order", "row", "index", "image_path"}
        )
        self.assertEqual(sorted(record["row"] for record in records), [1, 2, 3, 4])
        self.assertNotEqual([record["row"] for record in records], [1, 2, 3, 4])

    def test_build_and_loader_preserve_blind_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "priority.jsonl"
            source.write_text(
                "".join(json.dumps(source_row(row)) + "\n" for row in range(1, 5)),
                encoding="utf-8",
            )
            output = root / "review"
            summary = run_build(source, output, expected_count=4, seed=7)
            self.assertFalse(summary["contains_original_labels"])
            self.assertFalse(summary["contains_model_predictions"])
            loaded = load_review_rows(output / "review.jsonl", expected_count=4)
            self.assertEqual(len(loaded), 4)
            manifest_text = (output / "review.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("gold", manifest_text)
            self.assertNotIn("prediction", manifest_text)

    def test_annotation_validation_and_csv_are_prediction_free(self):
        complete = validate_annotation(
            {
                "review_decision": "BAD",
                "visible_severity": "obvious",
                "categories": ["手部异常", "其他"],
                "notes": "手指结构异常；餐具异常",
            }
        )
        self.assertTrue(complete["completed"])
        with self.assertRaisesRegex(BlindReviewError, "invalid categories"):
            validate_annotation({"categories": ["未知分类"]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"review_order": 1, "row": 3, "index": 2, "image_path": "/a.png"}
            ]
            store = AnnotationStore(records, root / "annotations.json", root / "reviewed.csv")
            store.save(3, complete)
            with (root / "reviewed.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported[0]["review_decision"], "BAD")
            self.assertEqual(exported[0]["categories"], "手部异常 | 其他")
            self.assertNotIn("gold_decision", exported[0])
            self.assertNotIn("prediction", exported[0])
            app = ReviewApplication(records, store, "token")
            self.assertFalse(app.state()["original_labels_exposed"])
            self.assertFalse(app.state()["model_predictions_exposed"])

    def test_wrapper_builds_once_and_uses_no_gpu_or_inference(self):
        wrapper = Path(
            "scripts/run_final_test250_priority_review_web.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("review_priority.jsonl", wrapper)
        self.assertIn("--expected-count 73", wrapper)
        self.assertIn("build_final_test250_priority_review.py", wrapper)
        self.assertIn("final_test250_review_web.py", wrapper)
        self.assertNotIn("swift infer", wrapper)
        self.assertNotIn("nvidia-smi", wrapper)

    def test_frontend_supports_zoom_multicause_and_hotkeys(self):
        root = Path("web/final-test250-review")
        html = (root / "index.html").read_text(encoding="utf-8")
        script = (root / "app.js").read_text(encoding="utf-8")
        self.assertIn("隐藏原标签与模型预测", html)
        self.assertIn('type="checkbox"', html)
        self.assertIn('classList.toggle("zoomed")', script)
        self.assertIn('key === "g"', script)


if __name__ == "__main__":
    unittest.main()
