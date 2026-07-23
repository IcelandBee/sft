import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.e4_crop_review_web import (
    CropReviewError,
    CropReviewStore,
    annotated_image_bytes,
    load_manifest,
    validate_review,
)


class E4CropReviewWebTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> Path:
        source = root / "source.png"
        crop = root / "crop.png"
        Image.new("RGB", (100, 80), "white").save(source)
        Image.new("RGB", (30, 30), "white").save(crop)
        value = {
            "source_scope": "broad_clean_train_only",
            "test_untouched": True,
            "dev_untouched": True,
            "samples": [
                {
                    "index": 0,
                    "sample_type": "T2_BAD",
                    "image_key": "bad",
                    "source_image": str(source),
                    "crop_image": str(crop),
                    "crop_box": [5, 5, 50, 50],
                    "bbox": [10, 10, 20, 20],
                    "bbox_area_ratio": 0.0125,
                    "crop_scale": 1.5,
                    "selection_reason": "手部异常:min_area",
                    "payload": {"decision": "BAD", "categories": ["手部异常"], "reasons": ["多指"]},
                }
            ],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_train_only_manifest_and_renders_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            records = load_manifest(self.make_manifest(Path(directory)))
            self.assertEqual(len(records), 1)
            payload = annotated_image_bytes(records[0])
            self.assertTrue(payload.startswith(b"\xff\xd8"))

    def test_validates_status_issue_contract(self):
        self.assertEqual(validate_review({"status": "pass"})["status"], "pass")
        self.assertEqual(
            validate_review({"status": "fail", "issues": ["bbox_misaligned"]})["issues"],
            ["bbox_misaligned"],
        )
        with self.assertRaisesRegex(CropReviewError, "requires at least one"):
            validate_review({"status": "fail", "issues": []})
        with self.assertRaisesRegex(CropReviewError, "PASS"):
            validate_review({"status": "pass", "issues": ["other"]})

    def test_store_persists_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = load_manifest(self.make_manifest(root))
            store = CropReviewStore(records, root / "annotations.json", root / "reviewed.csv")
            saved = store.save(0, {"status": "fail", "issues": ["context_too_little"], "notes": "太紧"})
            self.assertTrue(saved["completed"])
            self.assertTrue((root / "annotations.json").is_file())
            self.assertIn("context_too_little", (root / "reviewed.csv").read_text(encoding="utf-8-sig"))

    def test_rejects_non_train_or_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_manifest(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_scope"] = "dev"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CropReviewError, "Train-only"):
                load_manifest(path)


class E4CropReviewScriptTests(unittest.TestCase):
    def test_wrapper_uses_poc_manifest_and_separate_review_output(self):
        text = Path("scripts/run_e4_crop_review_web.sh").read_text(encoding="utf-8")
        self.assertIn('MANIFEST="$PREFLIGHT/poc/manifest.json"', text)
        self.assertIn('OUTPUT="$PREFLIGHT/crop-review-v1"', text)
        self.assertNotIn("dev.jsonl", text.lower())
        self.assertNotIn("test.jsonl", text.lower())


if __name__ == "__main__":
    unittest.main()
