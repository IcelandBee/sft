import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.audit_e4_bbox_coverage import BboxAuditError, audit_bbox_coverage


def instance(bbox, category="手部异常", reason="异常"):
    return {"bbox": bbox, "category": category, "reason": reason}


def row(key, image, decision, instances=None):
    instances = [] if instances is None else instances
    return {
        "image_key": key,
        "image_path": str(image),
        "decision": decision,
        "instances": instances,
    }


class E4BboxCoverageAuditTests(unittest.TestCase):
    def test_summarizes_bbox_scale_categories_and_crop_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{name}.png" for name in ("good", "bad1", "bad2")]
            for path in paths:
                Image.new("RGB", (100, 100), "white").save(path)
            rows = [
                row("g", paths[0], "GOOD"),
                row("b1", paths[1], "BAD", [instance([10, 10, 20, 20])]),
                row(
                    "b2",
                    paths[2],
                    "BAD",
                    [
                        instance([0, 0, 50, 50]),
                        instance([50, 50, 100, 100], "文字/符号异常"),
                    ],
                ),
            ]

            summary = audit_bbox_coverage(
                rows,
                source_sha256="abc",
                small_area_threshold=0.01,
                max_crops_per_bad_image=2,
                t1_rows=6,
                local_share=0.40,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["decision_counts"], {"BAD": 2, "GOOD": 1})
            self.assertEqual(summary["valid_bbox_instances"], 3)
            self.assertEqual(summary["small_bbox_instances"], 1)
            self.assertEqual(
                summary["category_instance_counts"],
                [
                    {"value": "手部异常", "count": 2},
                    {"value": "文字/符号异常", "count": 1},
                ],
            )
            projection = summary["crop_projection"]
            self.assertEqual(projection["local_total_target"], 4)
            self.assertEqual(projection["local_bad_target"], 2)
            self.assertEqual(projection["abnormal_candidates_uncapped"], 4)
            self.assertEqual(projection["abnormal_candidates_capped"], 4)
            self.assertEqual(projection["normal_candidates_two_per_good_image"], 2)

    def test_records_out_of_bounds_and_forbidden_other_without_using_them(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "bad.png"
            Image.new("RGB", (100, 100), "white").save(image)
            rows = [
                row(
                    "bad",
                    image,
                    "BAD",
                    [
                        instance([-1, 0, 10, 10]),
                        instance([20, 20, 30, 30], "其他"),
                    ],
                )
            ]

            summary = audit_bbox_coverage(rows, source_sha256="abc")

            self.assertEqual(summary["status"], "NEEDS_REVIEW")
            self.assertEqual(summary["valid_bbox_instances"], 0)
            self.assertEqual(
                summary["issues"]["counts"],
                {"bbox_out_of_bounds": 1, "forbidden_other_category": 1},
            )

    def test_rejects_bad_without_instances_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (10, 10), "white").save(image)
            with self.assertRaisesRegex(BboxAuditError, "must contain instances"):
                audit_bbox_coverage([row("bad", image, "BAD")], source_sha256="abc")
            with self.assertRaisesRegex(BboxAuditError, "duplicate image_key"):
                audit_bbox_coverage(
                    [row("same", image, "GOOD"), row("same", image.with_name("x.png"), "GOOD")],
                    source_sha256="abc",
                )


class E4BboxAuditScriptContractTests(unittest.TestCase):
    def test_wrapper_is_train_only_and_uses_frozen_counts(self):
        script = Path("scripts/run_e4_bbox_audit.sh").read_text(encoding="utf-8")
        self.assertIn('TRAIN="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"', script)
        self.assertIn("--expected-rows 8026", script)
        self.assertIn("--expected-good 6074", script)
        self.assertIn("--expected-bad 1952", script)
        self.assertNotIn("/dev.jsonl", script)
        self.assertNotIn("test.jsonl", script.lower())


if __name__ == "__main__":
    unittest.main()
