import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.build_e4_crop_aux_dataset import (
    E4DatasetError,
    adaptive_crop_scale,
    build_e4_dataset,
)


def label_row(key, image, decision, instances=None):
    return {
        "image_key": key,
        "image_path": str(image),
        "decision": decision,
        "instances": [] if instances is None else instances,
    }


def instance(bbox, category="手部异常", reason="异常"):
    return {"bbox": bbox, "category": category, "reason": reason}


def ms_row(image, decision):
    payload = {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["手部异常"],
        "reasons": [] if decision == "GOOD" else ["异常"],
    }
    return {
        "images": [str(image)],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }


class E4CropAuxDatasetTests(unittest.TestCase):
    def test_adaptive_scale_keeps_small_at_two_and_caps_large_crop_area(self):
        self.assertEqual(adaptive_crop_scale(0.005), (2.0, "small_2.0x"))
        self.assertEqual(adaptive_crop_scale(0.10), (1.5, "standard_1.5x"))
        scale_40, bucket_40 = adaptive_crop_scale(0.40)
        scale_50, bucket_50 = adaptive_crop_scale(0.50)
        self.assertEqual(bucket_40, "large_adaptive")
        self.assertAlmostEqual(scale_40, math.sqrt(0.70 / 0.40))
        self.assertAlmostEqual(scale_50, math.sqrt(0.70 / 0.50))
        self.assertEqual(adaptive_crop_scale(0.80), (1.0, "very_large_1.0x"))

    def test_builds_locked_mix_and_matches_t3_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(8):
                path = root / f"image-{index}.png"
                Image.new("RGB", (100 + index, 80 + index), "white").save(path)
                images.append(path)
            labels = [
                label_row("b1", images[0], "BAD", [instance([10, 10, 20, 20])]),
                label_row(
                    "b2",
                    images[1],
                    "BAD",
                    [instance([0, 0, 50, 50]), instance([60, 20, 90, 60], "文字/符号异常")],
                ),
            ] + [label_row(f"g{i}", images[i], "GOOD") for i in range(2, 8)]
            t1 = [ms_row(row["image_path"], row["decision"]) for row in labels]
            dev_image = root / "dev.png"
            Image.new("RGB", (50, 50), "white").save(dev_image)
            dev = [ms_row(dev_image, "GOOD")]
            dev_bytes = (json.dumps(dev[0], ensure_ascii=False) + "\n").encode("utf-8")
            output = root / "output"

            summary = build_e4_dataset(
                label_rows=labels,
                t1_rows=t1,
                dev_bytes=dev_bytes,
                dev_rows=dev,
                output_dir=output,
                label_sha256="labels",
                t1_sha256="t1",
                dev_sha256="dev",
                local_bad_target=4,
                local_good_target=4,
                max_per_bad_image=2,
            )

            self.assertEqual(
                summary["output"]["sample_type_counts"],
                {"T1_FULL": 8, "T2_BAD": 4, "T3_GOOD": 4},
            )
            self.assertEqual(summary["output"]["train_rows"], 16)
            self.assertEqual((output / "dev.jsonl").read_bytes(), dev_bytes)
            self.assertEqual(len((output / "train.jsonl").read_text(encoding="utf-8").splitlines()), 16)
            manifest = [
                json.loads(line)
                for line in (output / "local_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(manifest), 8)
            self.assertLessEqual(summary["t3_matching"]["max_width_ratio_error"], 0.01)
            self.assertLessEqual(summary["t3_matching"]["max_height_ratio_error"], 0.01)
            self.assertEqual(summary["training_contract"]["max_length"], 3072)
            self.assertTrue(summary["test_untouched"])

    def test_refuses_overwrite_and_train_dev_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "same.png"
            Image.new("RGB", (20, 20), "white").save(image)
            labels = [label_row("g", image, "GOOD")]
            t1 = [ms_row(image, "GOOD")]
            output = root / "exists"
            output.mkdir()
            with self.assertRaisesRegex(E4DatasetError, "already exists"):
                build_e4_dataset(
                    label_rows=labels,
                    t1_rows=t1,
                    dev_bytes=b"",
                    dev_rows=[ms_row(image, "GOOD")],
                    output_dir=output,
                    label_sha256="x",
                    t1_sha256="x",
                    dev_sha256="x",
                    local_bad_target=1,
                    local_good_target=1,
                )


class E4CropAuxWrapperTests(unittest.TestCase):
    def test_wrapper_locks_train_sources_corrected_dev_and_versioned_output(self):
        text = Path("scripts/run_build_e4_crop_aux_dataset.sh").read_text(encoding="utf-8")
        self.assertIn('LABELS="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"', text)
        self.assertIn('T1="$ROOT/ms_swift/e1_broad_clean_json_v1/train.jsonl"', text)
        self.assertIn('DEV="$ROOT/ms_swift/dev_adjudicated_v1/dev.jsonl"', text)
        self.assertIn('OUTPUT="$ROOT/ms_swift/e4_crop_aux_json_v1"', text)
        self.assertIn("--expected-label-good 6074", text)
        self.assertIn("--expected-label-bad 1952", text)
        self.assertIn("--expected-t1-good 6074", text)
        self.assertIn("--expected-t1-bad 3904", text)
        self.assertIn("--expected-dev-good 142", text)
        self.assertIn("--expected-dev-bad 58", text)
        self.assertIn(
            "--expected-dev-sha256 "
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            text,
        )
        self.assertNotIn("test.jsonl", text.lower())


if __name__ == "__main__":
    unittest.main()
