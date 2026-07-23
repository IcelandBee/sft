from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.build_e4_token_poc import build_token_poc, select_representative_samples
from scripts.check_e4_token_lengths import summarize_token_lengths


def instance(bbox, category, reason="异常"):
    return {"bbox": bbox, "category": category, "reason": reason}


def row(key, image, decision, instances=None):
    return {
        "image_key": key,
        "image_path": str(image),
        "decision": decision,
        "instances": [] if instances is None else instances,
    }


class E4TokenPocTests(unittest.TestCase):
    def test_builds_two_image_bad_and_good_samples_without_dev_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(5):
                path = root / f"image-{index}.png"
                Image.new("RGB", (100, 100), "white").save(path)
                images.append(path)
            rows = [
                row("g1", images[0], "GOOD"),
                row("g2", images[1], "GOOD"),
                row("b1", images[2], "BAD", [instance([10, 10, 20, 20], "手部异常")]),
                row("b2", images[3], "BAD", [instance([0, 0, 80, 80], "手部异常")]),
                row("b3", images[4], "BAD", [instance([20, 20, 40, 40], "文字/符号异常")]),
            ]
            output = root / "poc"

            manifest = build_token_poc(rows, output, source_sha256="abc")

            self.assertTrue((output / "poc.jsonl").is_file())
            self.assertEqual(manifest["source_scope"], "broad_clean_train_only")
            self.assertTrue(manifest["test_untouched"])
            self.assertTrue(manifest["dev_untouched"])
            self.assertGreaterEqual(manifest["sample_counts"]["T2_BAD"], 3)
            self.assertEqual(manifest["sample_counts"]["T3_GOOD"], 2)
            for sample in manifest["samples"]:
                self.assertTrue(Path(sample["crop_image"]).is_file())

    def test_summary_recommends_length_above_observed_max(self):
        summary = summarize_token_lengths(
            [
                {"sample_type": "T2_BAD", "total_tokens": 2100},
                {"sample_type": "T2_BAD", "total_tokens": 2500},
                {"sample_type": "T3_GOOD", "total_tokens": 1800},
            ]
        )
        self.assertEqual(summary["rows_exceeding"]["2048"], 2)
        self.assertEqual(summary["poc_recommended_max_length_with_64_token_margin"], 3072)


class E4TokenPreflightScriptTests(unittest.TestCase):
    def test_wrapper_disables_gpu_and_reads_train_only(self):
        script = Path("scripts/run_e4_token_preflight.sh").read_text(encoding="utf-8")
        self.assertIn('TRAIN="$ROOT/splits/dev200_v1_broad_clean/train.jsonl"', script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn("export IMAGE_MAX_TOKEN_NUM=1024", script)
        self.assertNotIn("/dev.jsonl", script)
        self.assertNotIn("test.jsonl", script.lower())


if __name__ == "__main__":
    unittest.main()
