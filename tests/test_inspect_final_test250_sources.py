from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook
from PIL import Image

from scripts.inspect_final_test250_sources import audit_sources


class FinalTest250SourceInspectionTests(unittest.TestCase):
    def test_reports_workbook_structure_image_matches_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("sample-a.jpg", "sample-b.png"):
                Image.new("RGB", (16, 12), "white").save(image_root / name)

            workbook_path = root / "labels.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "标注"
            sheet.append(["文件名", "结果"])
            sheet.append(["sample-a.jpg", "GOOD"])
            sheet.append(["sample-b.png", "BAD"])
            workbook.save(workbook_path)

            summary = audit_sources(workbook_path, image_root)

            self.assertEqual(summary["images"]["count"], 2)
            self.assertEqual(summary["images"]["unreadable_count"], 0)
            self.assertEqual(summary["workbook"]["sheet_count"], 1)
            profiles = summary["workbook"]["sheets"][0]["column_profiles"]
            self.assertEqual(profiles[0]["image_basename_matches"], 2)
            self.assertEqual(profiles[1]["top_values"][0]["count"], 1)
            self.assertEqual(len(summary["workbook"]["sha256"]), 64)
            self.assertEqual(len(summary["images"]["manifest_sha256"]), 64)
            self.assertTrue(summary["source_only"])
            self.assertFalse(summary["model_inference_run"])

    def test_wrapper_locks_user_paths_and_disables_gpu(self):
        script = Path("scripts/run_inspect_final_test250_sources.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("pose_260428_20260724-v1.xlsx", script)
        self.assertIn("/pose/pose/260428", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertNotIn("swift infer", script)
        source = Path("scripts/inspect_final_test250_sources.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("openpyxl", source)


if __name__ == "__main__":
    unittest.main()
