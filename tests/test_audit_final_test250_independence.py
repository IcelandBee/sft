from pathlib import Path
import json
import tempfile
import unittest

from openpyxl import Workbook
from PIL import Image

from scripts.audit_final_test250_independence import audit_independence


def ms_row(image):
    return {
        "images": [str(image)],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": '{"decision":"GOOD","categories":[],"reasons":[]}',
            },
        ],
    }


class FinalTest250IndependenceTests(unittest.TestCase):
    def test_detects_uploaded_copies_via_peer_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            workbook_path = root / "labels.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(
                [
                    "id",
                    "first_name",
                    "second_name",
                    "name",
                    "version",
                    "image_version",
                    "group_index",
                    "prompt_result",
                    "图片是否有异常",
                ]
            )
            labels = ["['无异常']"] * 186 + ["['有异常（画框）']"] * 64
            train_copy_dir = root / "train-copy"
            train_copy_dir.mkdir()
            train_rows = []
            for index, label in enumerate(labels, 1):
                group = f"A_{index:05d}"
                peer = image_root / f"{group}_20260428_source.jpg"
                target = image_root / f"{group}_20260724-v1_评测上传.png"
                Image.new("RGB", (8, 8), "white").save(peer)
                Image.new("RGB", (8, 8), "white").save(target)
                sheet.append(
                    [
                        index,
                        "pose",
                        "260428",
                        "user",
                        None,
                        "20260724-v1",
                        group,
                        "[]",
                        label,
                    ]
                )
                if index == 1:
                    copied_peer = train_copy_dir / peer.name
                    Image.new("RGB", (8, 8), "white").save(copied_peer)
                    train_rows.append(ms_row(copied_peer))
            workbook.save(workbook_path)
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in train_rows),
                encoding="utf-8",
            )
            dev_path.write_text("", encoding="utf-8")

            summary = audit_independence(
                workbook_path=workbook_path,
                image_root=image_root,
                train_path=train_path,
                dev_path=dev_path,
            )

            self.assertEqual(summary["test_rows"], 250)
            self.assertEqual(summary["label_counts"], {"BAD": 64, "GOOD": 186})
            self.assertEqual(
                summary["overlap_counts"]["peer_basename_train_overlap"], 1
            )
            self.assertEqual(summary["leakage_group_count"], 1)
            self.assertFalse(summary["independent_from_train_dev"])
            self.assertEqual(summary["status"], "LEAKAGE_DETECTED")

    def test_wrapper_uses_no_gpu_or_inference(self):
        text = Path(
            "scripts/run_audit_final_test250_independence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("e1_broad_clean_json_v1/train.jsonl", text)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertNotIn("swift infer", text)


if __name__ == "__main__":
    unittest.main()
