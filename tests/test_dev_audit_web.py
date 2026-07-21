import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.dev_audit_web import (
    AnnotationStore,
    AuditWebError,
    load_review_rows,
    validate_annotation,
)


def record(row=3, image="/image.jpg"):
    good = {"decision": "GOOD", "categories": [], "reasons": []}
    bad = {"decision": "BAD", "categories": ["手部异常"], "reasons": ["手指异常"]}
    return {
        "row": row,
        "review_group": "e1_only_correct",
        "decision_disagreement": True,
        "image_path": image,
        "gold": good,
        "e1": {"decision": "GOOD", "payload": good, "schema_valid": True},
        "e2": {"decision": "BAD", "payload": bad, "schema_valid": True},
    }


class DevAuditWebTests(unittest.TestCase):
    def test_loads_review_manifest_and_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.jsonl"
            path.write_text(json.dumps(record(), ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(load_review_rows(path)[0]["row"], 3)
            path.write_text(
                (json.dumps(record(), ensure_ascii=False) + "\n") * 2,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditWebError, "duplicate"):
                load_review_rows(path)

    def test_validates_required_enums_and_completion(self):
        complete = validate_annotation(
            {
                "label_status": "gold_correct",
                "visible_severity": "borderline",
                "review_decision": "GOOD",
                "primary_category": "无可见异常",
                "notes": "边界样本",
            }
        )
        self.assertTrue(complete["completed"])
        self.assertFalse(validate_annotation({})["completed"])
        with self.assertRaisesRegex(AuditWebError, "invalid review_decision"):
            validate_annotation({"review_decision": "MAYBE"})

    def test_store_persists_json_and_excel_friendly_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AnnotationStore(
                [record()], root / "annotations.json", root / "reviewed.csv"
            )
            saved = store.save(
                3,
                {
                    "label_status": "gold_incorrect",
                    "visible_severity": "obvious",
                    "review_decision": "BAD",
                    "primary_category": "手部异常",
                    "notes": "应改为 BAD",
                },
            )
            self.assertTrue(saved["completed"])
            self.assertEqual(
                json.loads(
                    (root / "annotations.json").read_text(encoding="utf-8")
                )["3"]["notes"],
                "应改为 BAD",
            )
            self.assertTrue((root / "reviewed.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            with (root / "reviewed.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["review_decision"], "BAD")
            self.assertEqual(rows[0]["gold_decision"], "GOOD")

    def test_store_rejects_unknown_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AnnotationStore([record()], root / "a.json", root / "a.csv")
            with self.assertRaisesRegex(AuditWebError, "unknown review row"):
                store.save(99, {})


if __name__ == "__main__":
    unittest.main()
