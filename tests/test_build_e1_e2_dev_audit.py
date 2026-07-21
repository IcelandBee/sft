import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_e1_e2_dev_audit import AuditError, compare_rows, run_audit


def dev_row(image, decision):
    payload = {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["手部异常"],
        "reasons": [] if decision == "GOOD" else ["手指异常"],
    }
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def parsed_row(index, image, decision):
    payload = {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["手部异常"],
        "reasons": [] if decision == "GOOD" else ["手指异常"],
    }
    return {
        "index": index,
        "image_path": image,
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": payload,
        "error_code": None,
    }


class BoundaryAuditTests(unittest.TestCase):
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

    def test_groups_boundary_cases(self):
        summary, review, both_wrong, disagreements = compare_rows(
            self.dev, self.e1, self.e2, expected_count=4
        )

        self.assertEqual(summary["both_correct"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["e1_only_correct"], 1)
        self.assertEqual(summary["e2_only_correct"], 1)
        self.assertEqual(summary["decision_disagreements"], 2)
        self.assertEqual(summary["review_total"], 3)
        self.assertEqual([row["row"] for row in disagreements], [3, 4])
        self.assertEqual([row["row"] for row in both_wrong], [2])
        self.assertEqual([row["row"] for row in review], [3, 4, 2])

    def test_rejects_alignment_drift(self):
        broken = [dict(row) for row in self.e2]
        broken[0] = {**broken[0], "image_path": "/wrong.jpg"}
        with self.assertRaisesRegex(AuditError, "E2 image mismatch"):
            compare_rows(self.dev, self.e1, broken, expected_count=4)

    def test_writes_review_package_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, rows in (("dev", self.dev), ("e1", self.e1), ("e2", self.e2)):
                path = root / f"{name}.jsonl"
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[name] = path

            output = root / "audit"
            summary = run_audit(
                paths["dev"], paths["e1"], paths["e2"], output, expected_count=4
            )

            self.assertEqual(summary["review_total"], 3)
            self.assertEqual(len((output / "both-wrong.jsonl").read_text().splitlines()), 1)
            self.assertEqual(
                len((output / "decision-disagreements.jsonl").read_text().splitlines()), 2
            )
            with (output / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["review_order"], "1")
            self.assertEqual(rows[0]["review_label_status"], "")
            with self.assertRaisesRegex(AuditError, "already exists"):
                run_audit(
                    paths["dev"], paths["e1"], paths["e2"], output, expected_count=4
                )


if __name__ == "__main__":
    unittest.main()
