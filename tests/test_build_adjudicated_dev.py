import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_adjudicated_dev import AdjudicatedDevError, build_rows, run_build


def payload(decision):
    return {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["手部异常"],
        "reasons": [] if decision == "GOOD" else ["手指异常"],
    }


def dev_row(image, decision):
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload(decision), ensure_ascii=False)},
        ],
    }


def review_row(row, image, decision):
    gold = payload(decision)
    return {
        "row": row,
        "review_group": "both_wrong",
        "decision_disagreement": False,
        "image_path": image,
        "gold": gold,
        "e1": {"decision": "BAD", "payload": payload("BAD"), "schema_valid": True},
        "e2": {"decision": "BAD", "payload": payload("BAD"), "schema_valid": True},
    }


def annotation(status, decision, category="", notes=""):
    return {
        "label_status": status,
        "visible_severity": "obvious" if decision == "BAD" else "none",
        "review_decision": decision,
        "primary_category": category,
        "notes": notes,
        "completed": True,
    }


class BuildAdjudicatedDevTests(unittest.TestCase):
    def setUp(self):
        self.dev = [dev_row("/a.jpg", "GOOD"), dev_row("/b.jpg", "BAD")]
        self.review = [
            review_row(1, "/a.jpg", "GOOD"),
            review_row(2, "/b.jpg", "BAD"),
        ]
        self.annotations = {
            1: annotation("gold_incorrect", "BAD", "手部异常", "右手六指"),
            2: annotation("gold_incorrect", "GOOD"),
        }

    def test_builds_schema_valid_binary_corrections(self):
        corrected, changes, summary = build_rows(
            self.dev,
            self.review,
            self.annotations,
            expected_count=2,
            expected_review=2,
            expected_changes=2,
        )
        values = [json.loads(row["messages"][-1]["content"]) for row in corrected]
        self.assertEqual(values[0], payload("BAD") | {"reasons": ["右手六指"]})
        self.assertEqual(values[1], payload("GOOD"))
        self.assertEqual(len(changes), 2)
        self.assertEqual(summary["adjudicated_counts"], {"BAD": 1, "GOOD": 1})

    def test_requires_auxiliary_supervision_for_good_to_bad(self):
        annotations = dict(self.annotations)
        annotations[1] = annotation("gold_incorrect", "BAD")
        with self.assertRaisesRegex(AdjudicatedDevError, "requires primary_category"):
            build_rows(
                self.dev,
                self.review,
                annotations,
                expected_count=2,
                expected_review=2,
                expected_changes=2,
            )

    def test_writes_versioned_dev_manifest_and_change_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev_path = root / "dev.jsonl"
            review_path = root / "review.jsonl"
            annotations_path = root / "annotations.json"
            dev_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.dev),
                encoding="utf-8",
            )
            review_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.review),
                encoding="utf-8",
            )
            annotations_path.write_text(
                json.dumps({str(k): v for k, v in self.annotations.items()}, ensure_ascii=False),
                encoding="utf-8",
            )
            output = root / "adjudicated"
            manifest = run_build(
                dev_path,
                review_path,
                annotations_path,
                output,
                expected_count=2,
                expected_review=2,
                expected_changes=2,
            )
            self.assertTrue(manifest["training_forbidden"])
            self.assertEqual(len((output / "dev.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                len((output / "decision-changes.jsonl").read_text(encoding="utf-8").splitlines()),
                2,
            )
            self.assertTrue((output / "manifest.json").is_file())
            with self.assertRaisesRegex(AdjudicatedDevError, "already exists"):
                run_build(
                    dev_path,
                    review_path,
                    annotations_path,
                    output,
                    expected_count=2,
                    expected_review=2,
                    expected_changes=2,
                )


if __name__ == "__main__":
    unittest.main()
