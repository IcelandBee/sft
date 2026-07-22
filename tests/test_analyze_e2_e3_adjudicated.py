import json
import unittest

from scripts.analyze_e2_e3_adjudicated import analyze_comparison


def dev_row(image, decision):
    payload = {
        "decision": decision,
        "categories": ["手部异常"] if decision == "BAD" else [],
        "reasons": ["手指异常"] if decision == "BAD" else [],
    }
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def prediction(index, image, decision):
    payload = {
        "decision": decision,
        "categories": ["手部异常"] if decision == "BAD" else [],
        "reasons": ["手指异常"] if decision == "BAD" else [],
    }
    return {
        "index": index,
        "image_path": image,
        "schema_valid": True,
        "predicted_decision": decision,
        "payload": payload,
    }


class E2E3ComparisonTests(unittest.TestCase):
    def test_attributes_all_four_binary_decision_transitions(self):
        images = [f"/{index}.jpg" for index in range(4)]
        dev = [
            dev_row(images[0], "BAD"),
            dev_row(images[1], "BAD"),
            dev_row(images[2], "GOOD"),
            dev_row(images[3], "GOOD"),
        ]
        original = list(dev)
        e2 = [
            prediction(0, images[0], "BAD"),
            prediction(1, images[1], "GOOD"),
            prediction(2, images[2], "GOOD"),
            prediction(3, images[3], "BAD"),
        ]
        e3 = [
            prediction(0, images[0], "GOOD"),
            prediction(1, images[1], "BAD"),
            prediction(2, images[2], "BAD"),
            prediction(3, images[3], "GOOD"),
        ]

        summary, records = analyze_comparison(
            dev,
            original,
            e2,
            e3,
            {},
            expected_count=4,
        )

        self.assertEqual(summary["paired_outcomes"], {
            "e2_only_correct": 2,
            "e3_only_correct": 2,
        })
        self.assertEqual(summary["decision_differences"], 4)
        self.assertEqual(len(records["e2_only_correct"]), 2)
        self.assertEqual(len(records["e3_only_correct"]), 2)
        self.assertEqual(len(records["e3_fn"]), 1)
        self.assertEqual(len(records["e3_fp"]), 1)

    def test_uses_adjudicated_gold_and_tracks_label_origin(self):
        image = "/changed.jpg"
        dev = [dev_row(image, "BAD")]
        original = [dev_row(image, "GOOD")]
        e2 = [prediction(0, image, "BAD")]
        e3 = [prediction(0, image, "GOOD")]
        annotations = {1: {"visible_severity": "obvious", "notes": "多出手指"}}

        summary, records = analyze_comparison(
            dev,
            original,
            e2,
            e3,
            annotations,
            expected_count=1,
        )

        self.assertEqual(summary["e2"]["confusion"]["tp"], 1)
        self.assertEqual(summary["e3"]["confusion"]["fn"], 1)
        self.assertEqual(records["e2_only_correct"][0]["label_origin"], "corrected_gold")


if __name__ == "__main__":
    unittest.main()
