import json
import unittest
from unittest.mock import patch

from scripts.rescore_base_adjudicated_dev import rescore_rows


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


def result_row(image, original_decision, response="response"):
    return {
        "images": [{"path": image, "bytes": None}],
        "labels": json.dumps(payload(original_decision), ensure_ascii=False),
        "response": response,
    }


def parsed(decision, schema=True, payload_valid=True):
    value = payload(decision) if decision else None
    return {
        "envelope_valid": True,
        "payload_json_valid": payload_valid,
        "schema_valid": schema,
        "decision": decision if schema else None,
        "payload": value,
        "error_code": None if schema else "schema_fields",
    }


class BaseAdjudicatedRescoreTests(unittest.TestCase):
    def test_rescores_corrected_labels_with_strict_and_decision_only_protocols(self):
        original = [
            dev_row("/a.jpg", "BAD"),
            dev_row("/b.jpg", "BAD"),
            dev_row("/c.jpg", "GOOD"),
            dev_row("/d.jpg", "GOOD"),
        ]
        corrected = [
            dev_row("/a.jpg", "GOOD"),
            dev_row("/b.jpg", "BAD"),
            dev_row("/c.jpg", "BAD"),
            dev_row("/d.jpg", "GOOD"),
        ]
        results = [
            result_row("/a.jpg", "BAD"),
            result_row("/b.jpg", "BAD"),
            result_row("/c.jpg", "GOOD"),
            result_row("/d.jpg", "GOOD"),
        ]
        parsed_values = [
            parsed("BAD"),
            parsed("GOOD"),
            parsed("BAD", schema=False),
            parsed(None, schema=False, payload_valid=False),
        ]

        with patch("scripts.rescore_base_adjudicated_dev.parse_prediction", side_effect=parsed_values):
            summary, rows = rescore_rows(
                original,
                corrected,
                results,
                expected_count=4,
                enforce_original_contract=False,
            )

        self.assertEqual(len(rows), 4)
        strict = summary["corrected_dev"]["strict_schema"]
        decision = summary["corrected_dev"]["decision_only"]
        self.assertEqual(
            (strict["tp"], strict["fn"], strict["fp"], strict["tn"]),
            (0, 2, 2, 0),
        )
        self.assertEqual(
            (decision["tp"], decision["fn"], decision["fp"], decision["tn"]),
            (1, 1, 2, 0),
        )

    def test_rejects_result_alignment_drift_before_scoring(self):
        original = [dev_row("/a.jpg", "GOOD")]
        corrected = [dev_row("/a.jpg", "GOOD")]
        results = [result_row("/wrong.jpg", "GOOD")]
        with self.assertRaisesRegex(ValueError, "image mismatch"):
            rescore_rows(original, corrected, results, expected_count=1)


if __name__ == "__main__":
    unittest.main()
