import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_e1_dev import (
    EvaluationError,
    NON_THINKING_PREFIX,
    SYSTEM_PROMPT,
    USER_PROMPT,
    evaluate_rows,
    parse_prediction,
    run_evaluation,
    validate_payload,
)


def payload(decision="GOOD", categories=None, reasons=None):
    if categories is None:
        categories = [] if decision == "GOOD" else ["手部异常"]
    if reasons is None:
        reasons = [] if decision == "GOOD" else ["手指畸形"]
    return {"decision": decision, "categories": categories, "reasons": reasons}


def result_row(gold="GOOD", prediction="GOOD", image="/images/a.jpg", raw=None):
    predicted_payload = payload(prediction)
    if raw is None:
        raw = NON_THINKING_PREFIX + json.dumps(
            predicted_payload, ensure_ascii=False, separators=(",", ":")
        )
    label = json.dumps(payload(gold), ensure_ascii=False, separators=(",", ":"))
    return {
        "response": raw,
        "labels": label,
        "logprobs": None,
        "images": [{"bytes": None, "path": image}],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": raw},
        ],
        "dataset": "/data/dev.jsonl",
    }


class PredictionParsingTests(unittest.TestCase):
    def test_accepts_only_canonical_empty_think_envelope(self):
        raw = NON_THINKING_PREFIX + '{"decision":"GOOD","categories":[],"reasons":[]}'

        parsed = parse_prediction(raw)

        self.assertTrue(parsed["envelope_valid"])
        self.assertTrue(parsed["payload_json_valid"])
        self.assertTrue(parsed["schema_valid"])
        self.assertFalse(parsed["raw_direct_json_valid"])
        self.assertEqual(parsed["decision"], "GOOD")
        self.assertIsNone(parsed["error_code"])

    def test_rejects_noncanonical_envelopes_and_extra_text(self):
        valid_json = '{"decision":"GOOD","categories":[],"reasons":[]}'
        cases = {
            "missing": (valid_json, "envelope_missing"),
            "nonempty": ("<think>reasoning</think>\n\n" + valid_json, "envelope_noncanonical"),
            "duplicate": (NON_THINKING_PREFIX + NON_THINKING_PREFIX + valid_json, "envelope_duplicate"),
            "markdown": ("```json\n" + valid_json + "\n```", "envelope_missing"),
            "trailing": (NON_THINKING_PREFIX + valid_json + " trailing", "payload_json_invalid"),
        }
        for name, (raw, error_code) in cases.items():
            with self.subTest(name=name):
                parsed = parse_prediction(raw)
                self.assertFalse(parsed["schema_valid"])
                self.assertEqual(parsed["error_code"], error_code)

    def test_rejects_invalid_schema_variants(self):
        cases = [
            (["not-object"], "schema_not_object"),
            ({"decision": "GOOD", "categories": []}, "schema_fields"),
            (payload("MAYBE"), "schema_decision"),
            (payload("GOOD", categories="x"), "schema_categories_type"),
            (payload("GOOD", reasons="x"), "schema_reasons_type"),
            (payload("GOOD", categories=[1]), "schema_categories_item"),
            (payload("GOOD", reasons=[""]), "schema_reasons_item"),
            (payload("GOOD", categories=["手部异常"]), "schema_good_aux"),
            (payload("BAD", categories=[], reasons=["异常"]), "schema_bad_aux"),
            (payload("BAD", categories=["异常"], reasons=[]), "schema_bad_aux"),
            (payload("BAD", categories=["a", "b", "c", "d"], reasons=["r"]), "schema_bad_aux"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(validate_payload(value), expected)


class EvaluationTests(unittest.TestCase):
    def test_invalid_predictions_are_counted_as_wrong(self):
        rows = [
            result_row("BAD", "BAD", "/images/tp.jpg"),
            result_row("BAD", "GOOD", "/images/fn.jpg"),
            result_row("GOOD", "BAD", "/images/fp.jpg"),
            result_row("GOOD", "GOOD", "/images/tn.jpg"),
            result_row("BAD", image="/images/invalid-bad.jpg", raw="not-json"),
            result_row("GOOD", image="/images/invalid-good.jpg", raw="not-json"),
        ]

        parsed, metrics = evaluate_rows(rows, expected_count=6)

        self.assertEqual(
            {key: metrics[key] for key in ("tp", "fn", "fp", "tn")},
            {"tp": 1, "fn": 2, "fp": 2, "tn": 1},
        )
        self.assertEqual(metrics["total"], 6)
        self.assertAlmostEqual(metrics["recall"], 1 / 3)
        self.assertAlmostEqual(metrics["fpr"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["f1"], 1 / 3)
        self.assertAlmostEqual(metrics["schema_valid_rate"], 4 / 6)
        self.assertEqual(metrics["invalid_by_gold"], {"GOOD": 1, "BAD": 1})
        self.assertEqual(sum(row["is_error"] for row in parsed), 4)

    def test_rejects_count_duplicates_bad_gold_and_message_mismatch(self):
        with self.assertRaisesRegex(EvaluationError, "expected 2"):
            evaluate_rows([result_row()], expected_count=2)

        duplicate = [result_row(image="/same.jpg"), result_row(image="/same.jpg")]
        with self.assertRaisesRegex(EvaluationError, "duplicate image"):
            evaluate_rows(duplicate, expected_count=2)

        bad_gold = result_row()
        bad_gold["labels"] = "not-json"
        with self.assertRaisesRegex(EvaluationError, "gold"):
            evaluate_rows([bad_gold], expected_count=1)

        mismatch = result_row()
        mismatch["messages"][-1]["content"] = "different"
        with self.assertRaisesRegex(EvaluationError, "assistant"):
            evaluate_rows([mismatch], expected_count=1)

    def test_run_evaluation_writes_three_deterministic_files_and_refuses_overwrite(self):
        rows = [
            result_row("GOOD", "GOOD", "/images/a.jpg"),
            result_row("BAD", "GOOD", "/images/b.jpg"),
        ]
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "raw.jsonl"
            result_path.write_text(text, encoding="utf-8")
            first = root / "first"
            second = root / "second"

            first_metrics = run_evaluation(result_path, first, expected_count=2)
            second_metrics = run_evaluation(result_path, second, expected_count=2)

            self.assertEqual(first_metrics, second_metrics)
            expected_files = ["errors.jsonl", "metrics.json", "parsed.jsonl"]
            self.assertEqual(sorted(path.name for path in first.iterdir()), expected_files)
            for name in expected_files:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            with self.assertRaisesRegex(EvaluationError, "already exists"):
                run_evaluation(result_path, first, expected_count=2)

    def test_expected_dev_locks_order_prompts_and_gold(self):
        rows = [
            result_row("GOOD", "GOOD", "/images/a.jpg"),
            result_row("BAD", "BAD", "/images/b.jpg"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "raw.jsonl"
            result_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            dev_rows = [
                {
                    "images": [row["images"][0]["path"]],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "assistant", "content": row["labels"]},
                    ],
                }
                for row in rows
            ]
            dev_path = root / "dev.jsonl"
            dev_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dev_rows),
                encoding="utf-8",
            )

            metrics = run_evaluation(
                result_path, root / "valid", expected_count=2, expected_dev=dev_path
            )
            self.assertEqual(len(metrics["dev_sha256"]), 64)

            reversed_path = root / "reversed.jsonl"
            reversed_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in reversed(rows)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationError, "Dev order"):
                run_evaluation(
                    reversed_path,
                    root / "invalid-order",
                    expected_count=2,
                    expected_dev=dev_path,
                )


if __name__ == "__main__":
    unittest.main()
