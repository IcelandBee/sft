import json
from pathlib import Path
import unittest

from scripts.run_qwen36_inference_poc import (
    select_balanced_dev,
    validate_generated_payload,
)


def row(decision: str) -> dict:
    payload = {"decision": decision, "categories": [], "reasons": []}
    return {
        "images": [f"{decision}.png"],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
    }


class Qwen36InferencePocTests(unittest.TestCase):
    def test_selects_balanced_single_image_dev_rows(self):
        rows = [row("GOOD") for _ in range(12)] + [row("BAD") for _ in range(8)]
        selected = select_balanced_dev(rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            [stratum for _, stratum, _ in selected].count("DEV_GOOD"), 5
        )
        self.assertEqual(
            [stratum for _, stratum, _ in selected].count("DEV_BAD"), 5
        )

    def test_accepts_only_exact_json_schema(self):
        valid = '{"decision":"GOOD","categories":[],"reasons":[]}'
        payload, error = validate_generated_payload(valid)
        self.assertIsNone(error)
        self.assertEqual(payload["decision"], "GOOD")

        for invalid in (
            "```json\n" + valid + "\n```",
            '{"decision":"MAYBE","categories":[],"reasons":[]}',
            '{"decision":"GOOD","categories":[]}',
        ):
            payload, error = validate_generated_payload(invalid)
            self.assertIsNone(payload)
            self.assertIsNotNone(error)

    def test_wrapper_uses_isolated_offline_transformers_gpu4_protocol(self):
        script = Path("scripts/run_qwen36_inference_poc.sh").read_text(encoding="utf-8")
        self.assertIn("/envs/qwen36_27b", script)
        self.assertIn("Qwen3.6-27B", script)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", script)
        self.assertIn("e5_crop_aux20_json_v1/train.jsonl", script)
        self.assertIn("cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=\"$GPU\"", script)
        self.assertIn("export HF_HUB_OFFLINE=1", script)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", script)
        self.assertIn("export IMAGE_MAX_TOKEN_NUM=1024", script)
        self.assertIn("for CHECK_GPU in 4 5 6 7", script)
        self.assertNotIn("/test", script.lower())


if __name__ == "__main__":
    unittest.main()
