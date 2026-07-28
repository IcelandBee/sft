import json
from pathlib import Path
import unittest

from scripts.check_qwen36_processor_preflight import (
    classify_stratum,
    select_poc_rows,
)


def make_row(image_count: int, decision: str) -> dict:
    payload = {
        "decision": decision,
        "categories": [] if decision == "GOOD" else ["artifact"],
        "reasons": [] if decision == "GOOD" else ["visible artifact"],
    }
    return {
        "images": [f"image-{index}.png" for index in range(image_count)],
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": json.dumps(payload, separators=(",", ":")),
            },
        ],
    }


class Qwen36ProcessorSelectionTests(unittest.TestCase):
    def test_classifies_all_e5_strata(self):
        self.assertEqual(classify_stratum(make_row(1, "GOOD")), "T1_GOOD")
        self.assertEqual(classify_stratum(make_row(1, "BAD")), "T1_BAD")
        self.assertEqual(classify_stratum(make_row(2, "BAD")), "T2_BAD")
        self.assertEqual(classify_stratum(make_row(2, "GOOD")), "T3_GOOD")

    def test_selects_five_evenly_spaced_rows_per_stratum(self):
        rows = []
        for image_count, decision in ((1, "GOOD"), (1, "BAD"), (2, "BAD"), (2, "GOOD")):
            rows.extend(make_row(image_count, decision) for _ in range(9))

        selected = select_poc_rows(rows)

        self.assertEqual(len(selected), 20)
        counts = {}
        for _, stratum, _ in selected:
            counts[stratum] = counts.get(stratum, 0) + 1
        self.assertEqual(
            counts,
            {"T1_GOOD": 5, "T1_BAD": 5, "T2_BAD": 5, "T3_GOOD": 5},
        )


class Qwen36ProcessorRunnerTests(unittest.TestCase):
    def test_runner_is_offline_cpu_only_and_uses_isolated_environment(self):
        script = Path("scripts/run_qwen36_processor_preflight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("/envs/qwen36_27b", script)
        self.assertIn("Qwen3.6-27B", script)
        self.assertIn("e5_crop_aux20_json_v1", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn("export HF_HUB_OFFLINE=1", script)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", script)
        self.assertIn("export IMAGE_MAX_TOKEN_NUM=1024", script)
        self.assertIn("--max-length 3072", script)
        self.assertIn("required executable is missing", script)
        self.assertIn("required file is missing or unreadable", script)
        self.assertIn("preflight wrapper failed at line", script)
        self.assertNotIn("dev.jsonl", script)
        self.assertNotIn("test.jsonl", script)


if __name__ == "__main__":
    unittest.main()
