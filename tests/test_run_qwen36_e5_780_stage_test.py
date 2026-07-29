import re
from pathlib import Path
import unittest


SCRIPT = Path("scripts/run_qwen36_e5_780_stage_test.sh").read_text(encoding="utf-8")


class Qwen36E5780StageTestTests(unittest.TestCase):
    def value(self, flag: str) -> str:
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", SCRIPT)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_locks_dev_selected_e5_780_and_qwen36_environment(self):
        self.assertIn("/envs/qwen36_27b", SCRIPT)
        self.assertIn("Qwen3.6-27B", SCRIPT)
        self.assertIn("v0-20260728-204136/checkpoint-780", SCRIPT)
        self.assertIn('dev_summary.get("selected_step") != 780', SCRIPT)
        self.assertIn('{"experiment": "E5", "checkpoint_step": 780}', SCRIPT)

    def test_locks_readjudicated_test_without_any_selection_sweep(self):
        self.assertIn("test_conditionally_readjudicated.jsonl", SCRIPT)
        self.assertIn(
            "c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f",
            SCRIPT,
        )
        self.assertIn("rows={len(rows)} labels={dict(labels)}", SCRIPT)
        self.assertIn("test_used_for_selection", SCRIPT)
        self.assertNotIn("checkpoint-975", SCRIPT)
        self.assertEqual(SCRIPT.count('"$ENV/bin/swift" infer'), 1)

    def test_uses_same_natural_deterministic_protocol_and_strict_evaluator(self):
        self.assertEqual(self.value("--infer_backend"), "transformers")
        self.assertEqual(self.value("--torch_dtype"), "bfloat16")
        self.assertEqual(self.value("--attn_impl"), "flash_attention_2")
        self.assertEqual(self.value("--add_non_thinking_prefix"), "true")
        self.assertEqual(self.value("--temperature"), "0")
        self.assertEqual(self.value("--max_batch_size"), "1")
        self.assertIn("--expected-count 241", SCRIPT)
        self.assertIn("--checkpoint-step 780", SCRIPT)
        self.assertNotIn("--structured_outputs_regex", SCRIPT)


if __name__ == "__main__":
    unittest.main()
