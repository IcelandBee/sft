import re
from pathlib import Path
import unittest


SCRIPT = Path("scripts/run_qwen36_base_dev.sh")


class Qwen36BaseDevScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, flag: str) -> str:
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", self.text)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_uses_isolated_qwen36_base_and_frozen_dev(self):
        self.assertIn("/envs/qwen36_27b", self.text)
        self.assertIn("Qwen3.6-27B", self.text)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            self.text,
        )
        self.assertNotIn("--adapters", self.text)
        self.assertNotIn("test.jsonl", self.text.lower())

    def test_uses_natural_deterministic_transformers_generation(self):
        self.assertEqual(self.value("--infer_backend"), "transformers")
        self.assertEqual(self.value("--torch_dtype"), "bfloat16")
        self.assertEqual(self.value("--attn_impl"), "flash_attention_2")
        self.assertEqual(self.value("--add_non_thinking_prefix"), "true")
        self.assertEqual(self.value("--temperature"), "0")
        self.assertEqual(self.value("--max_batch_size"), "1")
        self.assertNotIn("--structured_outputs_regex", self.text)

    def test_writes_protocol_and_strict_evaluation_artifacts(self):
        self.assertIn("qwen36_base_dev_natural_generation_v1", self.text)
        self.assertIn("protocol-manifest.json", self.text)
        self.assertIn("evaluate_e1_dev.py", self.text)
        self.assertIn("--expected-count 200", self.text)
        self.assertIn("QWEN36_BASE_DEV_NATURAL: PASS", self.text)


if __name__ == "__main__":
    unittest.main()
