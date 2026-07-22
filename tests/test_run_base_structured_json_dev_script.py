import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_base_structured_json_dev.sh"


class BaseStructuredJsonScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_base_model_without_adapter_and_frozen_corrected_dev(self):
        self.assertIn("Qwen3.5-27B", self.text)
        self.assertNotIn("--adapters", self.text)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn("cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb", self.text)

    def test_locks_vllm_guided_decoding_and_deterministic_generation(self):
        self.assertIn("--infer_backend vllm", self.text)
        self.assertIn("--structured_outputs_regex", self.text)
        self.assertIn("--add_non_thinking_prefix true", self.text)
        self.assertIn("--temperature 0", self.text)
        self.assertIn("--seed 42", self.text)

    def test_does_not_reference_test_data(self):
        self.assertNotIn("/test", self.text.lower())
        self.assertIn('"test_untouched": True', self.text)


if __name__ == "__main__":
    unittest.main()
