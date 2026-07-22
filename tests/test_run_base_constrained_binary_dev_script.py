import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_base_constrained_binary_dev.sh"


class BaseConstrainedBinaryScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_transformers_base_model_and_corrected_dev(self):
        self.assertIn("Qwen3.5-27B", self.text)
        self.assertNotIn("--adapter ", self.text)
        self.assertNotIn("vllm", self.text.lower())
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn("cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb", self.text)

    def test_locks_single_gpu_and_existing_strict_evaluator(self):
        self.assertIn("GPU=4", self.text)
        self.assertIn("evaluate_e1_dev.py", self.text)
        self.assertIn("expected-count 200", self.text)

    def test_does_not_reference_test_data(self):
        self.assertNotIn("/test", self.text.lower())


if __name__ == "__main__":
    unittest.main()
