from pathlib import Path
import unittest


class FinalTest250BaseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(
            "scripts/run_base_constrained_binary_final_test250.sh"
        ).read_text(encoding="utf-8")

    def test_locks_conditionally_readjudicated_dataset(self):
        self.assertIn(
            "c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f",
            self.script,
        )
        self.assertIn("--expected-count 241", self.script)
        self.assertIn("--expected-good 175", self.script)
        self.assertIn("--expected-bad 66", self.script)
        self.assertIn("--no-test-untouched", self.script)

    def test_runs_base_only_and_builds_lora_comparison(self):
        self.assertIn("base-constrained-binary", self.script)
        self.assertIn("final-model-comparison.json", self.script)
        self.assertIn('readjudication["model_results"]', self.script)
        self.assertNotIn("--adapters", self.script)
        self.assertNotIn("--adapter ", self.script)


if __name__ == "__main__":
    unittest.main()
