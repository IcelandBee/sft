import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_e1_e2_dev_audit.sh"


class BoundaryAuditScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_selected_e1_and_e2_checkpoints(self):
        self.assertIn("e1_broad_clean_8ckpt_v1/checkpoint-1248", self.text)
        self.assertIn("e2_broad_clean_aligner_8ckpt_v1/checkpoint-1248", self.text)

    def test_locks_observed_comparison_counts(self):
        for value in ("143", "37", "10", "20", "57"):
            self.assertIn(value, self.text)

    def test_does_not_reference_test_data(self):
        self.assertNotIn("test.jsonl", self.text.lower())
        self.assertIn("dev.jsonl", self.text)


if __name__ == "__main__":
    unittest.main()
