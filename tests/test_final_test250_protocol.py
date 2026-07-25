from pathlib import Path
import unittest


class FinalTest250ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.builder = (root / "scripts/build_final_test250_dataset.py").read_text(
            encoding="utf-8"
        )
        cls.build_wrapper = (
            root / "scripts/run_build_final_test250_dataset.sh"
        ).read_text(encoding="utf-8")
        cls.runner = (
            root / "scripts/run_final_test250_lora_checkpoints.sh"
        ).read_text(encoding="utf-8")

    def test_builder_locks_workbook_uploaded_version_and_binary_counts(self):
        self.assertIn("20260724-v1.xlsx", self.build_wrapper)
        self.assertIn(
            "884601f97be529420529c87798bcfedca63c90d0b516da525c42806bd03e38b6",
            self.build_wrapper,
        )
        self.assertIn('labels != {"GOOD": 186, "BAD": 64}', self.builder)
        self.assertIn("target_prefix", self.builder)
        self.assertIn("checkpoint_selection_forbidden", self.builder)

    def test_runner_freezes_four_preselected_lora_candidates(self):
        self.assertIn("e1-1248", self.runner)
        self.assertIn("e2-1248", self.runner)
        self.assertIn("e5-780-recall", self.runner)
        self.assertIn("e5-975-balanced", self.runner)
        self.assertIn("checkpoint-780", self.runner)
        self.assertIn("checkpoint-975", self.runner)
        self.assertIn("--expected-count 250", self.runner)

    def test_runner_uses_fixed_protocol_and_never_selects_by_test(self):
        self.assertIn("--temperature 0", self.runner)
        self.assertIn("--add_non_thinking_prefix true", self.runner)
        self.assertIn("checkpoint_selection_forbidden", self.runner)
        self.assertNotIn("selected_step", self.runner)
        self.assertNotIn("run_selection", self.runner)


if __name__ == "__main__":
    unittest.main()
