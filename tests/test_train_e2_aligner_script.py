import re
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_e2_aligner_r16_e2_v1.sh"
)


class E2TrainingScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, flag):
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", self.text)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_keeps_e1_data_and_core_optimizer_protocol(self):
        self.assertIn("e1_broad_clean_json_v1", self.text)
        self.assertEqual(self.value("--lora_rank"), "16")
        self.assertEqual(self.value("--lora_alpha"), "32")
        self.assertEqual(self.value("--learning_rate"), "5e-5")
        self.assertEqual(self.value("--gradient_accumulation_steps"), "4")
        self.assertEqual(self.value("--max_length"), "2048")

    def test_only_substantive_model_change_adds_aligner_lora(self):
        self.assertEqual(self.value("--freeze_llm"), "false")
        self.assertEqual(self.value("--freeze_vit"), "true")
        self.assertEqual(self.value("--freeze_aligner"), "false")
        self.assertEqual(self.value("--target_modules"), "all-linear")

    def test_caps_training_at_two_epochs_and_saves_every_eval(self):
        self.assertEqual(self.value("--num_train_epochs"), "2")
        self.assertEqual(self.value("--eval_steps"), "156")
        self.assertEqual(self.value("--save_steps"), "156")
        self.assertEqual(self.value("--save_total_limit"), "8")

    def test_has_non_mutating_preflight_and_distinct_output(self):
        self.assertIn('--preflight-only', self.text)
        self.assertIn('E2_PREFLIGHT_CHECK: PASS', self.text)
        self.assertIn('e2_broad_clean_aligner_r16_e2_v1', self.text)
        self.assertNotIn('e1_broad_clean_r16_e4_v1/v0-', self.text)


if __name__ == "__main__":
    unittest.main()
