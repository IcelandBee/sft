import re
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_e3_vit_aligner_r16_e2_v1.sh"
)


class E3TrainingScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, flag):
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", self.text)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_uses_same_train_and_frozen_adjudicated_dev(self):
        self.assertIn("e1_broad_clean_json_v1/train.jsonl", self.text)
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            self.text,
        )
        self.assertIn("training_forbidden", self.text)

    def test_single_model_change_extends_lora_to_vit(self):
        self.assertEqual(self.value("--freeze_llm"), "false")
        self.assertEqual(self.value("--freeze_vit"), "false")
        self.assertEqual(self.value("--freeze_aligner"), "false")
        self.assertEqual(self.value("--target_modules"), "all-linear")

    def test_keeps_e2_optimizer_and_two_epoch_protocol(self):
        self.assertEqual(self.value("--lora_rank"), "16")
        self.assertEqual(self.value("--lora_alpha"), "32")
        self.assertEqual(self.value("--learning_rate"), "5e-5")
        self.assertEqual(self.value("--num_train_epochs"), "2")
        self.assertEqual(self.value("--eval_steps"), "156")
        self.assertEqual(self.value("--save_steps"), "156")
        self.assertEqual(self.value("--save_total_limit"), "8")

    def test_has_safe_preflight_and_distinct_output(self):
        self.assertIn("--preflight-only", self.text)
        self.assertIn("E3_PREFLIGHT_CHECK: PASS", self.text)
        self.assertIn("e3_broad_clean_vit_aligner_r16_e2_v1", self.text)
        self.assertNotIn("test.jsonl", self.text.lower())


if __name__ == "__main__":
    unittest.main()
