import re
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_e4_crop_aux_aligner_r16_s1248_v1.sh"
)


class E4CropAuxTrainingScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, flag):
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", self.text)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_uses_frozen_e4_data_and_hashes(self):
        self.assertIn("e4_crop_aux_json_v1", self.text)
        self.assertIn(
            "23b61202f3d92b87847c13f6c3df3597db93b6390ef3e9896ecf9be886e34a08",
            self.text,
        )
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            self.text,
        )
        self.assertIn('{"T1_FULL": 9978, "T2_BAD": 3326, "T3_GOOD": 3326}', self.text)
        self.assertNotIn("test.jsonl", self.text.lower())

    def test_matches_e2_model_and_optimizer_scope(self):
        self.assertEqual(self.value("--freeze_llm"), "false")
        self.assertEqual(self.value("--freeze_vit"), "true")
        self.assertEqual(self.value("--freeze_aligner"), "false")
        self.assertEqual(self.value("--target_modules"), "all-linear")
        self.assertEqual(self.value("--lora_rank"), "16")
        self.assertEqual(self.value("--lora_alpha"), "32")
        self.assertEqual(self.value("--lora_dropout"), "0.05")
        self.assertEqual(self.value("--learning_rate"), "5e-5")
        self.assertEqual(self.value("--weight_decay"), "0.1")

    def test_locks_equal_optimization_budget_and_longer_sequence(self):
        self.assertEqual(self.value("--max_steps"), "1248")
        self.assertEqual(self.value("--max_length"), "3072")
        self.assertEqual(self.value("--gradient_accumulation_steps"), "4")
        self.assertEqual(self.value("--eval_steps"), "156")
        self.assertEqual(self.value("--save_steps"), "156")
        self.assertEqual(self.value("--save_total_limit"), "8")
        self.assertNotIn("--num_train_epochs", self.text)

    def test_has_non_mutating_preflight_and_distinct_output(self):
        self.assertIn("--preflight-only", self.text)
        self.assertIn("E4_PREFLIGHT_CHECK: PASS", self.text)
        self.assertIn("e4_crop_aux_aligner_r16_s1248_v1", self.text)
        self.assertIn("if [[ -e \"$OUT\" ]]", self.text)


if __name__ == "__main__":
    unittest.main()
