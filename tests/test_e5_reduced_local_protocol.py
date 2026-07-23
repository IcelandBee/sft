import re
import unittest
from pathlib import Path


class E5ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.builder = (root / "scripts/build_e5_reduced_local_dataset.py").read_text(
            encoding="utf-8"
        )
        cls.wrapper = (
            root / "scripts/run_build_e5_reduced_local_dataset.sh"
        ).read_text(encoding="utf-8")
        cls.training = (
            root / "scripts/train_e5_crop_aux20_aligner_r16_s1560_v1.sh"
        ).read_text(encoding="utf-8")

    def value(self, flag):
        match = re.search(rf"\s{re.escape(flag)}\s+([^\s\\]+)", self.training)
        self.assertIsNotNone(match, flag)
        return match.group(1).strip('"')

    def test_reuses_e4_crops_and_locks_80_20_mix(self):
        self.assertIn("e4_crop_aux_json_v1", self.wrapper)
        self.assertIn("e5_crop_aux20_json_v1", self.wrapper)
        self.assertIn("--local-pairs 1247", self.wrapper)
        self.assertIn('"T1_FULL": 9978', self.builder)
        self.assertIn('"T2_BAD": local_pairs', self.builder)
        self.assertIn('"T3_GOOD": local_pairs', self.builder)
        self.assertNotIn("test.jsonl", (self.builder + self.wrapper).lower())

    def test_keeps_e2_scope_with_two_effect_oriented_epochs(self):
        self.assertEqual(self.value("--freeze_llm"), "false")
        self.assertEqual(self.value("--freeze_vit"), "true")
        self.assertEqual(self.value("--freeze_aligner"), "false")
        self.assertEqual(self.value("--lora_rank"), "16")
        self.assertEqual(self.value("--learning_rate"), "5e-5")
        self.assertEqual(self.value("--max_steps"), "1560")
        self.assertEqual(self.value("--eval_steps"), "195")
        self.assertEqual(self.value("--save_steps"), "195")
        self.assertEqual(self.value("--max_length"), "3072")

    def test_has_safe_preflight_and_distinct_output(self):
        self.assertIn("--preflight-only", self.training)
        self.assertIn("E5_PREFLIGHT_CHECK: PASS", self.training)
        self.assertIn("e5_crop_aux20_aligner_r16_s1560_v1", self.training)
        self.assertIn("if [[ -e \"$OUT\" ]]", self.training)
        self.assertNotIn("test.jsonl", self.training.lower())


if __name__ == "__main__":
    unittest.main()
