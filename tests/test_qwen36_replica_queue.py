import re
from pathlib import Path
import unittest

from scripts.validate_qwen36_replica_data import CONTRACTS


TRAIN = Path("scripts/train_qwen36_replica.sh").read_text(encoding="utf-8")
QUEUE = Path("scripts/run_qwen36_replica_queue.sh").read_text(encoding="utf-8")


class Qwen36ReplicaTrainingTests(unittest.TestCase):
    def test_locks_all_five_original_data_contracts(self):
        self.assertEqual(CONTRACTS["E1"]["rows"], 9978)
        self.assertEqual(CONTRACTS["E2"], CONTRACTS["E1"])
        self.assertEqual(CONTRACTS["E3"], CONTRACTS["E1"])
        self.assertEqual(CONTRACTS["E4"]["rows"], 16630)
        self.assertEqual(CONTRACTS["E5"]["rows"], 12472)

    def test_uses_qwen36_isolated_environment_and_common_optimizer(self):
        self.assertIn("/envs/qwen36_27b", TRAIN)
        self.assertIn("Qwen3.6-27B", TRAIN)
        for value in (
            "--deepspeed zero2",
            "--attn_impl flash_attention_2",
            "--lora_rank 16",
            "--lora_alpha 32",
            "--lora_dropout 0.05",
            "--learning_rate 5e-5",
            "--gradient_accumulation_steps 4",
        ):
            self.assertIn(value, TRAIN)
        self.assertNotIn("test.jsonl", TRAIN.lower())

    def test_replicates_e1_to_e5_scope_steps_and_checkpoints(self):
        expected = {
            "E1": (2496, 2048, 156, 312, "true", "true"),
            "E2": (1248, 2048, 156, 156, "true", "false"),
            "E3": (1248, 2048, 156, 156, "false", "false"),
            "E4": (2080, 3072, 260, 260, "true", "false"),
            "E5": (1560, 3072, 195, 195, "true", "false"),
        }
        for experiment, values in expected.items():
            block = re.search(
                rf"{experiment}\).*?(?=\n\s*E[1-5]\)|\n\s*esac)", TRAIN, re.S
            )
            self.assertIsNotNone(block, experiment)
            text = block.group(0)
            steps, length, eval_steps, save_steps, vit, aligner = values
            self.assertIn(f"MAX_STEPS={steps}", text)
            self.assertIn(f"MAX_LENGTH={length}", text)
            self.assertIn(f"EVAL_STEPS={eval_steps}", text)
            self.assertIn(f"SAVE_STEPS={save_steps}", text)
            self.assertIn(f"FREEZE_VIT={vit}", text)
            self.assertIn(f"FREEZE_ALIGNER={aligner}", text)


class Qwen36ReplicaQueueTests(unittest.TestCase):
    def test_prioritizes_best_and_key_ablation_before_negative_experiments(self):
        self.assertIn("DEFAULT_ORDER=(E5 E2 E1 E3 E4)", QUEUE)
        self.assertIn("run_stage BASE_DEV", QUEUE)
        self.assertIn("--preflight-only", QUEUE)
        self.assertIn("status.tsv", QUEUE)
        self.assertNotIn(" &\n", QUEUE)


if __name__ == "__main__":
    unittest.main()
