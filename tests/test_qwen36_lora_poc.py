import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_qwen36_lora_poc_dataset import (
    STRATA_ORDER,
    build_interleaved_rows,
)
from scripts.validate_qwen36_lora_poc import load_gpu_peaks


def row(image_count: int, decision: str) -> dict:
    payload = {"decision": decision, "categories": [], "reasons": []}
    return {
        "images": [f"image-{index}.png" for index in range(image_count)],
        "messages": [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
    }


class Qwen36LoraPocDatasetTests(unittest.TestCase):
    def test_interleaves_one_row_from_each_stratum_per_global_batch(self):
        rows = []
        for image_count, decision in ((1, "GOOD"), (1, "BAD"), (2, "BAD"), (2, "GOOD")):
            rows.extend(row(image_count, decision) for _ in range(9))

        poc_rows, manifest = build_interleaved_rows(rows)

        self.assertEqual(len(poc_rows), 20)
        for offset in range(0, 20, 4):
            self.assertEqual(
                tuple(item["stratum"] for item in manifest[offset : offset + 4]),
                STRATA_ORDER,
            )


class Qwen36LoraPocValidationTests(unittest.TestCase):
    def test_reads_four_gpu_peak_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                "timestamp,gpu,memory_used_mib,memory_free_mib\n"
                "1,4,10,80000\n1,5,20,80000\n1,6,30,80000\n1,7,40,80000\n"
                "2,4,50000,30000\n2,5,51000,30000\n2,6,52000,30000\n2,7,53000,30000\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_gpu_peaks(path), {4: 50000, 5: 51000, 6: 52000, 7: 53000}
            )

    def test_wrapper_locks_two_step_four_gpu_lora_scope(self):
        script = Path("scripts/run_qwen36_lora_poc.sh").read_text(encoding="utf-8")
        self.assertIn("/envs/qwen36_27b", script)
        self.assertIn("Qwen3.6-27B", script)
        self.assertIn("e5_crop_aux20_json_v1/train.jsonl", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=4,5,6,7", script)
        self.assertIn("export NPROC_PER_NODE=4", script)
        self.assertIn("--deepspeed zero2", script)
        self.assertIn("--attn_impl flash_attention_2", script)
        self.assertIn("--target_modules all-linear", script)
        self.assertIn("--freeze_llm false", script)
        self.assertIn("--freeze_vit true", script)
        self.assertIn("--freeze_aligner false", script)
        self.assertIn("--lora_rank 16", script)
        self.assertIn("--max_steps 2", script)
        self.assertIn("--max_length 3072", script)
        self.assertNotIn("dev.jsonl", script)
        self.assertNotIn("test.jsonl", script.lower())


if __name__ == "__main__":
    unittest.main()
