from pathlib import Path
import unittest


class VllmLoraBenchmarkPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(
            "scripts/run_vllm_lora_benchmark_preflight.sh"
        ).read_text(encoding="utf-8")

    def test_targets_balanced_e5_checkpoint_and_is_read_only(self):
        self.assertIn("checkpoint-975", self.script)
        self.assertIn("adapter_model*.safetensors", self.script)
        self.assertNotIn("pip install", self.script)
        self.assertNotIn("conda create", self.script)
        self.assertNotIn("swift export", self.script)
        self.assertNotIn("swift infer \\", self.script)

    def test_detects_connector_lora_and_merge_capacity(self):
        self.assertIn(".visual.merger.", self.script)
        self.assertIn(
            "DIRECT_VLLM_ADAPTER_MODE=UNSAFE_WITHOUT_VERIFIED_TOWER_CONNECTOR_LORA",
            self.script,
        )
        self.assertIn("MERGED_MODEL_VLLM_MODE=RECOMMENDED", self.script)
        self.assertIn("MERGE_DISK_CHECK", self.script)

    def test_reports_runtime_and_cli_contract(self):
        for value in (
            "ms-swift",
            "torch",
            "transformers",
            "vllm",
            "vllm_tensor_parallel_size",
            "vllm_gpu_memory_utilization",
            "vllm_engine_kwargs",
        ):
            self.assertIn(value, self.script)


if __name__ == "__main__":
    unittest.main()
