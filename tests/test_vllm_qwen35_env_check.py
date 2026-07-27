from pathlib import Path
import unittest


class VllmQwen35EnvironmentCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path("scripts/run_vllm_qwen35_env_check.sh").read_text(
            encoding="utf-8"
        )

    def test_targets_dedicated_environment_and_is_read_only(self):
        self.assertIn("/envs/vllm_qwen35", self.script)
        self.assertNotIn("pip install", self.script)
        self.assertNotIn("conda install", self.script)
        self.assertNotIn("swift infer \\", self.script)
        self.assertNotIn("swift export", self.script)

    def test_checks_required_runtime_packages_and_imports(self):
        for value in (
            '"vllm"',
            '"ms-swift"',
            '"torch"',
            '"transformers"',
            "from vllm import LLM",
            "import swift",
            "pip check",
        ):
            self.assertIn(value, self.script)

    def test_reports_swift_vllm_contract_and_gpu_state(self):
        self.assertIn("vllm_tensor_parallel_size", self.script)
        self.assertIn("vllm_gpu_memory_utilization", self.script)
        self.assertIn("nvidia-smi", self.script)


if __name__ == "__main__":
    unittest.main()
