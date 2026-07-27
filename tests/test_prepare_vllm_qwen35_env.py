from pathlib import Path
import unittest


class PrepareVllmQwen35EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path("scripts/prepare_vllm_qwen35_env.sh").read_text(
            encoding="utf-8"
        )

    def test_only_targets_dedicated_environment(self):
        self.assertIn("/envs/vllm_qwen35", self.script)
        self.assertNotIn("/envs/qwen35_27b", self.script)

    def test_pins_protocol_packages(self):
        self.assertIn('"ms-swift==4.4.1"', self.script)
        self.assertIn('"peft==0.19.1"', self.script)
        self.assertIn('"qwen-vl-utils==0.0.14"', self.script)
        self.assertIn("--upgrade-strategy only-if-needed", self.script)

    def test_does_not_explicitly_replace_core_runtime(self):
        self.assertNotIn('"torch==', self.script)
        self.assertNotIn('"vllm==', self.script)
        self.assertNotIn('"transformers==', self.script)
        self.assertNotIn("conda install", self.script)

    def test_validates_imports_and_dependency_consistency(self):
        for value in ("import torch", "import vllm", "import swift", "import peft"):
            self.assertIn(value, self.script)
        self.assertIn('"$PYTHON" -m pip check', self.script)
        self.assertIn("VLLM_QWEN35_ENV_PREPARE: PASS", self.script)


if __name__ == "__main__":
    unittest.main()
