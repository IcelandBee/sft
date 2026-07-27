import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.validate_four_merged_models import validate


class MergeFourModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = Path("scripts/run_merge_four_models.sh").read_text(encoding="utf-8")

    def test_runner_contains_exact_four_candidates_and_target(self):
        for value in (
            "/DATA_71/h30082292/models",
            "e1-1248-merged-bf16",
            "e2-1248-merged-bf16",
            "e5-780-recall-merged-bf16",
            "e5-975-balanced-merged-bf16",
            "checkpoint-780",
            "checkpoint-975",
        ):
            self.assertIn(value, self.runner)
        self.assertEqual(self.runner.count("--entry \"${NAMES["), 4)

    def test_runner_only_merges_and_validates_hf_directories(self):
        for value in (
            "--preflight-only",
            "REQUIRED_FREE_GIB=260",
            "--merge_lora true",
            "--torch_dtype bfloat16",
            "FOUR_MODEL_MERGE: PASS",
        ):
            self.assertIn(value, self.runner)
        for forbidden in ("tar -", "pip install", "vllm_qwen35", "NPU"):
            self.assertNotIn(forbidden, self.runner)

    def test_validator_accepts_four_complete_hf_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            entries = []
            for index in range(4):
                name = f"model-{index}"
                adapter = root / f"adapter-{index}"
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text(
                    json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8"
                )
                merged = root / name
                merged.mkdir()
                (merged / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "qwen3_5",
                            "architectures": ["Qwen3_5ForConditionalGeneration"],
                        }
                    ),
                    encoding="utf-8",
                )
                for required in (
                    "generation_config.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                    "model.safetensors.index.json",
                ):
                    (merged / required).write_text("{}", encoding="utf-8")
                (merged / "model-00001-of-00001.safetensors").write_bytes(b"weights")
                entries.append((name, adapter))
            with mock.patch(
                "scripts.validate_four_merged_models.MIN_MERGED_WEIGHT_BYTES", 1
            ):
                manifest = validate(root, base, entries)
            self.assertEqual(manifest["model_count"], 4)
            self.assertEqual(manifest["format"], "Hugging Face BF16 full model")
            self.assertFalse(manifest["archive_included"])
            self.assertTrue((root / "four-merged-models-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
