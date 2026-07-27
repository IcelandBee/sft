import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import build_vllm_benchmark_requests as builder
from scripts.qwen35_benchmark_common import (
    IMAGE_PLACEHOLDER,
    MAX_IMAGE_PIXELS,
    NON_THINKING_PREFIX,
    make_raw_result,
    smart_resize,
)


class VllmNativeBenchmarkTests(unittest.TestCase):
    def test_resize_respects_factor_and_pixel_budget(self):
        for height, width in ((100, 100), (1024, 768), (8000, 6000), (56, 5600)):
            resized_h, resized_w = smart_resize(height, width)
            self.assertEqual(resized_h % 28, 0)
            self.assertEqual(resized_w % 28, 0)
            self.assertLessEqual(resized_h * resized_w, MAX_IMAGE_PIXELS)

    def test_raw_result_matches_existing_evaluator_contract(self):
        request = {
            "image": "/tmp/a.jpg",
            "label": '{"decision":"GOOD","categories":[],"reasons":[]}',
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "dataset": "/tmp/test.jsonl",
        }
        row = make_raw_result(request, '{"decision":"GOOD","categories":[],"reasons":[]}')
        self.assertTrue(row["response"].startswith(NON_THINKING_PREFIX))
        self.assertEqual(row["messages"][-1]["content"], row["response"])
        self.assertEqual(row["images"], [{"bytes": None, "path": "/tmp/a.jpg"}])

    def test_request_builder_freezes_qwen35_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            dataset = root / "test.jsonl"
            rows = []
            for index in range(241):
                decision = "BAD" if index < 66 else "GOOD"
                payload = (
                    {"decision": "BAD", "categories": ["手部异常"], "reasons": ["手指异常"]}
                    if decision == "BAD"
                    else {"decision": "GOOD", "categories": [], "reasons": []}
                )
                rows.append(
                    {
                        "images": [str(root / f"{index}.jpg")],
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "<image>\nquestion"},
                            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                        ],
                    }
                )
                (root / f"{index}.jpg").write_bytes(b"image")
            source = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            dataset.write_text(source, encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            output = root / "requests.jsonl"
            manifest = root / "manifest.json"
            with mock.patch.object(builder, "EXPECTED_SHA256", digest):
                result = builder.build(dataset, output, manifest)
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn(IMAGE_PLACEHOLDER, first["prompt"])
            self.assertTrue(first["prompt"].endswith(NON_THINKING_PREFIX))
            self.assertEqual(result["rows"], 241)

    def test_runner_uses_untouched_vllm_env_and_same_gpu_sequentially(self):
        script = Path("scripts/run_vllm_lora_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("/envs/vllm_qwen35/bin/python", script)
        self.assertNotIn("pip install", script)
        self.assertNotIn("/envs/vllm_qwen35/bin/swift", script)
        self.assertLess(script.index("run_transformers_merged_benchmark.py"), script.index("run_vllm_merged_benchmark.py"))
        self.assertIn("CUDA_VISIBLE_DEVICES=\"$GPU\"", script)
        self.assertIn("e5-975-balanced/evaluation/parsed.jsonl", script)

    def test_merge_uses_training_env_and_explicit_adapter(self):
        script = Path("scripts/run_merge_e5_975_for_vllm.sh").read_text(encoding="utf-8")
        self.assertIn("/envs/qwen35_27b", script)
        self.assertIn('SWIFT="$TRAIN_ENV/bin/swift"', script)
        self.assertIn("checkpoint-975", script)
        self.assertIn("--merge_lora true", script)
        self.assertNotIn("/envs/vllm_qwen35", script)


if __name__ == "__main__":
    unittest.main()
