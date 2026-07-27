import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_four_model_vllm_vote import (
    MODEL_NAMES,
    combine_results,
    load_input,
    parse_decision,
)


class FourModelVllmVoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path("scripts/run_four_model_vllm_vote.py").read_text(
            encoding="utf-8"
        )

    def test_script_has_four_merged_models_and_sequential_tp8_execution(self):
        for value in (
            "qwen35-27b-e1-1248-merged-bf16",
            "qwen35-27b-e2-1248-merged-bf16",
            "qwen35-27b-e5-780-recall-merged-bf16",
            "qwen35-27b-e5-975-balanced-merged-bf16",
            "--devices",
            "0,1,2,3,4,5,6,7",
            "TENSOR_PARALLEL_SIZE = 8",
            "tensor_parallel_size=TENSOR_PARALLEL_SIZE",
            "for name, model in models:",
            "CUDA_VISIBLE_DEVICES",
            "from vllm import LLM, SamplingParams",
        ):
            self.assertIn(value, self.script)
        self.assertNotIn("ThreadPoolExecutor", self.script)
        self.assertNotIn("request_shards", self.script)

    def test_parse_decision_accepts_json_and_rejects_missing_decision(self):
        self.assertEqual(parse_decision('{"decision":"BAD","categories":[]}'), "BAD")
        self.assertEqual(parse_decision("GOOD"), "GOOD")
        with self.assertRaises(ValueError):
            parse_decision('{"categories":[]}')

    def test_two_bad_votes_produce_bad_and_five_fields_are_binary(self):
        requests = [{"id": "a", "image": "/a.jpg", "prompt": "p"}]
        decisions = ("BAD", "GOOD", "BAD", "GOOD")
        results = {
            name: [{"id": "a", "decision": decision}]
            for name, decision in zip(MODEL_NAMES, decisions)
        }
        row = combine_results(requests, results)[0]
        self.assertEqual(row["ensemble_vote"], "BAD")
        self.assertEqual(
            set(row), {"id", "image", *MODEL_NAMES, "ensemble_vote"}
        )
        self.assertTrue(
            all(row[key] in {"GOOD", "BAD"} for key in (*MODEL_NAMES, "ensemble_vote"))
        )

    def test_load_input_supports_plain_paths_and_ms_swift_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "b.jpg"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            source = root / "input.jsonl"
            source.write_text(
                f"{first}\n"
                + json.dumps({"id": "custom", "images": [str(second)]})
                + "\n",
                encoding="utf-8",
            )
            rows = load_input(source)
            self.assertEqual([row["id"] for row in rows], ["1", "custom"])
            self.assertTrue(all("<|image_pad|>" in row["prompt"] for row in rows))

if __name__ == "__main__":
    unittest.main()
