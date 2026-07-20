import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_e1_dev_checkpoints import (
    RunnerError,
    assign_jobs,
    build_manifest,
    build_swift_command,
    validate_dev_file,
    write_manifest,
)
from scripts.select_e1_checkpoint import EXPECTED_STEPS


class CommandTests(unittest.TestCase):
    def test_build_command_locks_inference_protocol(self):
        command = build_swift_command(
            Path("/model"), Path("/run/checkpoint-624"), Path("/data/dev.jsonl"),
            Path("/eval/checkpoint-624/raw-result.jsonl"),
        )
        pairs = dict(zip(command[2::2], command[3::2]))

        self.assertEqual(command[:2], ["swift", "infer"])
        self.assertEqual(pairs["--infer_backend"], "transformers")
        self.assertEqual(pairs["--torch_dtype"], "bfloat16")
        self.assertEqual(pairs["--attn_impl"], "flash_attention_2")
        self.assertEqual(pairs["--temperature"], "0")
        self.assertEqual(pairs["--stream"], "false")
        self.assertEqual(pairs["--max_new_tokens"], "128")
        self.assertEqual(pairs["--max_batch_size"], "1")
        self.assertEqual(pairs["--val_dataset_shuffle"], "false")
        self.assertEqual(pairs["--strict"], "true")
        self.assertEqual(pairs["--seed"], "42")
        self.assertEqual(pairs["--data_seed"], "42")

    def test_commands_only_vary_in_adapter_and_result(self):
        first = build_swift_command(
            Path("/model"), Path("/run/checkpoint-312"), Path("/data/dev.jsonl"),
            Path("/eval/checkpoint-312/raw-result.jsonl"),
        )
        second = build_swift_command(
            Path("/model"), Path("/run/checkpoint-624"), Path("/data/dev.jsonl"),
            Path("/eval/checkpoint-624/raw-result.jsonl"),
        )
        first_pairs = dict(zip(first[2::2], first[3::2]))
        second_pairs = dict(zip(second[2::2], second[3::2]))
        differing = {
            key for key in first_pairs if first_pairs[key] != second_pairs[key]
        }
        self.assertEqual(differing, {"--adapters", "--result_path"})


class SchedulingAndManifestTests(unittest.TestCase):
    def test_assigns_two_checkpoints_to_each_gpu(self):
        jobs = assign_jobs(EXPECTED_STEPS, (4, 5, 6, 7))

        self.assertEqual([step for step, _ in jobs], list(EXPECTED_STEPS))
        self.assertEqual(Counter(gpu for _, gpu in jobs), {4: 2, 5: 2, 6: 2, 7: 2})
        self.assertEqual(jobs[:4], [(312, 4), (624, 5), (936, 6), (1248, 7)])

    def test_manifest_contains_eight_auditable_jobs_and_write_is_exclusive(self):
        manifest = build_manifest(
            model=Path("/model"),
            checkpoint_root=Path("/run"),
            dev=Path("/data/dev.jsonl"),
            output_root=Path("/eval"),
            steps=EXPECTED_STEPS,
            gpus=(4, 5, 6, 7),
            dev_sha256="a" * 64,
        )

        self.assertEqual(len(manifest["jobs"]), 8)
        self.assertEqual(Counter(job["gpu"] for job in manifest["jobs"]), {4: 2, 5: 2, 6: 2, 7: 2})
        self.assertEqual(manifest["dev_sha256"], "a" * 64)
        self.assertEqual(len(manifest["prompt_sha256"]["system"]), 64)
        self.assertEqual(len(manifest["prompt_sha256"]["user"]), 64)
        self.assertTrue(all(job["command"][:2] == ["swift", "infer"] for job in manifest["jobs"]))
        self.assertTrue(
            all(job["dev"] == str(Path("/data/dev.jsonl")) for job in manifest["jobs"])
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
            with self.assertRaisesRegex(RunnerError, "already exists"):
                write_manifest(path, manifest)


class DevValidationTests(unittest.TestCase):
    def test_validates_rows_labels_prompts_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            rows = [
                self.make_dev_row(image_a, "GOOD"),
                self.make_dev_row(image_b, "BAD"),
            ]
            path = root / "dev.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = validate_dev_file(
                path, expected_count=2, expected_good=1, expected_bad=1
            )

            self.assertEqual(summary["counts"], {"GOOD": 1, "BAD": 1, "total": 2})
            self.assertEqual(len(summary["sha256"]), 64)

            image_b.unlink()
            with self.assertRaisesRegex(RunnerError, "image does not exist"):
                validate_dev_file(path, expected_count=2, expected_good=1, expected_bad=1)

    @staticmethod
    def make_dev_row(image, decision):
        value = {
            "decision": decision,
            "categories": [] if decision == "GOOD" else ["手部异常"],
            "reasons": [] if decision == "GOOD" else ["手指畸形"],
        }
        return {
            "images": [str(image.resolve())],
            "messages": [
                {"role": "system", "content": "你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。严格只输出指定JSON，不要添加分析、解释或Markdown。"},
                {"role": "user", "content": "<image>\n检查这张图片。输出decision、categories和reasons。decision只能是GOOD或BAD。"},
                {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
            ],
        }


if __name__ == "__main__":
    unittest.main()
