import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_e4_dev_checkpoints.sh"
)


class E4DevCheckpointScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_locks_e4_run_and_all_eight_checkpoints(self):
        self.assertIn(
            "e4_crop_aux_aligner_r16_s2080_v1/v0-20260723-142528",
            self.text,
        )
        self.assertIn(
            "--steps 260 520 780 1040 1300 1560 1820 2080",
            self.text,
        )
        self.assertIn("e4_crop_aux_aligner_8ckpt_v1", self.text)

    def test_uses_corrected_dev_and_frozen_binary_protocol(self):
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            self.text,
        )
        self.assertIn("--expected-good 142", self.text)
        self.assertIn("--expected-bad 58", self.text)
        self.assertIn("run_e1_dev_checkpoints.py", self.text)
        self.assertNotIn("test.jsonl", self.text.lower())

    def test_supports_non_gpu_dry_run_without_overwriting_results(self):
        self.assertIn("--dry-run", self.text)
        self.assertNotIn("rm -", self.text)


if __name__ == "__main__":
    unittest.main()
