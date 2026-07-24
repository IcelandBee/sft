import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_e5_dev_checkpoints.sh"
)


class E5DevCheckpointScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_locks_e5_run_and_all_eight_checkpoints(self):
        self.assertIn(
            "e5_crop_aux20_aligner_r16_s1560_v1/v0-20260723-210158",
            self.text,
        )
        self.assertIn(
            "--steps 195 390 585 780 975 1170 1365 1560",
            self.text,
        )
        self.assertIn("e5_crop_aux20_aligner_8ckpt_v1", self.text)

    def test_uses_corrected_dev_and_existing_binary_protocol(self):
        self.assertIn("dev_adjudicated_v1/dev.jsonl", self.text)
        self.assertIn(
            "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb",
            self.text,
        )
        self.assertIn("--expected-good 142", self.text)
        self.assertIn("--expected-bad 58", self.text)
        self.assertIn("run_e1_dev_checkpoints.py", self.text)
        self.assertNotIn("test.jsonl", self.text.lower())

    def test_supports_non_destructive_dry_run(self):
        self.assertIn("--dry-run", self.text)
        self.assertNotIn("rm -", self.text)


if __name__ == "__main__":
    unittest.main()
