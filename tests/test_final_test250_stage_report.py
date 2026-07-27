from pathlib import Path
import unittest


class FinalTest250StageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = Path(
            "docs/experiments/2026-07-27-final-test250-stage-summary.md"
        ).read_text(encoding="utf-8")

    def test_records_dataset_review_contract(self):
        for value in (
            "GOOD 186、BAD 64",
            "GOOD 175、BAD 66",
            "重点复核 | 73",
            "排除“不确定” | 9",
            "虚拟或经过编辑",
            "带水印",
            "c59dc4dbd3752fc124a009d48bdbfcdf6f20aeb402a0db3bb41c8ce4c1fcda0f",
        ):
            self.assertIn(value, self.report)

    def test_records_single_model_and_ensemble_metrics(self):
        for value in (
            "Base Qwen3.5-27B（格式受控）",
            "E1 checkpoint-1248",
            "E2 checkpoint-1248",
            "E5 checkpoint-780",
            "E5 checkpoint-975",
            "74.24%",
            "83.82%",
            "72.73%",
            "13.71%",
            "69.57%",
        ):
            self.assertIn(value, self.report)

    def test_records_merge_and_speed_evidence(self):
        for value in (
            "241/241（100%）",
            "238/241（98.76%）",
            "3.64×",
            "1.581 秒",
            "0.434 秒",
            "301.04 秒",
            "236 张",
        ):
            self.assertIn(value, self.report)

    def test_records_capacity_plan_and_limitations(self):
        for value in (
            "1 万",
            "5 万",
            "10 万",
            "约 1.29 小时",
            "约 6.11 小时",
            "约 12.14 小时",
            "随机抽取 500 张",
            "至少 99%",
            "每 1,000 张",
            "探索性结果",
        ):
            self.assertIn(value, self.report)


if __name__ == "__main__":
    unittest.main()
