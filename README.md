# E1 Dev checkpoint evaluation

Qwen3.5-27B 多模态 LoRA 的 E1 Dev 生成式评测工具。该仓库只包含评测源码、单元测试与运行说明，不包含训练数据、模型权重、日志或评测结果。

## 功能

- 按固定协议在 4 张 GPU 上评测 8 个 checkpoint。
- 严格解析生成 JSON，并校验样本顺序、图片、提示词与 gold 标签。
- 输出 TP/FN/FP/TN、Recall、FPR、Accuracy、F1、JSON 合法率和逐样本错误记录。
- 根据固定 Dev 指标选择 checkpoint；Test 不参与 checkpoint、阈值或提示选择。
- 记录输入哈希、ms-swift 版本、作业状态和退出码，便于复现与审计。

## 环境

- Python 3.10+
- ms-swift 4.4.1（实际批量推理）
- pytest（可选，仅用于测试）

评测与选择脚本本身只使用 Python 标准库。

## 验证

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/evaluate_e1_dev.py scripts/select_e1_checkpoint.py scripts/run_e1_dev_checkpoints.py
```

## 使用

先阅读 [`docs/server/e1-dev-evaluation-runbook.md`](docs/server/e1-dev-evaluation-runbook.md)。服务器路径与 GPU 编号可通过命令行参数覆盖：

```bash
python scripts/run_e1_dev_checkpoints.py \
  --model /path/to/Qwen3.5-27B \
  --checkpoint-root /path/to/training-run \
  --dev /path/to/dev.jsonl \
  --output-root /path/to/evaluation-output \
  --gpus 4 5 6 7
```

建议先加 `--dry-run` 检查路径、checkpoint 完整性、GPU 分配和命令生成，再启动正式 Dev 推理。
