# 四模型 vLLM 推理与投票脚本使用说明

## 1. 用途

脚本对同一批图片分别运行 4 个已合并的 Qwen3.5-27B 模型，并输出 4 个单模型 GOOD/BAD 结果和 1 个投票结果。最终文件不保存 categories、reasons 或模型原始生成文本。

脚本路径：

```text
/home/data/h30082292/code/sft/scripts/run_four_model_vllm_vote.py
```

使用的模型：

1. E1 checkpoint-1248
2. E2 checkpoint-1248
3. E5 checkpoint-780（Recall 优先）
4. E5 checkpoint-975（Balanced）

模型根目录：

```text
/home/data/h30082292/DATA_71/h30082292/models/qwen35_27b_four_merged_models_v1
```

## 2. 运行环境

当前 vLLM Python：

```text
/home/data/h30082292/miniconda3/envs/vllm_qwen35/bin/python
```

每个模型约 51 GiB。本方案固定使用 8 张卡：每个模型启动 2 个单卡副本，输入图片按奇偶位置分成两份，因此形成“4 个模型 × 2 个数据分片 = 8 个并行进程”。该方式优先提高大批量图片吞吐，不采用跨卡张量并行。

## 3. 输入格式

推荐使用 JSONL，每行一张图片：

```json
{"id":"sample-001","image":"/absolute/path/image.jpg"}
```

同时支持：

- ms-swift 的单图 `images` 字段；
- 纯文本文件，每行一个图片路径。

要求图片存在且 `id` 不重复。

## 4. 运行命令

先更新代码：

```bash
cd /home/data/h30082292/code/sft
git pull --ff-only
```

八卡并行：

```bash
/home/data/h30082292/miniconda3/envs/vllm_qwen35/bin/python \
  /home/data/h30082292/code/sft/scripts/run_four_model_vllm_vote.py \
  --input /path/to/images.jsonl \
  --output /path/to/five_decisions.jsonl \
  --devices 0,1,2,3,4,5,6,7
```

`--devices` 必须提供 8 个互不重复的设备编号。设备分配顺序为：E1 使用第 1～2 张卡，E2 使用第 3～4 张卡，E5-780 使用第 5～6 张卡，E5-975 使用第 7～8 张卡。输出文件必须尚未存在，避免误覆盖已有结果。

## 5. 输出格式与投票规则

每张图片输出一行：

```json
{"id":"sample-001","image":"/absolute/path/image.jpg","e1_1248":"GOOD","e2_1248":"BAD","e5_780_recall":"BAD","e5_975_balanced":"GOOD","ensemble_vote":"BAD"}
```

投票规则：4 个模型中至少 2 个判断为 BAD，`ensemble_vote` 即为 BAD；否则为 GOOD。2:2 时判为 BAD，以保持此前在 Dev 上确定的召回优先策略。

模型按相同的非 Think 提示词、`temperature=0`、`seed=42` 和 `max_new_tokens=128` 推理。脚本只提取 decision；若任一响应不能解析为 GOOD/BAD，将直接报错，不生成伪造结果。

## 6. NPU 迁移说明

NPU 适配主要替换脚本中的 `run_worker()` 推理后端。以下逻辑可以保留：

- 输入读取和图片编号；
- 4 个模型的结果对齐；
- GOOD/BAD 合法性检查；
- 至少 2 票 BAD 的投票规则；
- 最终 JSONL 输出格式。

切换推理后端或数值精度后，应先抽取一批图片与原 vLLM 结果比较决策一致率，再开始正式大批量运行。

## 7. 当前验证状态

- 4 个合并模型均已通过 Hugging Face 完整性校验；
- E5-975 合并模型已完成 vLLM 实测；
- 本脚本已通过输入、解析、对齐、投票及并行调度相关单元测试；
- 四模型整批脚本及后续 NPU 版本仍需在目标运行环境完成一次端到端验证。
