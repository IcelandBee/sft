# Qwen3.6-27B 多模态 LoRA 复刻实验阶段总结

阶段完成于 2026-07-29。本轮在不修改既有 Qwen3.5 环境、模型和 LoRA 资产的前提下，将 Qwen3.5 阶段的 E1～E5 实验迁移到 Qwen3.6-27B，完成隔离环境、processor/推理/训练兼容性验证、5 组正式训练，以及固定 Dev 的 40 个 checkpoint 图片级评测。当前已经在固定 Dev 上锁定 E5 checkpoint-780；复核 Test N=241 尚未运行。

## 1. 环境与兼容性验证

- 基座：`/home/data/h30082292/DATA_71/public/models/Qwen3.6-27B`，共 15 个权重分片；权重索引引用完整、无缺失分片。模型配置中的 `model_type=qwen3_5`、`Qwen3_5ForConditionalGeneration` 是该模型当前官方文件的实际标记，不是误用 Qwen3.5 权重。
- 隔离环境：`/home/data/h30082292/miniconda3/envs/qwen36_27b`。核心版本为 Python 3.12.13、PyTorch 2.8.0、Transformers 5.12.1、ms-swift 4.4.1、PEFT 0.19.1、DeepSpeed 0.18.9、FlashAttention 2.8.3。原 `qwen35_27b` 环境未修改。
- Processor PoC 使用 Qwen3.6 自带文件加载为 `Qwen3VLProcessor` 和 `Qwen3_5Template`。从 E5 分层抽取 20 行、共 30 个图片输入，覆盖单图 T1 和双图 T2/T3；最大长度 1,789 token，低于正式训练的 `max_length=3072`，没有截断风险。
- 20 行自然生成推理通过 BF16、FlashAttention 2、non-thinking 和严格 JSON 检查；输出 20/20 可解析且没有 think 内容。
- 4×A100 的 2-step LoRA PoC 成功完成 ZeRO-2 前向、反向、保存和重载检查。可训练参数 117.03M，占 27.47B 参数的 0.426%；adapter 张量覆盖 LLM 992、aligner 4、ViT 0，符合“LLM+aligner LoRA、冻结 ViT”的设计。单卡峰值约 58～61 GiB。

服务器内核为 5.4，低于 PyTorch 建议的 5.5，但 PoC 和随后全部长训练均正常结束，未出现 NCCL 挂起、OOM 或 NaN。

## 2. 数据与统一训练协议

数据构造、broad-clean 规则、crop 生成方式和 Prompt 语义沿用 Qwen3.5 阶段，本文不再重复展开。所有数据均重新通过 Qwen3.6 processor tokenize，没有复用 Qwen3.5 token 缓存。

- E1～E3：broad-clean 单图 Train 9,978 行，GOOD 6,074、BAD 3,904。
- E4：T1/T2/T3 共 16,630 行，其中单图 9,978、双图 6,652。
- E5：降低局部辅助占比后的 12,472 行，其中 T1 9,978、T2 1,247、T3 1,247；GOOD 7,321、BAD 5,151。
- 固定 Dev：200 张，GOOD 142、BAD 58；SHA256 为 `cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb`。Dev 不进入训练，Train/Dev 无图片重叠。

统一配置为 4×A100 80GB（GPU4～7）、BF16、FlashAttention 2、DeepSpeed ZeRO-2、gradient checkpointing；LoRA `r=16`、`alpha=32`、`dropout=0.05`、`target_modules=all-linear`；学习率 `5e-5`、cosine、warmup 0.05、weight decay 0.1；单卡 batch 1、梯度累积 4、全局 batch 16；图片 token 上限 1,024；随机种子 42。图片级 `decision` 是核心目标，`categories` 和 `reasons` 只作为辅助监督。

| 实验 | 数据/目的 | LoRA 范围 | max_length | steps | checkpoint |
|---|---|---|---:|---:|---|
| E1 | broad-clean；复刻仅语言侧适配 | LLM；冻结 ViT/aligner | 2,048 | 2,496（4 epoch） | 每 312 step，共 8 个 |
| E2 | broad-clean；验证 aligner 适配 | LLM+aligner；冻结 ViT | 2,048 | 1,248（2 epoch） | 每 156 step，共 8 个 |
| E3 | broad-clean；验证 ViT 适配 | LLM+aligner+ViT | 2,048 | 1,248（2 epoch） | 每 156 step，共 8 个 |
| E4 | 40% 局部辅助任务 | LLM+aligner；冻结 ViT | 3,072 | 2,080（2 epoch） | 每 260 step，共 8 个 |
| E5 | 20% 局部辅助任务 | LLM+aligner；冻结 ViT | 3,072 | 1,560（2 epoch） | 每 195 step，共 8 个 |

## 3. 正式训练完成情况

训练队列按 Base Dev、E5、E2、E1、E3、E4 顺序串行执行。E1～E5 均达到预期最终 step，8 个 checkpoint、adapter 权重和 trainer state 完整，5 组训练合计约 16 小时 7 分钟。

| 实验 | 运行目录 | 训练时长 | 末次日志 loss | 最终 eval_loss | 最终 eval_token_acc |
|---|---|---:|---:|---:|---:|
| E1 | `v0-20260729-015806` | 4:23:21 | 0.0045 | 0.3496 | 93.39% |
| E2 | `v0-20260728-233940` | 2:18:23 | 0.1453 | 0.2154 | 94.36% |
| E3 | `v0-20260729-062123` | 2:24:52 | 0.1693 | **0.2028** | **94.57%** |
| E4 | `v0-20260729-084616` | 4:02:00 | 0.0670 | 0.2581 | 94.18% |
| E5 | `v0-20260728-204136` | 2:58:04 | 0.0854 | 0.2417 | 94.42% |

表中的末次 loss 是最后一个训练日志窗口，不是全程平均 `train_loss`。E3 的最终 token loss 最低，但业务目标是图片级 GOOD/BAD，因此不按 trainer 指标直接选模型。

## 4. 固定 Dev 图片级结果

每个实验的 8 个 checkpoint 均使用相同的 Transformers 自然生成协议评测：non-thinking、temperature 0、单图 batch 1、最多生成 128 token，不使用结构化解码强制合法 JSON。无效输出按保守规则计错。选择规则为：完整 schema 合法率不低于 99.5%、FPR 不高于 25%，随后依次按 Recall、Accuracy、F1 降序和 step 升序选择。

| 模型 | 选中 step | TP | FN | FP | TN | Recall | FPR | Accuracy | F1 | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 Base | — | 12 | 46 | 9 | 133 | 20.69% | 6.34% | 72.50% | 30.38% | 98.00% |
| **E5** | **780** | **46** | **12** | 25 | 117 | **79.31%** | 17.61% | 81.50% | **71.32%** | 100.00% |
| E2 | 1,248 | 38 | 20 | 11 | 131 | 65.52% | 7.75% | **84.50%** | 71.03% | 100.00% |
| E1 | 1,248 | 40 | 18 | 22 | 120 | 68.97% | 15.49% | 80.00% | 66.67% | 100.00% |
| E3 | 1,092 | 39 | 19 | 13 | 129 | 67.24% | 9.15% | 84.00% | 70.91% | 100.00% |
| E4 | 520 | 35 | 23 | **8** | **134** | 60.34% | **5.63%** | **84.50%** | 69.31% | 100.00% |

主要观察：

1. 五组 LoRA 的 Recall、Accuracy 和 F1 均显著高于未微调 Qwen3.6 Base，说明微调收益不只是 JSON 格式改善。以 E5-780 为例，Recall 提升 58.62 个百分点、Accuracy 提升 9.00 个百分点、F1 提升 40.94 个百分点，但 FPR 增加 11.27 个百分点。
2. E5-780 是召回优先模型，在 58 个 BAD 中识别 46 个，达到当前预登记选择规则；代价是 25 个 FP。E2-1248 与 E5-780 的 F1 接近（71.03% vs 71.32%），但 E2 少 14 个 FP、同时少 8 个 TP，属于更均衡的备选。
3. E2、E3 和 E4 的 Accuracy 接近，E3 的最低 token loss 没有转化为最佳图片级指标；再次证明不能只按 `eval_loss` 或 `eval_token_acc` 选 checkpoint。
4. E3 没有显示出解除 ViT 冻结的明确收益；其 Recall、FPR、Accuracy 和 F1 均未超过同数据的 E2。Qwen3.6 上仍优先保留“LLM+aligner LoRA、冻结 ViT”路线。
5. E4 的 40% 局部辅助数据偏保守；E5 将局部任务降至 20% 后显著恢复 Recall，但也提高 FPR。局部辅助比例仍是主要的召回/误报权衡变量。

## 5. 与 Qwen3.5 阶段的简要对照

本轮已证明 Qwen3.6 LoRA 明显超过自身未微调基线，但尚不能证明它整体优于已完成调优的 Qwen3.5。作为同一修正版 Dev 的参考，Qwen3.5 E2-1248 曾达到 Recall 74.14%、FPR 3.52%、Accuracy 90.00%、F1 81.13%；Qwen3.6 E5-780 的 Recall 更高 5.17 个百分点，但 FPR 高 14.09 个百分点、Accuracy 低 8.50 个百分点、F1 低 9.81 个百分点。Qwen3.6 E2-1248 也未复现 Qwen3.5 E2 的平衡指标。

因此当前 Qwen3.6 的阶段价值是“基座迁移和完整训练链路成功、相对自身 Base 提升显著、召回上限较高”，而不是替代现有 Qwen3.5 最佳平衡模型。

## 6. 当前结论与后续边界

- 按预登记的 Dev 选择协议，当前锁定 **Qwen3.6 E5 checkpoint-780** 作为召回优先候选；E2 checkpoint-1248 只作为平衡型备选，不能通过观察 Test 后重新选择。
- 复核 Test N=241 已在 Qwen3.5 阶段被观察并参与定向复核，只能用于阶段比较，不能用于 checkpoint、Prompt、阈值或训练参数选择。Qwen3.6 阶段 Test 脚本已经准备，但截至本文记录时尚未运行。
- 正式泛化结论仍需要一批不参与任何选择的新盲测数据。
- LoRA 合并等价性、vLLM/NPU 后端一致率尚未验证，应在阶段 Test 确认候选后继续执行。

## 7. 主要代码与产物

- 训练入口：`scripts/train_qwen36_replica.sh`
- 正式训练队列：`scripts/run_qwen36_replica_queue.sh`
- checkpoint Dev 评测：`scripts/run_qwen36_dev_checkpoints.sh`
- Dev 评测队列：`scripts/run_qwen36_dev_evaluation_queue.sh`
- Dev 汇总：`/home/data/h30082292/data/pose/artifact_detection_training/evaluations/qwen36_27b/dev_comparison_v1.json`
- 一次性阶段 Test：`scripts/run_qwen36_e5_780_stage_test.sh`（未运行）

相关代码提交依次为：processor 预检 `588fe50`、环境诊断 `7f7a84d`、自然推理 PoC `e803324`、四卡 LoRA PoC `145370b`、E1～E5 训练队列 `f9b6a6f`、固定 Dev checkpoint 评测 `beee6b4`、一次性阶段 Test 脚本 `e6849b7`。
