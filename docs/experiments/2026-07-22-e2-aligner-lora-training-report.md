# E2 aligner-LoRA 第二次训练总结

实验：Qwen3.5-27B 多模态 LoRA；训练完成于 2026-07-21。E2 用于验证视觉—语言 aligner 适配能否改善图片级 GOOD/BAD 判断。固定 Test 因标注仍待修正，全程未参与训练、checkpoint 或 Prompt 选择。

## 1. 数据集

- 开发数据共 8,376 张独立图片：Train 8,176 张、原始固定 Dev 200 张。
- 继续使用 E1 broad-clean Train：剔除 150 张唯一类别为“其他”的 BAD；混合 BAD 保留图片但删除“其他”辅助标签。
- 过滤后 Train 共 8,026 张：GOOD 6,074、BAD 1,952；BAD 重复采样一次，每 epoch 共 9,978 行。
- 训练时使用原始 Dev：GOOD 149、BAD 51；Train/Dev 无图片重叠。
- 训练完成后对模型边界样本进行人工复核，另行冻结修正版 Dev：GOOD 142、BAD 58。修正版只用于后续评估，不回流训练。

## 2. 训练策略、参数与 Prompt

- 基座 Qwen3.5-27B，4×A100 80GB，BF16，DeepSpeed ZeRO-2，FlashAttention 2。
- E1 只训练 LLM LoRA；E2 将 `all-linear` LoRA 扩展至 LLM 和 aligner，ViT 继续冻结。
- LoRA rank 16、alpha 32、dropout 0.05；学习率 5e-5，cosine，warmup ratio 0.05，weight decay 0.1。
- 全局 batch 16，`max_length=2048`，图像 token 上限 1,024，seed/data seed 均为 42。
- 根据 E1 过拟合结果，将训练缩短为 2 epoch，共 1,248 step；每 156 step 同时验证和保存，共保留 8 个 checkpoint。
- checkpoint 检查确认 adapter 含 996 个张量，其中 4 个为 aligner merger LoRA 张量，说明改动实际生效。

### Prompt 设计

System prompt：

```text
你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。严格只输出指定JSON，不要添加分析、解释或Markdown。
```

User prompt：

```text
<image>
检查这张图片。输出decision、categories和reasons。decision只能是GOOD或BAD。
```

图片级 `decision` 是核心目标，`categories` 和短 `reasons` 作为辅助监督；GOOD 的辅助字段为空。E2 与 E1 使用完全相同的 Prompt，确保结果变化可归因于模型训练范围。

## 3. 结果统计和分析

### 训练运行

- 完成 1,248/1,248 step，耗时约 2 小时 21 分；无 OOM、NaN 或异常退出。
- 最终 `train_loss=0.2053`；最终 Dev `eval_loss=0.2165`、`eval_token_acc=0.9441`。
- trainer 按 token 级 loss 选择 checkpoint-624（`eval_loss=0.1921`）；图片级生成评测选择 checkpoint-1248，再次说明不能只依赖 trainer loss 选择业务模型。

### 图片级二分类结果

在原始 Dev 上，E2 checkpoint-1248 与 E1 的汇总指标相同：TP 26、FN 25、FP 22、TN 127，Recall 50.98%、FPR 14.77%、Accuracy 76.50%、F1 52.53%。但两者有 20 张决策不同，说明相同汇总指标掩盖了样本层面的变化。

对 57 张边界样本人工复核后，以其余 143 张继续保留原 Gold 的条件修正版口径重算：

| 模型 | TP | FN | FP | TN | Recall | FPR | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 checkpoint-1248 | 39 | 19 | 9 | 133 | 67.24% | 6.34% | 86.00% | 73.58% |
| **E2 checkpoint-1248** | **43** | **15** | **5** | **137** | **74.14%** | **3.52%** | **90.00%** | **81.13%** |

E2 相比 E1 多识别 4 张 BAD，同时少误报 4 张 GOOD，共减少 8 个错误；达到 Recall≥72%、FPR≤25%、Accuracy≥72% 的 Go 标准，但 Recall 尚未达到 78% 挑战目标。

## 4. 归因

1. **原始 Dev 标签噪声掩盖了 E2 改进。** 57 张重点复核样本中有 27 张需要反转 GOOD/BAD；在原 Gold 下，E1/E2 的改进和退化互相抵消。
2. **aligner LoRA 有正向作用。** 条件修正后 E2 同时减少 FN 和 FP，说明视觉—语言映射适配比仅训练 LLM 更有效。
3. **剩余问题集中在细粒度视觉边界。** E2 仍有 15 个 FN，其中手部异常 8 个；5 个 FP 全部被预测为手部异常。
4. **不是简单的类别数量不足。** Train 已包含 965 张手部 BAD，但仍同时出现手部漏检和误报，更像是手指、小结构及遮挡场景的视觉区分不足。
5. **修正指标仍是开发口径。** 本次只定向复核了 E1/E2 的 57 张重点样本，其余 143 张未逐张重标；最终泛化能力仍需由修正后的隔离 Test 验证。

## 5. 下一步计划

1. 冻结修正版 Dev 200 张及 SHA256，不覆盖原始 Dev，也不将复核样本回流训练。
2. E3 保持 Train、采样、Prompt、LoRA rank、学习率和 2-epoch 协议不变，仅将 LoRA 从 LLM+aligner 扩展至 ViT。
3. E3 目标是在 58 个 BAD 中至少识别 46 个，即在不明显扩大 FP 的前提下，比 E2 至少多找回 3 个 TP。
4. 继续用图片级 Recall、FPR、Accuracy、F1 选择 checkpoint，不以 token loss 单独决策。
5. Test 完成标注修正前继续隔离；模型与协议锁定后只运行一次最终评测。
