# E3 ViT-LoRA 第三次训练总结

实验：Qwen3.5-27B 多模态 LoRA；训练完成于 2026-07-22。E3 用于验证在 E2 的 LLM+aligner LoRA 基础上继续适配 ViT，能否改善细粒度异常识别。实验技术上正常完成，但图片级指标低于 E1/E2，结论为负向。固定 Test 因标注仍待修正，全程未参与训练、checkpoint、Prompt 或协议选择。

## 1. 数据集

- 开发数据共 8,376 张独立图片：Train 8,176 张、固定 Dev 200 张。
- Train 继续使用 broad-clean：剔除 150 张唯一类别为“其他”的 BAD；混合 BAD 保留图片但删除“其他”辅助标签。
- 过滤后 Train 共 8,026 张：GOOD 6,074、BAD 1,952；BAD 重复采样一次，每 epoch 共 9,978 行。
- E3 使用人工复核后冻结的 Dev：GOOD 142、BAD 58，SHA256 为 `cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb`。
- Train/Dev 无图片重叠；修正版 Dev 只用于验证和 checkpoint 选择，不回流训练。

## 2. 训练策略、参数与 Prompt

- 基座 Qwen3.5-27B，4×A100 80GB，BF16，DeepSpeed ZeRO-2，FlashAttention 2。
- E2 对 LLM+aligner 使用 `all-linear` LoRA、冻结 ViT；E3 的唯一实质模型改动是解除 ViT 冻结，将相同 LoRA 扩展至 LLM+aligner+ViT。
- LoRA rank 16、alpha 32、dropout 0.05；学习率 5e-5，cosine，warmup ratio 0.05，weight decay 0.1。
- 全局 batch 16，`max_length=2048`，图像 token 上限 1,024，seed/data seed 均为 42。
- 训练 2 epoch，共 1,248 step；每 156 step 验证并保存，共保留 8 个 checkpoint。
- 可训练参数 124.73M，占模型参数的 0.4539%。最终 adapter 含 1,212 个张量，其中 ViT 216 个、aligner 4 个，确认视觉 LoRA 实际生效。

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

图片级 `decision` 是核心目标，`categories` 和短 `reasons` 为辅助监督。E3 与 E2 使用相同 Train、采样、Prompt 和优化参数，使结果变化主要归因于 ViT LoRA。

## 3. 结果统计和分析

### 训练运行

- 完成 1,248/1,248 step，耗时约 2 小时 30 分；显存稳定在 53.82 GiB，无 OOM、NaN 或异常退出。
- 最终 `train_loss=0.2176`；最终 Dev `eval_loss=0.2076`、`eval_token_acc=0.9475`。
- trainer 按 token loss 选择 checkpoint-624（`eval_loss=0.1985`）；8-checkpoint 图片级评测选择 checkpoint-1248。
- checkpoint-1248 的完整 schema 合法率为 100%。因此业务退化来自分类判断，而非输出格式。

### 图片级二分类结果

| 模型 | TP | FN | FP | TN | Recall | FPR | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 checkpoint-1248 | 39 | 19 | 9 | 133 | 67.24% | 6.34% | 86.00% | 73.58% |
| **E2 checkpoint-1248** | **43** | **15** | **5** | **137** | **74.14%** | **3.52%** | **90.00%** | **81.13%** |
| E3 checkpoint-1248 | 36 | 22 | 9 | 133 | 62.07% | 6.34% | 84.50% | 69.90% |

Base 在原自由生成协议下采用 decision-only 辅助解析，修正版 Dev 指标为 Recall 31.03%、FPR 41.55%、Accuracy 50.50%、F1 26.67%，完整 schema 合法率为 62.50%。补充的格式受控 Base 通过 token-trie 固定为合法 GOOD/BAD JSON 二选一，得到 TP 19、FN 39、FP 29、TN 113，Recall 32.76%、FPR 20.42%、Accuracy 66.00%、F1 35.85%，schema 合法率为 100%。受控合法率来自解码约束，且与 LoRA 自由生成协议不同，因此只作为排除 Base 格式问题后的补充参考。

E3 的 8 个 checkpoint 中，最高 Recall 为 checkpoint-1248 的 62.07%，没有节点达到 Recall≥72% 的 Go 标准。相比 E2，E3 少识别 7 张 BAD、多误报 4 张 GOOD，总错误增加 11 个；相比 E1 也少 3 个 TP。

逐图配对结果为：两者都正确 164 张、都错误 15 张、仅 E2 正确 16 张、仅 E3 正确 5 张，决策一致 179/200。E3 修复 5 个 E2 错误，但新引入 16 个错误。

## 4. 归因

1. **退化不是输出格式造成的。** E3 checkpoint-1248 的 payload 与完整 schema 合法率均为 100%，错误全部来自 GOOD/BAD 判断。
2. **E3 同时损失召回和特异性。** 相比 E2，E3 丢失 8 个原本检出的 BAD、仅找回 1 个 BAD；同时修复 4 个 FP、但新引入 8 个 FP，因此不是简单的保守或激进阈值偏移。
3. **损失覆盖多个异常类别。** E2 独有正确的 BAD 主要涉及关系/逻辑矛盾、手部、文字和常识；E3 的 FN 扩展到手部、关系/逻辑、文字、面部等类别，FP 也从 E2 的单一手部误报扩展到人体结构、文字和面部。
4. **结论不完全依赖修正标签。** E2 独有正确的 16 张中有 5 张使用修正 Gold；即使排除这 5 张，E2 在其余标签上仍以 11:5 领先 E3。
5. **当前强度的全 ViT LoRA 可能扰动预训练视觉表征。** 在仅 8,026 张训练图片和短文本辅助监督下，对完整 ViT 使用与 LLM/aligner 相同的 rank 与 5e-5 学习率，未改善细粒度边界，反而造成跨类别回退。该项是基于对照结果的解释性推断，不代表已定位到单一机制。
6. **token loss 与业务最优仍不一致。** checkpoint-624 的 eval loss 最低，但 checkpoint-1248 的图片级 Recall/F1 最好，checkpoint 必须继续按生成式二分类指标选择。

## 5. 下一步计划

1. E2 checkpoint-1248 继续作为当前最佳开发模型；E3 checkpoint-1248 作为负向实验留存，不进入 Test。
2. 按预注册停止条件，不继续增加当前全 ViT LoRA 的 rank、epoch 或解冻范围。
3. 格式受控 Base 已完成并达到 100% schema 合法率；当前工作留痕保留“自由生成同协议主对比 + 格式受控 Base 补充参考”，不重跑 E1/E2/E3。若后续需要严格量化同约束协议下的 LoRA 增益，只需补测当前最佳 E2 checkpoint-1248。
4. 后续训练回到 E2 路线，优先处理残余手部、文字和结构类样本；如再次尝试 ViT，只能作为更低学习率或更小解冻范围的独立对照实验。
5. 修正版 Dev 继续只用于开发选择；固定 Test 完成独立标注修正前保持隔离，最终协议锁定后只运行一次。
