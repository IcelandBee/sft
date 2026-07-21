# E1 broad-clean 首轮正式训练总结

实验：Qwen3.5-27B 多模态 LoRA；训练完成于 2026-07-17，固定 Dev 评测完成于 2026-07-20。当前只使用 Train/Dev，259 张 Test 因标注仍需修正而继续隔离。

## 1. 数据集

- 开发数据共 8,376 张独立图片，其中 Train 8,176 张、固定 Dev 200 张；Dev 为 GOOD 149、BAD 51。
- broad-clean 规则：Train 中若 BAD 的唯一类别为“其他”，整图剔除；若“其他”和明确类别共存，保留图片但删除“其他”辅助标签；Dev 原样保留。
- 共剔除 150 张“仅其他”BAD，过滤后 Train 为 8,026 张：GOOD 6,074、BAD 1,952。
- BAD 在训练 JSONL 中重复采样一次，即每张出现两次；每 epoch 共 9,978 行：GOOD 6,074、BAD 3,904。
- Train/Dev 无图片重叠，所有图片路径有效。

过滤后 1,952 张独立 BAD 的主要类别覆盖如下（多标签计数可重叠）：

| 类别 | 训练图片数 |
|---|---:|
| 手部异常 | 965 |
| 文字/符号异常 | 447 |
| 面部/五官异常 | 299 |
| 人体结构异常 | 230 |
| 常识不合理 | 171 |
| 关系/逻辑矛盾 | 170 |
| 其他 | 0 |

## 2. 训练策略、参数

- 基座模型：Qwen3.5-27B；4×A100 80GB，BF16，DeepSpeed ZeRO-2，FlashAttention 2。
- 冻结 ViT 和 aligner，只对 LLM 的 `all-linear` 层训练 LoRA。
- LoRA：rank 16、alpha 32、dropout 0.05。
- 学习率 5e-5，cosine scheduler，warmup ratio 0.05，weight decay 0.1。
- 训练 4 epoch；每卡 batch 1，梯度累积 4，全局 batch 16。
- `max_length=2048`，`IMAGE_MAX_TOKEN_NUM=1024`，seed/data seed 均为 42。
- 每 156 step 评估，每 312 step 保存，共得到 8 个 checkpoint：312–2496。

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

目标输出格式：

```json
{"decision":"GOOD","categories":[],"reasons":[]}
```

```json
{"decision":"BAD","categories":["手部异常"],"reasons":["手指畸形"]}
```

设计上将图片级 `decision` 作为核心目标并放在首字段，`categories` 和短 `reasons` 作为辅助监督；GOOD 的辅助字段必须为空，BAD 最多保留 3 项辅助信息。训练和推理均使用 non-thinking 模式，并要求固定 JSON schema，以减少自由文本造成的解析歧义。本版 Prompt 保持简短，没有加入类别定义、严重程度边界或正反例，这也可能是部分边界样本出现 FN/FP 的因素。

## 3. 结果统计和分析

### 训练运行

- 完成 2,496/2,496 step，耗时 4 小时 20 分；无 OOM、NaN 或异常退出。
- 最终 `train_loss=0.1226`；最终 Dev `eval_loss=0.3898`、`eval_token_acc=0.9370`。
- trainer 最低 Dev loss 出现在 checkpoint-624（1 epoch，`eval_loss=0.1923`）；之后 loss 持续上升，存在明显过拟合。

### 固定 Dev 图片级二分类

8 个 checkpoint 均使用完全相同的确定性生成协议评测固定 200 张 Dev。`BAD` 为正类，categories/reasons 不直接参与 TP/FN/FP/TN。

| Step | Recall | FPR | Accuracy | F1 |
|---:|---:|---:|---:|---:|
| 312 | 27.45% | 8.72% | 75.00% | 35.90% |
| 624 | 43.14% | 10.74% | 77.50% | 49.44% |
| 936 | 29.41% | 8.72% | 75.50% | 37.97% |
| **1248** | **50.98%** | **14.77%** | **76.50%** | **52.53%** |
| 1560 | 29.41% | 8.05% | 76.00% | 38.46% |
| 1872 | 41.18% | 12.75% | 75.50% | 46.15% |
| 2184 | 31.37% | 11.41% | 74.00% | 38.10% |
| 2496 | 31.37% | 11.41% | 74.00% | 38.10% |

按预注册规则选择 checkpoint-1248。其混淆矩阵为 TP 26、FN 25、FP 22、TN 127。模型预测 BAD 48 张，真实 BAD 51 张，数量接近，但正确匹配不足，说明主要问题是图片区分能力，而非单纯不愿预测 BAD。

主要类别 Recall：

| 类别 | Recall | Missed |
|---|---:|---:|
| 手部异常 | 54.55% | 10 |
| 文字/符号异常 | 64.29% | 5 |
| 面部/五官异常 | 44.44% | 5 |
| 常识不合理 | 25.00% | 3 |
| 人体结构异常 | 50.00% | 2 |
| 关系/逻辑矛盾 | 50.00% | 2 |
| 其他 | 71.43% | 2 |

排除 Dev 中 4 张“仅其他”BAD 后，Recall 仅从 50.98%变为 51.06%，最佳 checkpoint 仍为 1248，因此“其他”并非本轮低 Recall 的直接原因。

## 4. 归因

1. **视觉任务适配不足。** E1 只训练 LLM LoRA，ViT 和 aligner 全部冻结；模型能够学习输出格式和文本标签，但图片异常与 GOOD/BAD 决策之间的映射仍不稳定。
2. **存在过拟合。** trainer Dev loss 在 1 epoch 后持续恶化，图片级最佳点为 2 epoch，之后指标下降；4 epoch 训练过长。
3. **不是简单的类别数量或预测倾向问题。** 手部异常有 965 张训练图，仍同时贡献最多 FN 和 FP；模型预测 BAD 总量也接近真实数量。
4. **删除“其他”可能减少异常多样性，但不是已证实主因。** Train 中“其他”为 0，而 Dev“其他”Recall 为 71.43%；是否存在间接影响只能通过后续对照训练验证。
5. **token 指标与业务目标不一致。** token accuracy 较高，但图片级 Recall 低，后续必须继续以固定 Dev 二分类指标选择模型。

## 5. 下一步计划

1. 启动 E2：保持 broad-clean 数据、BAD×2、LoRA rank、学习率和提示词不变；ViT 继续冻结，但让 LoRA 同时覆盖 LLM 和 aligner。
2. E2 训练上限缩短为 2 epoch，并每 156 step 保存和评估一次，重点观察 Recall 能否提升且 FPR 保持在 25%以内。
3. 训练期间人工复核 checkpoint-1248 的 25 个 FN 和 22 个 FP，重点分析手部、文字、面部和常识类的标注边界及 Train/Dev 视觉差异。
4. E2 完成后使用同一固定 Dev 协议评测全部 checkpoint，并与 E1 checkpoint-1248 对比。
5. 暂不使用 259 张 Test；待标注修正、模型和协议锁定后再进行一次最终评测。
