# E1 broad-clean：Qwen3.5-27B 多模态 LoRA 首轮正式训练报告

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 实验编号 | E1 broad-clean r16 e4 v1 |
| 报告日期 | 2026-07-21 |
| 训练实际目录 | `/home/data/h30082292/data/pose/artifact_detection_training/runs/e1_broad_clean_r16_e4_v1/v0-20260717-185936` |
| 训练数据目录 | `/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e1_broad_clean_json_v1` |
| Dev 评测目录 | `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1` |
| 基座模型 | `/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B` |
| 框架 | ms-swift 4.4.1、PEFT LoRA、DeepSpeed ZeRO-2 |
| 当前状态 | 训练和固定 Dev 评测完成；隔离 Test 尚未运行 |

本报告用于记录首轮正式训练的数据、协议、结果、已确认结论、待验证归因和后续计划。文中将事实与假设分开陈述，避免使用 Test 反向选择 checkpoint、阈值或提示词。

## 2. 目标与预注册成功标准

核心目标是图片级 `GOOD/BAD` 二分类明显超过未微调基线。`categories`、`reasons` 等字段只作为辅助监督和可解释输出，不直接参与 TP/FN/FP/TN 计算；`BAD` 为正类。

固定 Test 共 259 张，完全隔离。未微调基线如下：

| TP | FN | FP | TN | Recall | FPR | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 78 | 44 | 42 | 95 | 63.93% | 30.66% | 66.80% | 64.46% |

预注册目标：

- Go：Recall ≥ 72%、FPR ≤ 25%、Accuracy ≥ 72%；
- 挑战目标：Recall ≥ 78%、FPR ≤ 20%、Accuracy ≥ 76%；
- Test 不用于 checkpoint、阈值或提示词选择，只允许在 Dev 完成模型选择后运行一次。

说明：当前仅有 Dev 结果。Dev 与 Test 不是同一批图片，不能把 Dev 指标直接当作 Test 结果；本报告只把目标阈值作为 Dev 阶段的风险参考。

## 3. 数据集与处理过程

### 3.1 数据划分

- 开发数据原始共 8,376 张独立图片；
- 固定 Dev：200 张，其中 GOOD 149、BAD 51；
- 原始 Train：8,176 张；
- 固定 Test：另有 259 张，保持完全隔离；
- Train 与 Dev 无图片重叠，转换检查全部通过。

### 3.2 broad-clean 过滤规则

首轮训练采用 broad-clean：

1. Train 中，若 BAD 图片的唯一异常类别为“其他”，整张图片从训练集剔除；
2. 若 BAD 图片同时包含“其他”和其他明确异常类别，保留图片，但移除“其他”辅助类别实例；
3. Dev 不做此过滤，保持原始 200 张不变；
4. GOOD 图片保持不变。

过滤结果：

| 数据 | 独立图片 | GOOD | BAD |
|---|---:|---:|---:|
| 原始 Train | 8,176 | 待补充 | 待补充 |
| broad-clean Train | 8,026 | 6,074 | 1,952 |
| 被整图剔除 | 150 | 0 | 150 |
| Dev（原样保留） | 200 | 149 | 51 |

当前尚缺“混合 BAD 中被移除的‘其他’辅助实例数”，待训练集覆盖审计后补充。

### 3.3 训练采样

为缓解类别不平衡，BAD 图片在训练 JSONL 中总计出现两次：

- GOOD：6,074 行；
- BAD：1,952 × 2 = 3,904 行；
- 每个 epoch：9,978 行；
- 独立图片仍为 8,026 张，重复只发生在训练采样层；
- 每 epoch 行级比例约为 GOOD 60.88%、BAD 39.12%。

数据路径：

- Train：`.../e1_broad_clean_json_v1/train.jsonl`；
- Dev：`.../e1_broad_clean_json_v1/dev.jsonl`。

### 3.4 broad-clean Train 类别覆盖审计

对 9,978 行训练 JSONL 按图片路径去重后，得到 8,026 张独立图片；其中 1,952 行来自 BAD 的第二次采样，重复图片标签一致。独立图片决策统计为 GOOD 6,074、BAD 1,952，与数据转换记录一致。

在 1,952 张独立 BAD 图片上统计辅助类别：

| 类别 | 独立 BAD 图片数 |
|---|---:|
| 手部异常 | 965 |
| 文字/符号异常 | 447 |
| 面部/五官异常 | 299 |
| 人体结构异常 | 230 |
| 常识不合理 | 171 |
| 关系/逻辑矛盾 | 170 |
| 其他 | 0 |

类别为多标签，计数之和可以超过 1,952。该审计确认 broad-clean Train 的最终辅助标签中已不存在“其他”。

## 4. 训练策略与参数

### 4.1 硬件与精度

- 4 × NVIDIA A100 80GB，使用物理 GPU 4–7；
- BF16；
- FlashAttention 2；
- DeepSpeed ZeRO-2；
- 单卡训练显存稳定约 56.26 GiB；
- 无 OOM、NaN 或异常退出。

### 4.2 可训练参数范围

- 基座：Qwen3.5-27B 多模态模型；
- 冻结 ViT；
- 冻结视觉对齐器 aligner；
- LLM 使用 `all-linear` LoRA；
- LoRA rank 16；
- LoRA alpha 32；
- LoRA dropout 0.05。

这意味着首轮只调整语言模型线性层的 LoRA 参数，视觉编码器和视觉—语言对齐部分保持基座状态。

### 4.3 优化与批量参数

| 参数 | 值 |
|---|---:|
| Learning rate | 5e-5 |
| Scheduler | cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.1 |
| Epoch | 4 |
| 每卡 batch size | 1 |
| GPU 数 | 4 |
| Gradient accumulation | 4 |
| 全局 batch size | 16 |
| Max length | 2,048 |
| IMAGE_MAX_TOKEN_NUM | 1,024 |
| Gradient checkpointing | 开启 |
| Seed / data seed | 42 / 42 |

### 4.4 评估和保存频率

- 每 156 step 运行一次 trainer Dev 评估，约每 0.25 epoch 一次；
- 每 312 step 保存一次 checkpoint，约每 0.5 epoch 一次；
- 共保留 8 个 checkpoint：312、624、936、1248、1560、1872、2184、2496；
- `save_total_limit=8`；
- 最终完成 2,496/2,496 step。

### 4.5 数据加载与模板开关

- `split_dataset_ratio=0`，明确使用固定 Train/Dev 文件；
- `dataset_shuffle=true`，`val_dataset_shuffle=false`；
- `strict=true`；
- `lazy_tokenize=true`；
- `add_non_thinking_prefix=true`；
- `per_device_eval_batch_size=1`；
- `dataset_num_proc=4`，`dataloader_num_workers=2`；
- `save_only_model=false`，每个 checkpoint 同时保存 trainer state；
- `report_to=none`。

## 5. 训练技术结果

### 5.1 运行状态

- 完整完成 4 epoch；
- 训练耗时 4 小时 20 分 43 秒；
- 平均约 6.27 秒/step；
- 最终 `train_loss=0.1226`；
- 最终 Dev `eval_loss=0.3897803`；
- 最终 Dev `eval_token_acc=0.9370`；
- 正常保存 `checkpoint-2496`；
- ms-swift 按最低 `eval_loss` 标记 `checkpoint-624` 为 best checkpoint；
- 退出时只有未显式调用 `destroy_process_group()` 的 NCCL warning，训练已正常结束，该 warning 不影响 checkpoint 有效性。

### 5.2 保存点的 trainer Dev loss

| Step | Epoch | Eval loss | Eval token accuracy |
|---:|---:|---:|---:|
| 312 | 0.5 | 0.2056 | 0.9383 |
| 624 | 1.0 | **0.1923** | 0.9447 |
| 936 | 1.5 | 0.2126 | 0.9413 |
| 1248 | 2.0 | 0.2301 | 0.9410 |
| 1560 | 2.5 | 0.2938 | 0.9429 |
| 1872 | 3.0 | 0.3287 | 0.9392 |
| 2184 | 3.5 | 0.3841 | 0.9389 |
| 2496 | 4.0 | 0.3898 | 0.9370 |

`eval_loss` 在 1 epoch 的 step 624 达到最低点，随后总体持续上升；训练尾部 loss 已接近完全拟合，而 Dev loss 明显恶化，构成清晰的过拟合信号。但 trainer 的 token 级 loss/accuracy不等价于图片级二分类效果，因此没有直接用 step 624 作为最终模型。

## 6. 固定 Dev 生成式评测协议

对全部 8 个保存 checkpoint 使用完全相同的生成协议评测固定 200 张 Dev：

- `infer_backend=transformers`；
- BF16、FlashAttention 2；
- 单卡、`max_batch_size=1`；
- `temperature=0.0`、`num_beams=1`；
- `max_new_tokens=128`；
- `add_non_thinking_prefix=true`；
- `IMAGE_MAX_TOKEN_NUM=1024`；
- 严格要求固定空 think 包络及指定 JSON schema；
- INVALID 输出按 gold 类别保守计错；
- 严格校验 200 张图片的顺序、路径、system/user prompt 与 gold；
- 不使用 Test 数据。

固定 system prompt：

```text
你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。严格只输出指定JSON，不要添加分析、解释或Markdown。
```

固定 user prompt：

```text
<image>
检查这张图片。输出decision、categories和reasons。decision只能是GOOD或BAD。
```

四张 GPU 并行，每卡顺序处理两个 checkpoint。每个 checkpoint 的 200 张推理约 529–556 秒，包含模型加载和评估的墙钟时间约 10–11 分钟；两轮总墙钟时间约 21 分钟。8 个任务均为 200 行，推理和评估返回码均为 0。

固定 Dev SHA256：

`453d4a0e39fd51585ddc36fa2da1461fdca46c24d84c4cd4ef0503de6625b584`

## 7. Dev 二分类结果

| Step | TP | FN | FP | TN | Recall | FPR | Accuracy | F1 | Schema valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 312 | 14 | 37 | 13 | 136 | 27.45% | 8.72% | 75.00% | 35.90% | 100% |
| 624 | 22 | 29 | 16 | 133 | 43.14% | 10.74% | 77.50% | 49.44% | 100% |
| 936 | 15 | 36 | 13 | 136 | 29.41% | 8.72% | 75.50% | 37.97% | 100% |
| **1248** | **26** | **25** | **22** | **127** | **50.98%** | **14.77%** | **76.50%** | **52.53%** | **100%** |
| 1560 | 15 | 36 | 12 | 137 | 29.41% | 8.05% | 76.00% | 38.46% | 99.5% |
| 1872 | 21 | 30 | 19 | 130 | 41.18% | 12.75% | 75.50% | 46.15% | 100% |
| 2184 | 16 | 35 | 17 | 132 | 31.37% | 11.41% | 74.00% | 38.10% | 100% |
| 2496 | 16 | 35 | 17 | 132 | 31.37% | 11.41% | 74.00% | 38.10% | 100% |

预注册 checkpoint 选择规则为：schema 合法率 ≥ 99.5%、FPR ≤ 25%，然后按 Recall、Accuracy、F1、较早 step 依次排序。由此选中 `checkpoint-1248`。其原始结果 SHA256 为：

`689d30376e97bca17fd2b8f6e8a7aba9e5d092e22e37660c9cb9608fc7cf484d`

需要特别区分：`test_unlocked=true` 表示 Dev 选择协议已产生唯一 checkpoint，不代表模型已经达到业务 Go 指标。

## 8. checkpoint-1248 错误分析

### 8.1 总体错误形态

- Gold BAD：51 张；
- 模型预测 BAD：48 张，即 TP 26 + FP 22；
- TP 26、FN 25、FP 22、TN 127；
- 模型预测 BAD 的总量接近真实 BAD 总量，但匹配关系较差。

因此，本轮问题不是简单的“模型过于保守、很少预测 BAD”，而是图片级区分能力不足：一边漏掉真实异常，一边在其他 GOOD 图片上产生近似数量的误报。

### 8.2 BAD 类别表现

| Gold 类别 | Dev support | Detected | Missed | Recall |
|---|---:|---:|---:|---:|
| 手部异常 | 22 | 12 | 10 | 54.55% |
| 文字/符号异常 | 14 | 9 | 5 | 64.29% |
| 面部/五官异常 | 9 | 4 | 5 | 44.44% |
| 常识不合理 | 4 | 1 | 3 | 25.00% |
| 其他 | 7 | 5 | 2 | 71.43% |
| 人体结构异常 | 4 | 2 | 2 | 50.00% |
| 关系/逻辑矛盾 | 4 | 2 | 2 | 50.00% |

类别为多标签统计，同一图片可能计入多个类别，因此 support 和 missed 之和可能超过 BAD 图片总数。

主要 FN 原因为手指畸形、图案不成形、文字错乱、牙齿异常、瞳孔异常，以及多手、肢体融合、餐具结构、物理/常识不合理等。

FP 的预测类别主要为：

- 手部异常：11；
- 文字/符号异常：7；
- 人体结构异常：2；
- 常识不合理：1；
- 面部/五官异常：1。

手部和文字既贡献较多 FN，也贡献最多 FP，说明问题不只是类别召回不足，还可能包含正常/异常边界不一致、难例覆盖不足或视觉判别不稳定。

### 8.3 “其他”过滤的专项复算

固定 Dev 中：

- BAD 共 51 张；
- `categories` 仅为“其他”的 BAD 只有 4 张；
- 同时包含“其他”和其他类别的 BAD 有 3 张；
- 排除 4 张“仅其他”后，Dev 为 196 张。

checkpoint-1248 的复算结果：

| 口径 | Recall | FPR | Accuracy | F1 |
|---|---:|---:|---:|---:|
| 完整 Dev 200 | 50.98% | 14.77% | 76.50% | 52.53% |
| 排除“仅其他”后的诊断 Dev 196 | 51.06% | 14.77% | 77.04% | 51.61% |

排除后 Recall 只增加 0.08 个百分点，最佳 checkpoint 仍为 1248；同时“其他”整体 Recall 为 71.43%。因此，删除“其他”不能直接解释本轮低 Recall。完整 200 张仍是正式 Dev，196 张结果只作为诊断，不替代正式结果。

## 9. 归因分析

### 9.1 已被证据支持的结论

1. **训练技术上成功。** 所有 step 完成，checkpoint 完整，无 OOM/NaN/异常退出；NCCL 尾部 warning 非阻断。
2. **存在明显过拟合。** trainer Dev loss 在 1 epoch 最低，后续持续恶化；图片级最佳点在 2 epoch，之后二分类指标总体下降。
3. **token 级指标不能代表二分类质量。** 最终 token accuracy 仍约 93.7%，但图片级 Recall 仅约 31%；必须保留生成式图片级评测。
4. **低 Recall 不是由 4 张“仅其他”Dev 样本直接造成。** 过滤复算几乎没有改变 Recall，且“其他”类别自身并不弱。
5. **错误不是单纯的预测 BAD 数量不足。** 预测 BAD 48 张、真实 BAD 51 张，但 TP 仅 26，说明核心是判别错配。
6. **主要错误集中在手部、文字、面部/五官和常识类。** 手部和文字同时存在较多漏检与误报。
7. **训练类别数量不是唯一解释。** 手部异常在独立 BAD Train 中有 965 张，是覆盖最多的类别，但 Dev 仍有 10 个该类别 FN，且 22 个 FP 中有 11 个被预测为手部异常。
8. **Train 中“其他”为 0，但模型对 Dev“其他”的 Recall 为 71.43%。** 这表明基座能力和其他异常监督可以产生一定跨类别泛化，也进一步削弱了“缺少其他导致本轮全部低 Recall”的直接归因。

### 9.2 尚未证实的可能原因

以下内容是待验证假设，不应写成既定结论：

- broad-clean 删除“其他”可能间接降低异常形态多样性，进而影响其他类别的泛化；需要保持其他变量不变的 E2 对照训练才能证明；
- Train 中各明确类别覆盖不均，且 Dev 难例与 Train 可能存在视觉分布偏移；类别数量已完成统计，但还需进一步比较同类别内部的异常形态、尺度和难度；
- 手部、文字等类别的 GOOD/BAD 标注边界可能不一致，模型因此在相似视觉模式上同时产生 FN 和 FP；需人工复核 47 张错误图；
- 冻结 ViT 和 aligner 可能限制特定视觉异常的适配能力；该假设只能在数据与标注问题排除后，通过单变量解冻实验验证；
- 完整生成 `decision/categories/reasons` 的 token 监督可能使低损失更多反映文本模板拟合，而非图片级判别提升；需与更聚焦二分类决策的监督方案做对照。

## 10. 结论

E1 首轮正式训练在工程上成功、流程可复现，并在固定 Dev 上表现为较低 FPR 和较高 Accuracy；但 Recall 和 F1 不足，尚未实现“图片级 GOOD/BAD 二分类明显超过基线”的核心目标。当前选定 checkpoint-1248 适合作为 E1 的 Dev 代表模型和后续误差分析基准，不应默认使用最终 checkpoint-2496。

在未完成错误图和训练覆盖审计前，不建议把问题简单归因于“其他”过滤，也不建议立即消耗隔离 Test。

## 11. 下一步计划

### P0：完成数据和错误审计

1. 已完成 broad-clean Train 独立图片去重及 GOOD/BAD、BAD 类别覆盖统计；
2. 补充原始 Train 的 GOOD/BAD 数量及混合 BAD 中移除“其他”辅助实例的数量；
3. 人工复核 checkpoint-1248 的 25 FN 和 22 FP，重点检查手部、文字、面部、常识类；
4. 对高频错误类别比较 Train 与 Dev 的异常形态、尺度、构图和难度，而不只比较样本数量；
5. 标记错误来源：模型漏看、标注边界、图片本身模糊、Dev 标签错误、训练覆盖不足或分布偏移。

### P1：确定 E2 的单变量改动

数据审计完成后再锁定 E2，优先候选为：

1. 将人工确认有效的“仅其他”BAD 重新加入训练，保持其他训练配置不变，验证异常多样性的间接影响；
2. 修正边界不一致或明显错误的训练标签；
3. 针对高频 FN 类别补充/重采样难例，同时避免只增加易样本；
4. 缩短训练到约 1.5–2 epoch，并在 0.5–2 epoch 范围提高保存密度，降低后期过拟合风险；
5. 每轮只改变一个主要变量，继续用同一固定 Dev 生成协议比较。

暂不优先解冻 ViT/aligner，也不同时修改学习率、LoRA rank、数据和提示词，以免无法归因。

### P2：Test 使用条件

- 先在固定 Dev 上选出唯一 checkpoint；
- Dev Recall/FPR/Accuracy 至少呈现接近目标的可信趋势；
- 冻结 checkpoint、提示词、生成参数和评测代码；
- 最后只对隔离 Test 运行一次，并将结果与未微调基线直接比较；
- Test 结果不得用于回选 checkpoint、阈值或提示词。

## 12. 可复现性与审计产物

- 代码仓库：`https://github.com/IcelandBee/sft`；
- 初始 Dev 评测工具 commit：`8793ba6`；
- FN/FP 审计工具 commit：`bbb0c63`；
- 每个 checkpoint 均保存 `raw-result.jsonl`、`infer.log`、`job-status.json`、`evaluation/metrics.json`、`evaluation/parsed.jsonl` 和 `evaluation/errors.jsonl`；
- checkpoint-1248 额外保存 `error-analysis-v1/summary.json`、`fn.jsonl`、`fp.jsonl`；
- 所有正式评测均保持 Test 隔离。

## 13. 待补充清单

- [ ] 原始 Train 的 GOOD/BAD 数量；
- [ ] broad-clean 混合 BAD 中被删除的“其他”辅助实例数；
- [x] broad-clean Train 独立图片级类别覆盖统计；
- [ ] 25 FN / 22 FP 的人工视觉复核结论；
- [ ] E2 唯一主要变量与预注册配置；
- [ ] 最终冻结候选后的单次 Test 结果。
