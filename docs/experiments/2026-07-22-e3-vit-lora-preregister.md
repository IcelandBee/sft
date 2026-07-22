# E3 ViT-LoRA 预注册

## 结论依据

修正版固定 Dev 共 200 张（GOOD 142、BAD 58）。E2 checkpoint-1248 的条件修正指标为 Recall 74.14%、FPR 3.52%、Accuracy 90.00%、F1 81.13%，已经达到 Go 标准，但未达到 Recall 78% 的挑战目标。

E2 剩余 15 个 FN 中，8 个为明显异常、7 个为边界异常；类别集中在手部异常 8、文字/符号异常 3、人体结构异常 2。5 个 FP 全部是无异常图片，且全部被预测为手部异常。Train 中已有 965 张独立手部 BAD，说明主要矛盾不是类别数量不足，而是细粒度局部视觉判别边界不足。

## 唯一实质模型改动

- E2：LLM 与 aligner 使用 `all-linear` LoRA，ViT 冻结。
- E3：LLM、aligner 与 ViT 均使用 `all-linear` LoRA。

对应训练参数仅将 `--freeze_vit true` 改为 `--freeze_vit false`。LoRA rank、学习率、采样、Prompt 和训练时长保持不变，以便归因。

## 固定数据与训练协议

- Train：broad-clean，8,026 张独立图片；GOOD 6,074、BAD 1,952；BAD 重复一次，每 epoch 9,978 行。
- Dev：`dev_adjudicated_v1`，200 张；GOOD 142、BAD 58；SHA256 为 `cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb`。
- 修正版 Dev 仅用于验证和 checkpoint 选择，不进入训练。
- LoRA r16、alpha32、dropout0.05；LR 5e-5，cosine，warmup0.05，weight decay0.1。
- 全局 batch 16、BF16、ZeRO-2、max length 2,048、2 epochs。
- 每 156 step 同时 eval/save，共保留 8 个 checkpoint。
- 固定 Test 继续隔离，不用于任何选择。

## 成功与停止条件

在相同生成式推理协议下，对 8 个 checkpoint 使用修正版固定 Dev：

1. schema 合法率不低于 99.5%；
2. FPR 低于 20%；
3. Recall 不低于 78%（58 个 BAD 至少识别 46 个，即相对 E2 至少多 3 个 TP）；
4. Accuracy 不低于 76%；
5. 若 Recall 未提高或 FP 明显扩张，不继续增加 ViT LoRA rank、epoch 或解冻范围。

修正版 Dev 的复核范围由 E1/E2 错误触发，因此它适合作为后续开发基准，但最终泛化结论仍必须等待修正后的隔离 Test 一次性验证。
