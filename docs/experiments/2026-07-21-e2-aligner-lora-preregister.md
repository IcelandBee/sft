# E2 aligner-LoRA 预注册

## 目标

在不使用固定 Test 的前提下，验证 E1 图片级判别不足是否与视觉—语言 aligner 完全冻结有关，优先提升固定 Dev Recall，同时约束 FPR。

## 唯一实质模型改动

- E1：LLM `all-linear` LoRA，ViT 与 aligner 均冻结；
- E2：LLM 与 aligner 的 `all-linear` 层使用 LoRA，ViT 仍冻结。

按照 ms-swift 多模态 LoRA 参数语义，`freeze_aligner=false` 会使 `all-linear` LoRA 覆盖 aligner 线性层，更新随 adapter checkpoint 保存。

## 保持不变

- broad-clean Train 独立图片 8,026 张；
- GOOD 6,074，BAD 1,952，BAD 总计采样两次；
- 每 epoch 9,978 行；
- 固定 Dev 200 张，GOOD 149、BAD 51；
- LoRA r16、alpha32、dropout0.05；
- LR 5e-5、cosine、warmup0.05、weight decay0.1；
- 全局 batch 16、BF16、ZeRO-2、max length 2,048；
- seed/data seed 42；
- 生成式 Dev 推理和严格评分协议保持不变；
- 固定 Test 继续隔离。

## 资源性调整

- 训练上限从 4 epoch 缩短为 2 epoch，因为 E1 在 2 epoch 后 trainer Dev loss 和图片级指标均恶化；
- 每 156 step 同时 eval 和 save，保留 0.25 epoch 粒度的 8 个 checkpoint；
- 预计总计 1,248 update steps，训练时间约 2–3 小时。

缩短训练只删除 E1 已被证明无益的后半程，不改变前 2 epoch 的优化协议。

## Dev 选择与停止条件

对 156、312、468、624、780、936、1092、1248 八个 checkpoint 使用同一固定 Dev 协议：

1. schema 合法率 ≥ 99.5%；
2. FPR ≤ 25%；
3. Recall 优先，其次 Accuracy、F1、较早 step；
4. 与 E1 checkpoint-1248 的 Recall 50.98%、FPR 14.77%、Accuracy 76.50%、F1 52.53% 比较；
5. Test 不参与任何选择。

## 当前不同时修改的项目

- 不重新加入“其他”；
- 不改变 BAD 重复倍数；
- 不改变学习率、LoRA rank 或 prompt；
- 不解冻 ViT；
- 不使用 E1 的 FN/FP 回流训练。

这些变量待 E2 完成后逐项决定，避免一次训练无法归因。
