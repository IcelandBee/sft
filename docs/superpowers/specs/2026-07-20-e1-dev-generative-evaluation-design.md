# E1 Dev 生成式评测设计

## 目标与边界

本阶段使用固定 200 张 Dev，对 E1 broad-clean 训练保留的 8 个 LoRA checkpoint 做图片级 GOOD/BAD 生成式评测，并只依据 Dev 选择一个 checkpoint。bbox、类别和短原因仅用于输出格式检查及错误诊断，不参与 checkpoint 主排序。

固定 259 张 Test 在 checkpoint、prompt、推理后端、解码参数、JSON 解析规则和选择规则全部锁定前不得运行。Test 只允许在最终选择后运行一次。

## 已确认输入

- 基座模型：`/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B`
- 正式数据目录：`/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e1_broad_clean_json_v1`
- Dev：上述目录中的 `dev.jsonl`，固定 200 行，GOOD 149、BAD 51，不打乱
- 训练版本：`e1_broad_clean_r16_e4_v1/v0-20260717-185936`
- checkpoint：312、624、936、1248、1560、1872、2184、2496
- 推理框架：训练环境中的 ms-swift 4.4.1 原生 `swift infer`，Transformers 后端
- 图像预算：`IMAGE_MAX_TOKEN_NUM=1024`
- 模型精度与注意力实现：BF16、FlashAttention 2
- 模板行为：保持 `add_non_thinking_prefix=true`

所有 checkpoint 已通过 adapter 权重、配置和 trainer state 完整性检查，单个目录大小均为 1.741 GiB。trainer `eval_loss` 在 checkpoint-624 最低，但该结果只作为参考，不能代替图片级生成式评测。

## 固定提示词

System：

```text
你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。严格只输出指定JSON，不要添加分析、解释或Markdown。
```

User：

```text
<image>
检查这张图片。输出decision、categories和reasons。decision只能是GOOD或BAD。
```

正式评测直接复用 Dev JSONL 中的 system/user 消息。文件中的 assistant 消息是 gold label，推理时必须由 ms-swift 当作标签移除，不能作为模型输入。单图冒烟测试必须先证明这一点。

## 推理流程

### 1. 单图冒烟测试

先使用 PoC LoRA `checkpoint-10` 和 `e1_poc20_v1/poc20.jsonl` 的第一张图片运行一次 `swift infer`。检查：

1. 基座模型与 LoRA 能成功联合加载；
2. 实际输入只包含 system、user 和图片，不包含 gold assistant；
3. 生成结果是新的模型响应；
4. 原始响应被完整保存；
5. 退出无 OOM、NaN 或模型/template 不兼容错误。

冒烟测试未通过时不得启动正式 Dev。

### 2. 正式批量评测

8 个 checkpoint 顺序串行运行，每次只切换 adapter 路径，其余参数完全不变。使用单卡、batch size 1，避免并行和动态 batch 引入不必要的差异。每个 checkpoint 写入独立结果目录，已完成且通过完整性校验的结果允许断点跳过。

固定生成规则：

- `temperature=0`；ms-swift 4.4.1 会在 Transformers 推理引擎中将其转换为 `do_sample=false`
- `max_new_tokens=128`
- 不启用流式输出
- 不使用额外 system prompt、few-shot、JSON 修复提示或重试 prompt

实现前以服务器上 `swift infer --help` 为唯一 CLI 参数依据；不得根据其他 ms-swift 版本猜测参数名。

## 输出与审计

每个 checkpoint 必须保存：

- 原始生成结果 JSONL，不覆盖、不清洗；
- 严格解析后的逐图记录；
- TP、FN、FP、TN 和汇总指标；
- 所有分类错误及 JSON/schema 错误；
- 运行清单：checkpoint、ms-swift 版本、推理参数、数据文件 SHA-256、prompt SHA-256、开始/结束时间和退出状态。

每条逐图记录至少包含稳定样本序号、图片路径或 image key、gold decision、原始响应、解析状态、预测 decision 和错误类型。输出目录不得包含 Test 结果。

## 严格解析规则

Qwen3.5 的 ms-swift 模板在非思考模式下会给响应加入固定包络 `'<think>\n\n</think>\n\n'`。解析时先对原始响应执行首尾空白去除，然后必须验证它以这一段精确包络开头且只出现一次；仅移除该固定包络，再对全部剩余文本调用 `json.loads`。

非空 think 内容、重复包络、其他 `<think>` 变体、Markdown code fence、JSON 前后附加文本、花括号子串抽取、正则猜测 GOOD/BAD 和自动 JSON 修复均不允许。缺少固定包络也属于协议错误。

合法响应必须同时满足：

1. 顶层是 JSON object；
2. 字段恰好为 `decision`、`categories`、`reasons`；
3. `decision` 恰好为大写 `GOOD` 或 `BAD`；
4. `categories` 和 `reasons` 均为字符串数组；
5. GOOD 的两个数组均为空；
6. BAD 的两个数组均至少包含一个非空字符串，且各不超过 3 项。

固定包络失败、payload JSON 解析失败和 schema 失败统一标为 `INVALID`，但分别记录原因。严格指标中 INVALID 一律按错误计：gold BAD 的 INVALID 计为 FN，gold GOOD 的 INVALID 计为 FP。这样 TP+FN+FP+TN 始终等于 200，不会通过丢弃无效输出来抬高指标。同时额外报告固定包络合法率、payload JSON 可解析率、schema 合法率、原始全文直接 JSON 可解析率和有效样本指标，便于诊断。

## 指标

每个 checkpoint 计算：

- TP、FN、FP、TN；
- Recall = TP / (TP + FN)；
- FPR = FP / (FP + TN)；
- Accuracy = (TP + TN) / 200；
- Precision 和 F1；
- 固定包络合法率；
- payload JSON 可解析率；
- 原始全文直接 JSON 可解析率；
- schema 合法率；
- GOOD/BAD 各自的 INVALID 数量。

同时保留逐图错误清单，后续按类别、原因和 bbox 面积做诊断，但这些辅助字段不改变图片级 gold decision。

## checkpoint 选择规则

选择只使用 Dev，按以下预先锁定的顺序执行：

1. schema 合法率至少 99.5%，即 200 张中最多允许 1 条 schema 无效；
2. FPR 不高于 25%；
3. 在满足前两项的 checkpoint 中选择 Recall 最高者；
4. Recall 相同时依次比较 Accuracy、F1；
5. 仍相同时选择训练 step 更早者。

若没有 checkpoint 同时满足 schema 合法率和 FPR 门槛，则本轮不解锁 Test。此时只输出 Dev 的 Pareto 结果和错误分析，决定是否调整训练或生成协议；不得用 Test 协助选择。

trainer `eval_loss` 和 `eval_token_acc` 不参与上述排序。

## 验证与失败处理

- 运行前验证 Dev 恰好 200 行、GOOD 149、BAD 51，图片均存在，数据 SHA-256 固定。
- 每个 checkpoint 必须恰好产生 200 条且样本顺序/标识与 Dev 一一对应；缺行、重复行或错位均使该 checkpoint 评测失败。
- 单个样本生成失败不得偷偷重试为不同解码协议；记录失败后可用同一参数做一次技术性重跑，并在审计清单中注明。
- checkpoint 任务异常退出时保留临时产物，但只有完成 200/200 且汇总校验通过后才能标记完成。
- 选择结果锁定后，保存最终配置与摘要；之后才允许对固定 Test 运行一次完全相同的协议。
