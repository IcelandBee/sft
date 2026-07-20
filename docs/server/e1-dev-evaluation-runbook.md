# E1 Dev 批量评测服务器执行说明

## 工具文件

把以下三个文件放在服务器同一目录中：

```text
evaluate_e1_dev.py
select_e1_checkpoint.py
run_e1_dev_checkpoints.py
```

推荐服务器目录：

```text
/home/data/h30082292/code/scripts/Qwen35_27B/e1_dev_eval_v1
```

## 第一步：只执行 dry-run

dry-run 会检查：

- 当前环境能导入 ms-swift 并记录版本；
- 固定 Dev 恰好 200 行、GOOD 149、BAD 51；
- 200 张图片全部存在且顺序唯一；
- system/user prompt 和 gold JSON 未漂移；
- 8 个 checkpoint 的 adapter 配置和权重均存在；
- GPU 4–7 各至少 70000 MiB 空闲；
- 8 条推理命令除 adapter、结果路径和 GPU 分配外协议一致；
- 输出根尚不存在。

dry-run 不加载模型、不占用显存，也不会创建正式输出根，只写一份审计 manifest。

```bash
TOOL_DIR=/home/data/h30082292/code/scripts/Qwen35_27B/e1_dev_eval_v1
OUTPUT_ROOT=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1
DRY_MANIFEST=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/e1_broad_clean_8ckpt_v1-dry-run-manifest.json

python "$TOOL_DIR/run_e1_dev_checkpoints.py" \
    --output-root "$OUTPUT_ROOT" \
    --dry-run \
    --manifest-path "$DRY_MANIFEST"
```

预期最后出现：

```text
DRY_RUN_CHECK: PASS
```

在 dry-run 输出和 manifest 未审阅前，不执行正式命令。

## 正式执行设计

正式模式把 checkpoint 固定分为四个单卡队列：

| GPU | checkpoint |
|---:|---|
| 4 | 312 → 1560 |
| 5 | 624 → 1872 |
| 6 | 936 → 2184 |
| 7 | 1248 → 2496 |

每张 GPU 同时只加载一个 checkpoint，四张 GPU 并行。每个 checkpoint 完成后立即严格评测；只有 8 个结果都完整时才生成 `checkpoint-summary.json`。runner 不接受 Test 参数。
