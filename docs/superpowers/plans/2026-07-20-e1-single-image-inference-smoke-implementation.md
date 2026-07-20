# E1 Single-Image Inference Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用服务器已安装的 ms-swift 4.4.1，以 PoC `checkpoint-10` 对 PoC JSONL 第一张图片完成一次可审计的原生 Transformers 生成式推理。

**Architecture:** 本计划只解决单图推理契约，不启动正式 Dev。先从服务器本机 `swift infer --help` 验证参数名，再复制 PoC 第一条记录作为一次性输入，最后用固定、确定性的生成参数运行并保存原始结果和日志。冒烟输出确认后，另建批量 Dev runner 与严格评测器实施计划。

**Tech Stack:** Bash、ms-swift 4.4.1 `swift infer`、Transformers backend、Qwen3.5-27B、PEFT LoRA、Python 3 标准库

---

## 文件与产物结构

本阶段不修改训练数据、checkpoint 或 Test，只在服务器新建一个独立冒烟目录：

```text
/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/
└── smoke_poc_checkpoint10/
    ├── swift-infer-help.txt
    ├── input-first.jsonl
    ├── input-summary.json
    ├── infer.log
    ├── raw-result.jsonl
    └── sha256sums.txt
```

`swift-infer-help.txt` 固定本机 CLI 契约；`input-first.jsonl` 是 PoC 原文件第一条的只读副本；`raw-result.jsonl` 保存未经清洗的原始推理结果；`infer.log` 保存完整运行日志。

### Task 1: 验证本机 CLI 契约

**Files:**
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/swift-infer-help.txt`

- [ ] **Step 1: 定义路径并执行只读前置检查**

Run:

```bash
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
POC=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e1_poc20_v1/poc20.jsonl
CKPT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e1_poc20_r16_v1/v0-20260717-150947/checkpoint-10
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

command -v swift
test -r "$MODEL/config.json"
test -r "$POC"
test -r "$CKPT/adapter_config.json"
ls "$CKPT"/adapter_model*.safetensors >/dev/null

echo "PREFLIGHT_CHECK: PASS"
```

Expected: 输出当前 Conda 环境中的 `swift` 路径，四项文件检查均以状态 0 完成，最后一行严格为 `PREFLIGHT_CHECK: PASS`。

- [ ] **Step 2: 保存完整帮助并验证本计划使用的参数名**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

mkdir -p "$SMOKE"
if [[ ! -s "$SMOKE/swift-infer-help.txt" ]]; then
    swift infer --help > "$SMOKE/swift-infer-help.txt" 2>&1
fi

for ARG in \
    --model \
    --adapters \
    --val_dataset \
    --add_non_thinking_prefix \
    --torch_dtype \
    --attn_impl \
    --infer_backend \
    --max_new_tokens \
    --temperature \
    --stream \
    --max_batch_size \
    --result_path; do
    grep -q -- "$ARG" "$SMOKE/swift-infer-help.txt" || {
        echo "MISSING_CLI_ARG: $ARG"
        exit 2
    }
done

echo "CLI_CONTRACT_CHECK: PASS"
```

Expected: 最后一行严格为 `CLI_CONTRACT_CHECK: PASS`。出现任何 `MISSING_CLI_ARG` 时立即停止，不执行后续推理，并根据本机 help 修订命令。

### Task 2: 构造并验证单条冒烟输入

**Files:**
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/input-first.jsonl`
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/input-summary.json`

- [ ] **Step 1: 复制 PoC 第一条记录**

Run:

```bash
POC=/home/data/h30082292/data/pose/artifact_detection_training/ms_swift/e1_poc20_v1/poc20.jsonl
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

for PATH_TO_CHECK in "$SMOKE/input-first.jsonl" "$SMOKE/input-summary.json"; do
    if [[ -e "$PATH_TO_CHECK" ]]; then
        echo "ERROR: refusing to overwrite existing artifact: $PATH_TO_CHECK"
        exit 4
    fi
done

head -n 1 "$POC" > "$SMOKE/input-first.jsonl"
```

Expected: `input-first.jsonl` 恰好一行；原始 `poc20.jsonl` 不发生变化。

- [ ] **Step 2: 验证输入协议并记录摘要**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

python - "$SMOKE/input-first.jsonl" "$SMOKE/input-summary.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
lines = source.read_text(encoding="utf-8-sig").splitlines()
assert len(lines) == 1, f"expected 1 row, got {len(lines)}"

row = json.loads(lines[0])
assert set(row) == {"images", "messages"}
assert len(row["images"]) == 1
assert Path(row["images"][0]).is_file()
assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]

expected_system = (
    "你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。"
    "严格只输出指定JSON，不要添加分析、解释或Markdown。"
)
expected_user = (
    "<image>\n检查这张图片。输出decision、categories和reasons。"
    "decision只能是GOOD或BAD。"
)
assert row["messages"][0]["content"] == expected_system
assert row["messages"][1]["content"] == expected_user

gold = json.loads(row["messages"][2]["content"])
assert gold["decision"] in {"GOOD", "BAD"}

summary = {
    "rows": 1,
    "image": row["images"][0],
    "roles": [m["role"] for m in row["messages"]],
    "gold": gold,
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("SMOKE_INPUT_CHECK: PASS")
PY
```

Expected: 输出一条现存图片路径、`system/user/assistant` 三个角色、合法 gold JSON，最后一行为 `SMOKE_INPUT_CHECK: PASS`。

### Task 3: 运行单图原生生成式推理

**Files:**
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/infer.log`
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/raw-result.jsonl`

- [ ] **Step 1: 检查 GPU 4 当前可用显存**

Run:

```bash
FREE=$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
echo "GPU 4 free: ${FREE} MiB"
if [[ "$FREE" -lt 70000 ]]; then
    echo "ERROR: GPU 4 has less than 70000 MiB free"
    exit 5
fi
echo "GPU_PREFLIGHT_CHECK: PASS"
```

Expected: GPU 4 空闲显存不少于 70000 MiB，最后一行严格为 `GPU_PREFLIGHT_CHECK: PASS`。本步骤不加载模型。

- [ ] **Step 2: 拒绝覆盖既有冒烟输出**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

for PATH_TO_CHECK in "$SMOKE/infer.log" "$SMOKE/raw-result.jsonl"; do
    if [[ -e "$PATH_TO_CHECK" ]]; then
        echo "ERROR: refusing to overwrite existing artifact: $PATH_TO_CHECK"
        exit 4
    fi
done

echo "OUTPUT_OVERWRITE_CHECK: PASS"
```

Expected: 最后一行严格为 `OUTPUT_OVERWRITE_CHECK: PASS`。若发现已有产物则停止并保留现场，不覆盖或删除。

- [ ] **Step 3: 使用固定参数运行一次推理**

Run:

```bash
MODEL=/home/data/h30082292/DATA_71/public/models/Qwen3.5-27B
CKPT=/home/data/h30082292/data/pose/artifact_detection_training/runs/e1_poc20_r16_v1/v0-20260717-150947/checkpoint-10
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

export CUDA_VISIBLE_DEVICES=4
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -o pipefail

swift infer \
    --model "$MODEL" \
    --adapters "$CKPT" \
    --val_dataset "$SMOKE/input-first.jsonl" \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --infer_backend transformers \
    --max_new_tokens 128 \
    --temperature 0 \
    --stream false \
    --max_batch_size 1 \
    --result_path "$SMOKE/raw-result.jsonl" \
    2>&1 | tee "$SMOKE/infer.log"
```

Expected: 基座模型和 `checkpoint-10` adapter 成功加载，只处理一条样本，进程状态为 0，无 OOM、NaN、template 或图像加载错误，并生成非空 `raw-result.jsonl`。

- [ ] **Step 4: 检查日志中的阻断错误**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

test -s "$SMOKE/infer.log"
test -s "$SMOKE/raw-result.jsonl"

if grep -Eqi 'traceback \(most recent call last\)|cuda out of memory|torch\.outofmemoryerror|cuda error:|\[error:swift\]|filenotfounderror' "$SMOKE/infer.log"; then
    echo "SMOKE_LOG_CHECK: FAILED"
    exit 3
fi

echo "SMOKE_LOG_CHECK: PASS"
```

Expected: 最后一行严格为 `SMOKE_LOG_CHECK: PASS`。

### Task 4: 审计原始结果并固化哈希

**Files:**
- Create: `/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10/sha256sums.txt`

- [ ] **Step 1: 验证结果是一条可读取 JSON 记录并显示所有顶层字段**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

python - "$SMOKE/raw-result.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
assert len(lines) == 1, f"expected exactly 1 result row, got {len(lines)}"
row = json.loads(lines[0])
assert isinstance(row, dict), type(row).__name__

print("result_keys:", sorted(row))
print(json.dumps(row, ensure_ascii=False, indent=2))
print("RAW_RESULT_STRUCTURE_CHECK: PASS")
PY
```

Expected: 输出结果的全部顶层字段和完整原始记录，最后一行为 `RAW_RESULT_STRUCTURE_CHECK: PASS`。本步骤不猜测生成文本所在字段，也不清洗输出；字段结构将作为下一阶段评测器设计的唯一依据。

- [ ] **Step 2: 验证固定非思考包络、payload JSON、schema 和 gold 分离**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

python - "$SMOKE/raw-result.jsonl" <<'PY'
import json
import sys
from pathlib import Path

prefix = "<think>\n\n</think>\n\n"
path = Path(sys.argv[1])
lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
assert len(lines) == 1
row = json.loads(lines[0])

assert isinstance(row.get("response"), str)
assert isinstance(row.get("labels"), str)
assert row["messages"][-1] == {"role": "assistant", "content": row["response"]}

raw = row["response"].strip()
assert raw.startswith(prefix), repr(raw[:40])
assert raw.count("<think>") == 1
assert raw.count("</think>") == 1

payload_text = raw[len(prefix):].strip()
payload = json.loads(payload_text)
gold = json.loads(row["labels"])

def validate_schema(value):
    assert isinstance(value, dict)
    assert set(value) == {"decision", "categories", "reasons"}
    assert value["decision"] in {"GOOD", "BAD"}
    assert isinstance(value["categories"], list)
    assert isinstance(value["reasons"], list)
    assert all(isinstance(item, str) and item.strip() for item in value["categories"])
    assert all(isinstance(item, str) and item.strip() for item in value["reasons"])
    if value["decision"] == "GOOD":
        assert value["categories"] == []
        assert value["reasons"] == []
    else:
        assert 1 <= len(value["categories"]) <= 3
        assert 1 <= len(value["reasons"]) <= 3

validate_schema(payload)
validate_schema(gold)

print("prediction:", json.dumps(payload, ensure_ascii=False))
print("gold:", json.dumps(gold, ensure_ascii=False))
print("label_leakage_check:", payload != gold)
print("STRICT_ENVELOPE_JSON_SCHEMA_CHECK: PASS")
PY
```

Expected: prediction 为 BAD、gold 为 GOOD、`label_leakage_check: True`，最后一行严格为 `STRICT_ENVELOPE_JSON_SCHEMA_CHECK: PASS`。

- [ ] **Step 3: 固化输入、结果、日志和 CLI 帮助的 SHA-256**

Run:

```bash
SMOKE=/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_dev_v1/smoke_poc_checkpoint10

cd "$SMOKE"
if [[ -e sha256sums.txt ]]; then
    echo "ERROR: refusing to overwrite existing artifact: $SMOKE/sha256sums.txt"
    exit 4
fi
sha256sum \
    swift-infer-help.txt \
    input-first.jsonl \
    input-summary.json \
    infer.log \
    raw-result.jsonl \
    > sha256sums.txt
sha256sum -c sha256sums.txt
```

Expected: 五个文件均报告 `OK`。

## 完成门槛

只有以下条件全部满足，单图冒烟测试才算通过：

1. `CLI_CONTRACT_CHECK: PASS`；
2. `SMOKE_INPUT_CHECK: PASS`；
3. `swift infer` 状态为 0；
4. `SMOKE_LOG_CHECK: PASS`；
5. `RAW_RESULT_STRUCTURE_CHECK: PASS`；
6. `STRICT_ENVELOPE_JSON_SCHEMA_CHECK: PASS`；
7. `sha256sum -c` 五项全部为 `OK`；
8. 原始结果中能区分模型生成响应与数据集 gold label，确认 gold assistant 未作为输入续写。

如第 8 项不能从实际结果结构中确认，则停止，不启动正式 Dev；下一步检查 ms-swift 4.4.1 的本地源码中 dataset-to-infer-request 转换逻辑。

## 版本控制说明

本地项目目录当前不是 Git 仓库，因此本计划不包含 commit 操作。计划文件和后续代码改动通过明确路径、测试输出及 SHA-256 审计；不得为了满足流程临时初始化 Git 仓库。
