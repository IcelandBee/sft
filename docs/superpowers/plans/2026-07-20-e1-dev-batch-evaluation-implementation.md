# E1 Dev Batch Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可测试、可审计的 E1 Dev 严格评测器、checkpoint 选择器和 4×A100 并行推理 runner，对 8 个正式 checkpoint 使用同一协议评测固定 200 张 Dev。

**Architecture:** 纯 Python 评测器只读取 ms-swift 原始 JSONL，严格验证固定空 think 包络、payload JSON 和 schema，并保守计分 INVALID。选择器只读取各 checkpoint 的 `metrics.json`，按预注册 Dev 规则选模型。runner 以 GPU 4–7 启动最多四个独立单卡 `swift infer` 进程，两波完成 8 个 checkpoint；每卡 batch size 1，checkpoint 间除 adapter 路径外参数完全一致。

**Tech Stack:** Python 3.12 标准库、unittest、ms-swift 4.4.1、Transformers backend、PEFT LoRA、4×A100 80GB

---

## 文件结构

- Create: `scripts/evaluate_e1_dev.py` — 严格解析单个 checkpoint 原始结果并生成逐图记录、错误清单和指标。
- Create: `tests/test_evaluate_e1_dev.py` — 包络、JSON、schema、INVALID 保守计分、重复图片和原子输出测试。
- Create: `scripts/select_e1_checkpoint.py` — 汇总 8 份指标并执行预注册 checkpoint 选择规则。
- Create: `tests/test_select_e1_checkpoint.py` — 门槛、排序、并列和无合格 checkpoint 测试。
- Create: `scripts/run_e1_dev_checkpoints.py` — 远端前置检查、dry-run 命令审计、4 GPU 并行调度和完成校验。
- Create: `tests/test_run_e1_dev_checkpoints.py` — 命令一致性、GPU 分配、路径拒绝覆盖和 dry-run manifest 测试。

### Task 1: 严格评测器

**Files:**
- Create: `tests/test_evaluate_e1_dev.py`
- Create: `scripts/evaluate_e1_dev.py`

- [ ] **Step 1: 写包络与 schema 的失败测试**

测试直接调用以下目标接口：

```python
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"

def parse_prediction(raw: object) -> dict:
    """Return status, payload text, parsed payload and precise error code."""

def validate_payload(value: object) -> str | None:
    """Return None for valid schema, otherwise a stable error code."""
```

覆盖：精确空 think 前缀成功；无前缀、非空 think、重复前缀、Markdown、尾随文本、非法 JSON、额外字段、GOOD 非空辅助数组、BAD 空辅助数组、非字符串数组项均失败。

- [ ] **Step 2: 运行测试确认 RED**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_evaluate_e1_dev.py`

Expected: collection 失败，原因是 `scripts.evaluate_e1_dev` 尚不存在。

- [ ] **Step 3: 最小实现包络、JSON 和 schema 解析**

实现稳定状态字段：`envelope_valid`、`raw_direct_json_valid`、`payload_json_valid`、`schema_valid`、`decision`、`error_code`。只允许移除精确 `NON_THINKING_PREFIX`，不做正则抽取或 JSON 修复。

- [ ] **Step 4: 运行解析测试确认 GREEN**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_evaluate_e1_dev.py`

Expected: 包络与 schema 测试全部通过。

- [ ] **Step 5: 写混淆矩阵和端到端输出的失败测试**

目标接口：

```python
def evaluate_rows(rows: list[dict], expected_count: int) -> tuple[list[dict], dict]:
    """Evaluate rows in source order; INVALID is always counted as wrong."""

def run_evaluation(result_path: Path, output_dir: Path, expected_count: int = 200) -> dict:
    """Atomically write parsed.jsonl, errors.jsonl and metrics.json."""
```

测试断言：gold BAD 的 INVALID 计 FN；gold GOOD 的 INVALID 计 FP；TP+FN+FP+TN 等于总数；重复图片、行数错误、非法 gold、response 与最后 assistant 不一致均拒绝；完整输出只有三个确定性文件。

- [ ] **Step 6: 运行新增测试确认 RED**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_evaluate_e1_dev.py`

Expected: 因 `evaluate_rows` 或 `run_evaluation` 未实现而失败。

- [ ] **Step 7: 实现指标与原子输出**

指标必须包含 `tp/fn/fp/tn`、Recall、FPR、Accuracy、Precision、F1、固定包络合法率、payload JSON 可解析率、原始全文直接 JSON 可解析率、schema 合法率及 GOOD/BAD INVALID 数量。输出目录存在时拒绝覆盖；写入 staging 后一次重命名。

- [ ] **Step 8: 运行评测器测试确认 GREEN**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_evaluate_e1_dev.py`

Expected: 全部通过。

### Task 2: checkpoint 选择器

**Files:**
- Create: `tests/test_select_e1_checkpoint.py`
- Create: `scripts/select_e1_checkpoint.py`

- [ ] **Step 1: 写选择规则失败测试**

目标接口：

```python
EXPECTED_STEPS = (312, 624, 936, 1248, 1560, 1872, 2184, 2496)

def select_checkpoint(metrics: list[dict]) -> dict:
    """Apply schema>=0.995, FPR<=0.25, then Recall/Accuracy/F1/earlier-step ranking."""
```

覆盖：Recall 优先；Accuracy/F1 依次打破并列；完全并列选更早 step；缺 checkpoint、重复 step、样本数非 200 均拒绝；无合格 checkpoint 返回 `selected_step=None` 且 `test_unlocked=False`。

- [ ] **Step 2: 运行测试确认 RED**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_select_e1_checkpoint.py`

Expected: collection 失败，原因是模块尚不存在。

- [ ] **Step 3: 实现选择与汇总 CLI**

CLI 读取 `checkpoint-*/metrics.json`，输出排序后的 `checkpoint-summary.json`。trainer eval loss 不作为输入。任何不合格或缺失结果均保持 `test_unlocked=False`。

- [ ] **Step 4: 运行选择器测试确认 GREEN**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_select_e1_checkpoint.py`

Expected: 全部通过。

### Task 3: 四卡并行 runner

**Files:**
- Create: `tests/test_run_e1_dev_checkpoints.py`
- Create: `scripts/run_e1_dev_checkpoints.py`

- [ ] **Step 1: 写命令生成和 dry-run 失败测试**

目标接口：

```python
def build_swift_command(model: Path, adapter: Path, dev: Path, result: Path) -> list[str]:
    """Build the one fixed ms-swift command; only adapter/result may vary."""

def assign_jobs(steps: tuple[int, ...], gpus: tuple[int, ...]) -> list[tuple[int, int]]:
    """Assign 8 steps round-robin to GPUs 4–7."""
```

断言命令固定包含：Transformers、BF16、FlashAttention 2、`temperature=0`、`stream=false`、`max_new_tokens=128`、`max_batch_size=1`、`val_dataset_shuffle=false`、`strict=true`。dry-run 只生成 manifest，不调用 subprocess；输出根存在时拒绝覆盖。

- [ ] **Step 2: 运行测试确认 RED**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_run_e1_dev_checkpoints.py`

Expected: collection 失败，原因是模块尚不存在。

- [ ] **Step 3: 实现前置检查、manifest 和并行调度**

runner 验证 8 个 adapter、Dev 200 行/149 GOOD/51 BAD、图片存在、GPU 4–7 各至少 70000 MiB 空闲、输出根不存在。正式模式最多四个并发子进程，每个进程环境只暴露一张 GPU；失败时停止启动新任务、等待已启动任务结束并返回非零。每个成功 checkpoint 立即调用 `evaluate_e1_dev.py`。

- [ ] **Step 4: 运行 runner 测试确认 GREEN**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_run_e1_dev_checkpoints.py`

Expected: 全部通过，且测试不调用真实 GPU 或 swift。

### Task 4: 完整验证与服务器交付

- [ ] **Step 1: 运行新增测试**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q tests/test_evaluate_e1_dev.py tests/test_select_e1_checkpoint.py tests/test_run_e1_dev_checkpoints.py`

Expected: 全部通过。

- [ ] **Step 2: 运行完整回归测试**

Run: `$env:PYTHONPATH=(Get-Location).Path; pytest -q`

Expected: 既有 109 tests、40 subtests 加新增测试全部通过。

- [ ] **Step 3: 本地 dry-run 生成命令审计**

使用临时目录和伪造的 8 个 checkpoint/200 行 Dev 运行 runner `--dry-run --skip-image-existence-check`，检查 manifest 中 8 个任务、GPU 4–7 各两个、除 adapter/result 外参数一致。

- [ ] **Step 4: 服务器先执行正式 dry-run**

将三个脚本同步到服务器后执行 runner `--dry-run`。只有前置计数、图片、checkpoint、GPU、CLI 参数和 manifest 全部通过，才允许去掉 `--dry-run` 启动正式 Dev；Test 路径不作为 runner 参数提供。
