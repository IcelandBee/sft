# E1 / E2 Dev 决策边界复核网页

该网页只用于人工复核 E1/E2 的 57 个重点 Dev 样本，不读取固定 Test，也不会修改原始 Dev、推理结果或训练数据。

## 启动

先生成审计清单，再启动网页：

```bash
cd /home/data/h30082292/code/sft
bash scripts/run_e1_e2_dev_audit.sh
bash scripts/run_e1_e2_dev_audit_web.sh
```

服务默认只监听服务器的 `127.0.0.1:8765`。启动日志会打印带随机访问 token 的完整 URL。

在本地电脑另开终端建立 SSH 转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 h00484736@127A100
```

保持两个终端运行，在本地浏览器打开服务日志打印的完整 URL。若本地端口 8765 已占用，可在两侧统一换成其他端口。

## 复核字段

- 原 Gold 标注：正确、错误、不确定。
- 可见异常程度：明显、边界、无异常、不确定。
- 图片级复核结论：GOOD、BAD、UNSURE。
- 主要异常类别和自由文本备注为选填。

前三项全部填写后，该样本才计入完成进度。页面支持筛选、上一张/下一张、原图缩放、快捷键和 CSV 导出。

## 保存产物

默认保存在：

```text
/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e1_e2_dev_boundary_audit_v1/
```

- `annotations.json`：网页的权威增量标注文件，每次保存原子替换。
- `reviewed.csv`：每次保存同步生成的完整表格，UTF-8 BOM，可直接用 Excel 打开。

停止或重启服务不会丢失已经保存的内容。Dev 复核结果只用于标签诊断、E3 方案设计和模型选择，不能回灌训练集。
