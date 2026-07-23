# E4 Crop PoC 视觉复核页面

该页面只复核 Train 生成的 20 条双图 PoC，不读取或修改 Dev/Test，也不修改原始图片、bbox、Crop 或训练 JSONL。

## 启动服务

```bash
cd /home/data/h30082292/code/sft
bash scripts/run_e4_crop_review_web.sh
```

服务默认监听服务器 `127.0.0.1:8877`，终端会打印带随机 token 的完整 URL。保持该终端运行。

## Windows 建立 SSH 转发

在本地 PowerShell 使用不同的本地端口，避免此前 8765 的占用或权限问题：

```powershell
ssh -N -L 18877:127.0.0.1:8877 h00484736@10.50.113.56
```

将服务器打印 URL 中的端口 `8877` 改为本地端口 `18877`，保留完整 token，例如：

```text
http://127.0.0.1:18877/?token=服务器打印的随机token
```

## 复核规则

- 红框：人工异常 bbox；蓝框：实际 Crop 范围。
- `通过`：bbox、Crop 和上下文均可直接进入正式数据。
- `有问题`：至少选择一个问题类型并填写必要说明。
- `不确定`：当前无法可靠判断，正式构建前单独处理。
- 点击图片可切换原始尺寸；方向键切换样本；`Ctrl+Enter` 保存。

复核结果独立保存在：

```text
/home/data/h30082292/data/pose/artifact_detection_training/evaluations/e4_crop_aux_v1/token_preflight_v1/crop-review-v1
```

- `annotations.json`：权威复核记录；
- `reviewed.csv`：便于查看和归档的完整表格。

全部 20 条完成后，停止服务并运行后续完成度检查。正式 Crop 数据只能在复核结论通过后构建。
