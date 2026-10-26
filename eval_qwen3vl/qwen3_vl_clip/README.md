# Qwen3-VL + CLIP frame filtering

这是给审稿人补充的 model-based filtering 基线。它只替换原流水线第一阶段的
HDiff/histogram 帧筛选，Qwen3-VL 推理、prompt、答案解析和可选的第二阶段 attention
token drop 均保持不变。

## 相似度定义

对每帧用 `ckpt/clip-vit-base-patch32` 的 vision encoder 提取最后一层 patch hidden
states（去掉 CLS）。相邻帧的相似度为相同空间位置 patch cosine similarity 的平均：

```text
S(F[t-1], F[t]) = mean_p cosine(z[t-1,p], z[t,p])
```

默认 `relative` 策略固定 drop ratio：保留第 0 帧，并按相邻帧相似度由低到高保留
变化最大的帧。这适合与 HDiff 在相同帧预算下公平比较。`threshold` 策略则在相似度
低于阈值时保留当前帧。

## 运行

### StreamingBench（论文主结果）

该入口会在 `process_vision_info` 解码并按 FPS 采样之后，显式执行一次 CLIP 筛帧，
再把保留帧交给原有 Qwen3-VL 推理流程。因此它只替换第一阶段，不会被 Qwen processor
再次采样或过滤。

```bash
RUN_NAME=8B_clip20_streamingbench \
CLIP_DROP_RATIO=0.2 \
FPS=1 MAX_FRAMES=1000 \
bash eval_qwen3vl/scripts/qwen3_vl_clip/run_streamingbench.sh
```

先做单样本端到端 smoke test 时，设置 CSV 中的 `question_id`：

```bash
QUESTION_ID='<question_id>' bash eval_qwen3vl/scripts/qwen3_vl_clip/run_streamingbench.sh
```

叠加 attention 第二阶段：

```bash
RUN_NAME=8B_clip20_attn375_streamingbench \
CLIP_DROP_RATIO=0.2 \
ENABLE_ATTN=1 ATTN_DROP_RATIO=0.375 \
bash eval_qwen3vl/scripts/qwen3_vl_clip/run_streamingbench.sh
```

每次运行会在 `eval_qwen3vl/result_streamingbench/summary/` 写入一个 JSON，包含
overall/各任务类型 Accuracy、端到端 mean/p50/p95 latency、CLIP 的预处理/forward/选择
分项 latency、保留帧比例和所有结果文件路径。`error/` 中的样本不会进入 Accuracy 分母。

### lmms-eval（其他视频基准）

```bash
TASKS=mvbench \
RUN_NAME=8B_clip_drop20 \
CLIP_DROP_RATIO=0.2 \
bash eval_qwen3vl/scripts/qwen3_vl_clip/run_lmms_eval.sh
```

叠加 attention 第二阶段：

```bash
TASKS=mvbench \
RUN_NAME=8B_clip20_attn375 \
CLIP_DROP_RATIO=0.2 \
ENABLE_ATTN=1 \
ATTN_DROP_RATIO=0.375 \
bash eval_qwen3vl/scripts/qwen3_vl_clip/run_lmms_eval.sh
```

`TASKS` 可替换为当前 lmms-eval 支持的任务，如 `mvbench`、`mlvu_dev_chat`、
`longvideobench_val_v`、`egoschema_subset`。lmms-eval 的结果 JSON 给出 Accuracy；
`${RUN_NAME}_clip_latency_summary.json` 给出 CLIP filtering 的 mean/p50/p95 latency、
forward time、预处理时间和实际保留帧比例。

论文比较时应让 HDiff 和 CLIP 使用相同 `drop_ratio`、FPS、最大帧数、生成参数和硬件。
StreamingBench 的 summary 同时给出端到端 latency 和 CLIP filter latency；其中后者包含
CLIP resize/normalize、vision forward 和相似度/选择，不包含视频解码与 Qwen 推理。首次
模型加载不计入逐视频 latency。

## 测试

```bash
source eval_qwen3vl/.venv/bin/activate
PYTHONPATH=eval_qwen3vl python -m unittest qwen3_vl_clip.test_frame_selector
```
