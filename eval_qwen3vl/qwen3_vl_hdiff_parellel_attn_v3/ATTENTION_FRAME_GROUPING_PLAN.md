# Attention 视频帧分组重构与一致性校验

## 目标

在不改变 Attention score 计算、帧级聚合、Top-K 选择和 token 删除逻辑的前提下，替换依赖 `vision_start_token_id` / `vision_end_token_id` 配对的旧分帧方式，避免不同 Transformers 版本的嵌套或重复视觉标记把时间戳、分隔符等非视频 token 错当成视频 token。

## 实现方案

1. 只把 `input_ids == config.video_token_id` 的位置视为视频 token。
2. 按序遍历视频 token，在相邻位置不连续或 MRoPE temporal ID 变化时开始新帧。
3. Attention 逻辑保持不变：从目标层重建 text-to-video Attention；每个 patch 对文本 query 取最大值；一帧内 patch score 取均值；按 `top_k_ratio` 保留分数最高的帧；删除未入选帧的全部 video token。

## 校验条件

每个样本均检查：

1. 所有 frame group 非空且索引在序列范围内。
2. group 中每个位置均为 `video_token_id`。
3. group 不重叠、组内严格递增。
4. 所有 group 的并集与实际 video token 位置完全一致。
5. 每个 group 内 token 连续且 MRoPE temporal ID 唯一。
6. 若提供 `video_grid_thw`，分组数等于 `sum(grid_t)`，每帧 token 数等于 `grid_h * grid_w / spatial_merge_size^2`，总 token 数也必须一致。
7. Attention 删除后，实际删除位置必须全是视频 token，实际删除帧/token 数必须与 Top-K 选择推导值一致。

## 失败策略和日志

- 默认仅输出警告并继续运行，校验详情写入 drop-ratio JSONL 的 `frame_validation` 字段。
- `attn_frame_validation_strict=True` 时任一校验失败立即抛出异常，供 smoke test 和回归测试使用。
- 日志记录实际/预期帧数、实际/预期视频 token 数、逐帧 token 数和失败原因。

## 测试计划

1. Python 编译检查。
2. 构造测试覆盖独立 token 段、连续 token 段、嵌套视觉标记、多视频、grid 不匹配及 Top-K 取整边界。
3. 已知问题样本 GPU smoke test：drop 50% 和 drop 60% 的实际帧/token drop 仅允许整数取整误差。
4. 对比未启用 Attention 的推理结果，确认普通路径不变。

## 边界说明

- “帧”指 Qwen3-VL 编码后的 temporal grid 单元，不保证与原始解码图像一一对应。
- 最后一层之后 drop 不会作用于后续 Transformer 层；本次不改变原有 drop 时机。
- 当前环境固定 `transformers==5.6.2`。新方法减少了对视觉边界标记格式的依赖，但仍依赖 Qwen3-VL 的 `video_token_id` 和 MRoPE temporal ID 语义。
