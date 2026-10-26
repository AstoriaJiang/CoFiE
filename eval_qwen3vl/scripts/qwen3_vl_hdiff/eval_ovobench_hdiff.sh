#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,2,3,4

TASK_JSON="./datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR="./datasets/OVO-Bench/src_videos"

RUN_NAME=8B_hdiff
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_ovobench"

DROP_METHOD=feature
DROP_THRESHOLD=0.5
TEMPERATURE=0
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}" # 1 时使用相对筛选：按相邻帧 hist 差排序，保留 top k；HIST_THRESHOLD 表示丢弃比例。

HIST_ENABLED=1
HIST_BINS=32
HIST_THRESHOLD=0.15
HIST_MIN_KEEP_FRAMES=4
HIST_MAX_INTERVAL=30

HIST_ARGS=()
if [[ "${HIST_ENABLED}" == "1" ]]; then
    HIST_ARGS+=(--hist_enabled)
fi

DROP_ARGS=()
if [[ "${DROP_RELATIVE_HDIFF}" == "1" || "${DROP_RELATIVE_HDIFF}" == "true" || "${DROP_RELATIVE_HDIFF}" == "True" ]]; then
    DROP_ARGS+=(--drop_relative_hdiff)
fi

python eval_qwen3vl/scripts/qwen3_vl_hdiff/ovobench_hdiff.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE" \
    "${DROP_ARGS[@]}" \
    "${HIST_ARGS[@]}" \
    --hist_bins $HIST_BINS \
    --hist_threshold $HIST_THRESHOLD \
    --hist_min_keep_frames $HIST_MIN_KEEP_FRAMES \
    --hist_max_interval $HIST_MAX_INTERVAL
