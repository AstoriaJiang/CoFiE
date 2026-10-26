#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

TASK_CSV="/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding"

RUN_NAME=8B_feature_T0_r30
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_streamingbench"

TEMPERATURE=0

# DTD arguments
DROP_METHOD=feature     # feature or pixel or none
DROP_THRESHOLD=0.3



python eval_qwen3vl/scripts/qwen3_vl/streamingbench.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --drop_relative \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"\
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE"
