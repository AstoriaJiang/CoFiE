#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="./datasets/Video-MME/data/"

RUN_NAME=8B_none_debug
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme"

# DTD arguments
DROP_METHOD=none    # feature or pixel or none
DROP_THRESHOLD=0
TEMPERATURE=0

python eval_qwen3vl/scripts/qwen3_vl/videomme.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --drop_relative \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE"
