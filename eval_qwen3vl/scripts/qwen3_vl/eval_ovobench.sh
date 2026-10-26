#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,2,3,4

TASK_JSON="./datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR="./datasets/OVO-Bench/src_videos"

RUN_NAME=8B_none
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_ovobench"

# DTD arguments
DROP_METHOD=none     # feature or pixel or none
DROP_THRESHOLD=0
TEMPERATURE=0

python eval_qwen3vl/scripts/qwen3_vl/ovobench.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --drop_relative \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE"
