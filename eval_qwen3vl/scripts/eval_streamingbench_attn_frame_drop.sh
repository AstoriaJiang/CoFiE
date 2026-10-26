#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ---- Data ----
# TASK_CSV="/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
TASK_CSV="/data/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding_5pct.csv"
VIDEO_DIR="${VIDEO_DIR:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding}"

# ---- Model ----
CKPT_PATH="${CKPT_PATH:-ckpt/Qwen3-VL-4B-Instruct}"
RUN_NAME="${RUN_NAME:-attn_v2_frame_drop_50}"
RESULT_DIR="${RESULT_DIR:-eval_qwen3vl/result_streamingbench}"

# ---- Attention frame-drop (single-pass, no re-prefill) ----
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"      # LM layer index (negative OK)
TOP_K_RATIO="${TOP_K_RATIO:-0.5}"            # fraction of frames to KEEP

python eval_qwen3vl/scripts/streamingbench_attn_frame_drop.py \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --attn_layer_idx "$ATTN_LAYER_IDX" \
    --top_k_ratio "$TOP_K_RATIO"

    # --question_id "Real-Time Visual Understanding_sample_124_1"
