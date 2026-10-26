#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

export TASK_JSON="./datasets/OVO-Bench/ovo_bench_new.json"
export VIDEO_DIR="./datasets/OVO-Bench/src_videos"

RUN_NAME=8B_attn_v2_50
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_ovobench"

# ---- Attention frame-drop (v2: chunked scorer, flash/sdpa) ----
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
TOP_K_RATIO="${TOP_K_RATIO:-0.5}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

args=(
    --run_name "$RUN_NAME"
    --ckpt_path "$CKPT_PATH"
    --task_json "$TASK_JSON"
    --video_dir "$VIDEO_DIR"
    --result_dir "$RESULT_DIR"
    --attn_layer_idx "$ATTN_LAYER_IDX"
    --top_k_ratio "$TOP_K_RATIO"
    --scorer_chunk_size "$SCORER_CHUNK_SIZE"
    --attn_implementation "$ATTN_IMPL"
    --max_frames 1000
    --fps 1
)
python eval_qwen3vl/scripts/qwen3_vl_attn_v2/ovobench_attn_v2.py "${args[@]}"
