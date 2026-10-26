#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="/root/autodl-tmp/CoFiE/huggingface"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
TASK="longvideobench_val_v"
RUN_NAME="8B_attn_v2_longvideobench"
OUTPUT_PATH="eval_qwen3vl/result_longvideobench"

ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
TOP_K_RATIO="${TOP_K_RATIO:-0.5}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-500}"
FPS="${FPS:-1}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DROP_METHOD="${DROP_METHOD:-attention}"

# !MAX_NUM_FRAMES=500
MODEL_ARGS="pretrained=${CKPT_PATH},drop_method=${DROP_METHOD},attn_layer_idx=${ATTN_LAYER_IDX},top_k_ratio=${TOP_K_RATIO},scorer_chunk_size=${SCORER_CHUNK_SIZE},max_num_frames=${MAX_NUM_FRAMES},fps=${FPS},attn_implementation=${ATTN_IMPL}"

accelerate launch --num_processes 4 --main_process_port 29501 -m lmms_eval \
    --model qwen3_vl_attn_v2 \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --verbosity=INFO
