#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/eval_qwen3vl/lmms-eval:${REPO_ROOT}/eval_qwen3vl:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASKS="${TASKS:-mvbench}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
CLIP_PATH="${CLIP_PATH:-${REPO_ROOT}/ckpt/clip-vit-base-patch32}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval_qwen3vl/result_clip}"
RUN_NAME="${RUN_NAME:-qwen3vl_clip_drop20}"
STATS_PATH="${STATS_PATH:-${OUTPUT_PATH}/${RUN_NAME}_clip_latency.jsonl}"

CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
CLIP_DTYPE="${CLIP_DTYPE:-float16}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"
CLIP_STRATEGY="${CLIP_STRATEGY:-relative}"
CLIP_DROP_RATIO="${CLIP_DROP_RATIO:-0.2}"
CLIP_SIMILARITY_THRESHOLD="${CLIP_SIMILARITY_THRESHOLD:-0.95}"
CLIP_MAX_INTERVAL="${CLIP_MAX_INTERVAL:-30}"
MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-4}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-1000}"
FPS="${FPS:-1}"

ENABLE_ATTN="${ENABLE_ATTN:-0}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.375}"
TOP_K_RATIO="${TOP_K_RATIO:-$(awk "BEGIN {print 1 - ${ATTN_DROP_RATIO}}")}" 
DROP_METHOD=none
if [[ "${ENABLE_ATTN}" == "1" || "${ENABLE_ATTN}" == "true" ]]; then
    DROP_METHOD=attention
fi

mkdir -p "${OUTPUT_PATH}"
rm -f "${STATS_PATH}"

MODEL_ARGS="pretrained=${CKPT_PATH},clip_model_path=${CLIP_PATH},clip_device=${CLIP_DEVICE},clip_dtype=${CLIP_DTYPE},clip_batch_size=${CLIP_BATCH_SIZE},clip_strategy=${CLIP_STRATEGY},clip_drop_ratio=${CLIP_DROP_RATIO},clip_similarity_threshold=${CLIP_SIMILARITY_THRESHOLD},clip_min_keep_frames=${MIN_NUM_FRAMES},clip_max_interval=${CLIP_MAX_INTERVAL},clip_stats_path=${STATS_PATH},drop_method=${DROP_METHOD},attn_layer_idx=${ATTN_LAYER_IDX},top_k_ratio=${TOP_K_RATIO},max_num_frames=${MAX_NUM_FRAMES},min_num_frames=${MIN_NUM_FRAMES},fps=${FPS},attn_implementation=flash_attention_2"

echo "TASKS=${TASKS}"
echo "MODEL_ARGS=${MODEL_ARGS}"
SECONDS=0
accelerate launch --num_processes 1 -m lmms_eval \
    --model qwen3_vl_clip \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --verbosity=INFO
WALL_SECONDS="${SECONDS}"

python -m qwen3_vl_clip.summarize_latency "${STATS_PATH}" \
    --output "${OUTPUT_PATH}/${RUN_NAME}_clip_latency_summary.json"
echo "End-to-end wall time: ${WALL_SECONDS}s"
