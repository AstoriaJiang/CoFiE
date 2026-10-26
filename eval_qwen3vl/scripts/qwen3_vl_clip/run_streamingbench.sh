#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/eval_qwen3vl:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR="${QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR:-1}"

RUN_NAME="${RUN_NAME:-8B_clip_drop20}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
TASK_CSV="${TASK_CSV:-${REPO_ROOT}/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/datasets/StreamingBench/Real-Time_Visual_Understanding}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/eval_qwen3vl/result_streamingbench}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/log-NGFF_ablation}"
QUESTION_ID="${QUESTION_ID:-}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

CLIP_PATH="${CLIP_PATH:-${REPO_ROOT}/ckpt/clip-vit-base-patch32}"
CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
CLIP_DTYPE="${CLIP_DTYPE:-float16}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"
CLIP_STRATEGY="${CLIP_STRATEGY:-relative}"
CLIP_DROP_RATIO="${CLIP_DROP_RATIO:-0.2}"
CLIP_SIMILARITY_THRESHOLD="${CLIP_SIMILARITY_THRESHOLD:-0.95}"
CLIP_MIN_KEEP_FRAMES="${CLIP_MIN_KEEP_FRAMES:-4}"
CLIP_MAX_INTERVAL="${CLIP_MAX_INTERVAL:-30}"

FPS="${FPS:-1}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
TEMPERATURE="${TEMPERATURE:-0.7}"
ENABLE_ATTN="${ENABLE_ATTN:-0}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.375}"
TOP_K_RATIO="${TOP_K_RATIO:-$(awk "BEGIN {print 1 - ${ATTN_DROP_RATIO}}")}" 

DROP_METHOD=none
if [[ "${ENABLE_ATTN}" == "1" || "${ENABLE_ATTN}" == "true" ]]; then
    DROP_METHOD=attention
fi

QUESTION_ARGS=()
if [[ -n "${QUESTION_ID}" ]]; then
    QUESTION_ARGS+=(--question-id "${QUESTION_ID}")
fi

python -m qwen3_vl_clip.streamingbench \
    --run-name "${RUN_NAME}" \
    --ckpt-path "${CKPT_PATH}" \
    --task-csv "${TASK_CSV}" \
    --video-dir "${VIDEO_DIR}" \
    --result-dir "${RESULT_DIR}" \
    --log-dir "${LOG_DIR}" \
    "${QUESTION_ARGS[@]}" \
    --sample-fraction "${SAMPLE_FRACTION}" \
    --sample-seed "${SAMPLE_SEED}" \
    --temperature "${TEMPERATURE}" \
    --max-frames "${MAX_FRAMES}" \
    --fps "${FPS}" \
    --clip-model-path "${CLIP_PATH}" \
    --clip-device "${CLIP_DEVICE}" \
    --clip-dtype "${CLIP_DTYPE}" \
    --clip-batch-size "${CLIP_BATCH_SIZE}" \
    --clip-strategy "${CLIP_STRATEGY}" \
    --clip-drop-ratio "${CLIP_DROP_RATIO}" \
    --clip-similarity-threshold "${CLIP_SIMILARITY_THRESHOLD}" \
    --clip-min-keep-frames "${CLIP_MIN_KEEP_FRAMES}" \
    --clip-max-interval "${CLIP_MAX_INTERVAL}" \
    --drop-method "${DROP_METHOD}" \
    --attn-layer-idx "${ATTN_LAYER_IDX}" \
    --top-k-ratio "${TOP_K_RATIO}"
