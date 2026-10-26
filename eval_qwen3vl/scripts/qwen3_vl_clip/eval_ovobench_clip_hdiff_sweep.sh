#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${REPO_ROOT}/ckpt/clip-vit-base-patch32}"
TASK_JSON="${TASK_JSON:-${REPO_ROOT}/eval_qwen3vl/result_ovobench/ovobench_10pct_preencoder_eval.json}"
VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/datasets/OVO-Bench/src_videos}"
RESULT_DIR="${RESULT_DIR:-eval_qwen3vl/result_ovobench}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/log-clip}"
CLIP_STATS_DIR="${CLIP_STATS_DIR:-${RESULT_DIR}/clip_stats}"
RUN_INSTANCE="${RUN_INSTANCE:-$(date +%Y%m%d_%H%M%S)}"

TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
MIN_FRAMES="${MIN_FRAMES:-4}"
FPS="${FPS:-1}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# The CLIP selector runs while the dataset is preprocessed. Keeping it in the
# main process prevents one GPU-resident CLIP model from being created per
# DataLoader worker.
CLIP_NUM_WORKERS="${CLIP_NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
CLIP_DTYPE="${CLIP_DTYPE:-float16}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"
CLIP_STRATEGY="${CLIP_STRATEGY:-relative}"
CLIP_DROP_RATIO="${CLIP_DROP_RATIO:-0.20}"
CLIP_SIMILARITY_THRESHOLD="${CLIP_SIMILARITY_THRESHOLD:-0.95}"
CLIP_MAX_INTERVAL="${CLIP_MAX_INTERVAL:-30}"

HDIFF_BINS="${HDIFF_BINS:-32}"
HDIFF_DROP_RATIO="${HDIFF_DROP_RATIO:-0.20}"
HDIFF_MAX_INTERVAL="${HDIFF_MAX_INTERVAL:-30}"

ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"

# Run one Hdiff experiment with relative 20% frame dropping.
# RUN_NAME|FRAME_SELECTOR|ENABLE_ATTN|ATTN_DROP_RATIO
EXPERIMENTS=(
    # "8B_clip_rel_drop20_attn0_T0.7|clip|0|0.00"
    # "8B_clip_rel_drop20_attn50_T0.7|clip|1|0.50"
    # "8B_hdiff_rel_drop20_attn50_T0.7|hdiff|1|0.50"
    "8B_hdiff_rel_drop20_attn0_T0.7|hdiff|0|0"
)

if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${CLIP_MODEL_PATH}" ]]; then
    echo "CLIP checkpoint not found: ${CLIP_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TASK_JSON}" ]]; then
    echo "Task JSON not found: ${TASK_JSON}" >&2
    exit 1
fi
if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "Video directory not found: ${VIDEO_DIR}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${CLIP_STATS_DIR}"

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r RUN_NAME FRAME_SELECTOR ENABLE_ATTN ATTN_DROP_RATIO <<< "${EXP}"

    RUN_NUM_WORKERS="${NUM_WORKERS}"
    FRAME_SELECTOR_ARGS=()
    if [[ "${FRAME_SELECTOR}" == "clip" ]]; then
        RUN_NUM_WORKERS="${CLIP_NUM_WORKERS}"
        FRAME_SELECTOR_ARGS=(
            --clip_enabled
            --clip_model_path "${CLIP_MODEL_PATH}"
            --clip_device "${CLIP_DEVICE}"
            --clip_dtype "${CLIP_DTYPE}"
            --clip_batch_size "${CLIP_BATCH_SIZE}"
            --clip_strategy "${CLIP_STRATEGY}"
            --clip_drop_ratio "${CLIP_DROP_RATIO}"
            --clip_similarity_threshold "${CLIP_SIMILARITY_THRESHOLD}"
            --clip_min_keep_frames "${MIN_FRAMES}"
            --clip_max_interval "${CLIP_MAX_INTERVAL}"
            --clip_stats_path "${CLIP_STATS_DIR}/${RUN_NAME}_${RUN_INSTANCE}.jsonl"
        )
    elif [[ "${FRAME_SELECTOR}" == "hdiff" ]]; then
        FRAME_SELECTOR_ARGS=(
            --hist_enabled
            --hist_bins "${HDIFF_BINS}"
            --hist_threshold "${HDIFF_DROP_RATIO}"
            --drop_relative_hdiff
            --hist_min_keep_frames "${MIN_FRAMES}"
            --hist_max_interval "${HDIFF_MAX_INTERVAL}"
        )
    else
        echo "Unknown frame selector: ${FRAME_SELECTOR}" >&2
        exit 1
    fi

    LOADER_ARGS=(
        --batch_size "${BATCH_SIZE}"
        --num_workers "${RUN_NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
    )
    if [[ "${PIN_MEMORY}" == "1" || "${PIN_MEMORY}" == "true" || "${PIN_MEMORY}" == "True" ]]; then
        LOADER_ARGS+=(--pin_memory)
    fi
    if [[ "${RUN_NUM_WORKERS}" != "0" && ( "${PERSISTENT_WORKERS}" == "1" || "${PERSISTENT_WORKERS}" == "true" || "${PERSISTENT_WORKERS}" == "True" ) ]]; then
        LOADER_ARGS+=(--persistent_workers)
    fi

    ATTN_ARGS=(--drop_method none --drop_threshold 0)
    if [[ "${ENABLE_ATTN}" == "1" ]]; then
        TOP_K_RATIO="$(awk "BEGIN {print 1 - ${ATTN_DROP_RATIO}}")"
        ATTN_ARGS=(
            --drop_method attention
            --drop_threshold 0
            --attn_layer_idx "${ATTN_LAYER_IDX}"
            --top_k_ratio "${TOP_K_RATIO}"
            --scorer_chunk_size "${SCORER_CHUNK_SIZE}"
        )
    fi

    echo "========== OVO-Bench CLIP/Hdiff experiment =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "TASK_JSON=${TASK_JSON}"
    echo "FRAME_SELECTOR=${FRAME_SELECTOR}"
    echo "HDIFF_DROP_RATIO=${HDIFF_DROP_RATIO}"
    echo "NUM_WORKERS=${RUN_NUM_WORKERS}"
    echo "ATTN_DROP_RATIO=${ATTN_DROP_RATIO}"
    echo "ATTN_LAYER_IDX=${ATTN_LAYER_IDX}"

    python eval_qwen3vl/scripts/qwen3_vl_clip/ovobench_hdiff_parallel.py \
        --run_name "${RUN_NAME}" \
        --ckpt_path "${CKPT_PATH}" \
        --task_json "${TASK_JSON}" \
        --video_dir "${VIDEO_DIR}" \
        --result_dir "${RESULT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --min_pixels "${MIN_PIXELS}" \
        --max_pixels "${MAX_PIXELS}" \
        --min_frames "${MIN_FRAMES}" \
        --max_frames "${MAX_FRAMES}" \
        --fps "${FPS}" \
        --temperature "${TEMPERATURE}" \
        "${LOADER_ARGS[@]}" \
        "${ATTN_ARGS[@]}" \
        "${FRAME_SELECTOR_ARGS[@]}"
done
