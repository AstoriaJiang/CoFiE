#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/eval_qwen3vl/lmms-eval:${REPO_ROOT}/eval_qwen3vl:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/huggingface}"
unset ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_WORKERS}}"

CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
TASK="mvbench"
RUN_NAME="${RUN_NAME:-8B_hdiff_parellel_mvbench}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_qwen3vl/result_mvbench}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-1000}"
MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-4}"
FPS="${FPS:-1}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29504}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ENABLE_HDIFF="${ENABLE_HDIFF:-1}"
HIST_BINS="${HIST_BINS:-32}"
HIST_THRESHOLD="${HIST_THRESHOLD:-0.5}"
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}"
HIST_MIN_KEEP_FRAMES="${HIST_MIN_KEEP_FRAMES:-4}"
HIST_MAX_INTERVAL="${HIST_MAX_INTERVAL:-30}"

ENABLE_ATTN_V2="${ENABLE_ATTN_V2:-1}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.5}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"

if [[ -z "${TOP_K_RATIO:-}" ]]; then
    TOP_K_RATIO="$(awk "BEGIN {print 1 - $ATTN_DROP_RATIO}")"
fi

bool_arg() {
    case "$1" in
        1|true|True|TRUE|yes|Yes|YES) echo true ;;
        *) echo false ;;
    esac
}

HDIFF_ENABLED="$(bool_arg "$ENABLE_HDIFF")"
DROP_RELATIVE_HDIFF_BOOL="$(bool_arg "$DROP_RELATIVE_HDIFF")"

DROP_METHOD=none
if [[ "$(bool_arg "$ENABLE_ATTN_V2")" == "true" ]]; then
    DROP_METHOD=attention
fi

MODEL_ARGS="pretrained=${CKPT_PATH},drop_method=${DROP_METHOD},attn_layer_idx=${ATTN_LAYER_IDX},top_k_ratio=${TOP_K_RATIO},scorer_chunk_size=${SCORER_CHUNK_SIZE},max_num_frames=${MAX_NUM_FRAMES},min_num_frames=${MIN_NUM_FRAMES},fps=${FPS},attn_implementation=${ATTN_IMPL},hist_enabled=${HDIFF_ENABLED},hist_bins=${HIST_BINS},hist_threshold=${HIST_THRESHOLD},drop_relative_hdiff=${DROP_RELATIVE_HDIFF_BOOL},hist_min_keep_frames=${HIST_MIN_KEEP_FRAMES},hist_max_interval=${HIST_MAX_INTERVAL}"

echo "MODEL_ARGS=${MODEL_ARGS}"
accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" -m lmms_eval \
    --model qwen3_vl_hdiff_parellel \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --verbosity=INFO
