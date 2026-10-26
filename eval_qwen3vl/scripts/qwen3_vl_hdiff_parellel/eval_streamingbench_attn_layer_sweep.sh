#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Qwen3-VL-8B-Instruct has 36 hidden layers, indexed 0..35.
# "RUN_NAME|CKPT_PATH|LOG_DIR|HIST_ENABLED|ENABLE_ATTN_V2|ATTN_LAYER_IDX|ATTN_DROP_RATIO|SCORER_CHUNK_SIZE"

CKPT_PATH="${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct"
LOG_DIR="${REPO_ROOT}/eval_qwen3vl/result_streamingbench/log-attn_layer"

EXPERIMENTS=(
    "8B_attn_layer0_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|0|0.60|512"
    "8B_attn_layer4_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|4|0.60|512"
    "8B_attn_layer8_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|8|0.60|512"
    "8B_attn_layer12_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|12|0.60|512"
    "8B_attn_layer18_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|18|0.60|512"
    "8B_attn_layer24_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|24|0.60|512"
    "8B_attn_layer30_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|30|0.60|512"
    "8B_attn_layer35_drop60_T0.7|${CKPT_PATH}|${LOG_DIR}|0|1|35|0.60|512"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r RUN_NAME CKPT_PATH LOG_DIR HIST_ENABLED ENABLE_ATTN_V2 ATTN_LAYER_IDX ATTN_DROP_RATIO SCORER_CHUNK_SIZE <<< "${EXP}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    echo "========== StreamingBench attention layer experiment =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "HIST_ENABLED=${HIST_ENABLED}"
    echo "ENABLE_ATTN_V2=${ENABLE_ATTN_V2}"
    echo "ATTN_LAYER_IDX=${ATTN_LAYER_IDX}"
    echo "ATTN_DROP_RATIO=${ATTN_DROP_RATIO}"
    echo "SCORER_CHUNK_SIZE=${SCORER_CHUNK_SIZE}"
    echo "USE_SUBSET=1"
    echo "SAMPLE_FRAC=0.10"
    echo "SUBSET_STRATIFY_COL=task_type"

    RUN_NAME="${RUN_NAME}" \
    CKPT_PATH="${CKPT_PATH}" \
    LOG_DIR="${LOG_DIR}" \
    USE_SUBSET=1 \
    SAMPLE_FRAC=0.10 \
    SAMPLED_TASK_CSV="eval_qwen3vl/result_streamingbench/tmp/streamingbench_10pct_seed42.csv" \
    SUBSET_STRATIFY_COL=task_type \
    HIST_ENABLED="${HIST_ENABLED}" \
    DROP_RELATIVE_HDIFF=0 \
    ENABLE_ATTN_V2="${ENABLE_ATTN_V2}" \
    ATTN_LAYER_IDX="${ATTN_LAYER_IDX}" \
    ATTN_DROP_RATIO="${ATTN_DROP_RATIO}" \
    SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE}" \
    bash eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_streamingbench_hdiff_parallel.sh

done
