#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# "RUN_NAME|CKPT_PATH|LOG_DIR|HIST_ENABLED|HIST_THRESHOLD|ENABLE_ATTN_V2|ATTN_DROP_RATIO"

EXPERIMENTS=(
    "ckpt3000_hdiff_30_attn_43_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|eval_qwen3vl/result_ovobench/log-full|1|0.3|1|0.43"
    "ckpt3000_hdiff_40_attn_50_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|eval_qwen3vl/result_ovobench/log-full|1|0.4|1|0.50"
    "ckpt3000_hdiff_50_attn_60_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|eval_qwen3vl/result_ovobench/log-full|1|0.5|1|0.60"
    # "ckpt3000_hdiff_60_attn_70_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|eval_qwen3vl/result_ovobench/log-full|1|0.6|1|0.70"

    "8B_hdiff_20_attn_37.5_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|1|0.2|1|0.375"
    "8B_hdiff_30_attn_43_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|1|0.3|1|0.43"
    "8B_hdiff_40_attn_50_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|1|0.4|1|0.50"

    "8B_hdiff_50_attn_60_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|1|0.5|1|0.60"
    "8B_hdiff_60_attn_75_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|1|0.6|1|0.75"
    "8B_hdiff_0_attn_0_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|eval_qwen3vl/result_ovobench/log-full|0|0|0|0"
    
    
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r RUN_NAME CKPT_PATH LOG_DIR HIST_ENABLED HIST_THRESHOLD ENABLE_ATTN_V2 ATTN_DROP_RATIO <<< "${EXP}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    echo "========== OVO-Bench hdiff experiment =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "HIST_ENABLED=${HIST_ENABLED}"
    echo "HIST_THRESHOLD=${HIST_THRESHOLD}"
    echo "ENABLE_ATTN_V2=${ENABLE_ATTN_V2}"
    echo "ATTN_DROP_RATIO=${ATTN_DROP_RATIO}"

    RUN_NAME="${RUN_NAME}" \
    CKPT_PATH="${CKPT_PATH}" \
    LOG_DIR="${LOG_DIR}" \
    HIST_ENABLED="${HIST_ENABLED}" \
    HIST_THRESHOLD="${HIST_THRESHOLD}" \
    ENABLE_ATTN_V2="${ENABLE_ATTN_V2}" \
    ATTN_DROP_RATIO="${ATTN_DROP_RATIO}" \
    bash eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_ovobench_hdiff_parallel.sh

done
