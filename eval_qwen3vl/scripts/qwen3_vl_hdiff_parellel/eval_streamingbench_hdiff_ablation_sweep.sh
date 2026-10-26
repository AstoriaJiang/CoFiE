#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# "RUN_NAME|CKPT_PATH|LOG_DIR|HIST_ENABLED|HIST_THRESHOLD|DROP_RELATIVE_HDIFF|HIST_MIN_KEEP_FRAMES|HIST_MAX_INTERVAL|ENABLE_ATTN_V2|ATTN_DROP_RATIO"

CKPT_PATH="${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="${REPO_ROOT}/eval_qwen3vl/result_streamingbench"
LOG_DIR="${RESULT_DIR}/log"

# StreamingBench contains 2,500 records.  Use one deterministic 10% (250
# samples) stratified subset for every run in this sweep.
SAMPLE_FRAC="0.10"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLED_TASK_CSV="${RESULT_DIR}/tmp/streamingbench_10pct_seed${SAMPLE_SEED}.csv"

# Hdiff thresholds are relative drop ratios.  The first entry intentionally
# keeps Hdiff enabled with a zero threshold, so it is a no-frame-drop baseline.
EXPERIMENTS=(
    "8B_hdiff_rel_drop0_attn0_T0.7|${CKPT_PATH}|${LOG_DIR}|1|0.00|1|4|15|0|0.00"
    "8B_hdiff_rel_drop20_attn37.5_T0.7|${CKPT_PATH}|${LOG_DIR}|1|0.20|1|4|15|1|0.375"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r RUN_NAME CKPT_PATH LOG_DIR HIST_ENABLED HIST_THRESHOLD DROP_RELATIVE_HDIFF HIST_MIN_KEEP_FRAMES HIST_MAX_INTERVAL ENABLE_ATTN_V2 ATTN_DROP_RATIO <<< "${EXP}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    echo "========== StreamingBench hdiff experiment =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "HIST_ENABLED=${HIST_ENABLED}"
    echo "HIST_THRESHOLD=${HIST_THRESHOLD}"
    echo "DROP_RELATIVE_HDIFF=${DROP_RELATIVE_HDIFF}"
    echo "HIST_MIN_KEEP_FRAMES=${HIST_MIN_KEEP_FRAMES}"
    echo "HIST_MAX_INTERVAL=${HIST_MAX_INTERVAL}"
    echo "ENABLE_ATTN_V2=${ENABLE_ATTN_V2}"
    echo "ATTN_DROP_RATIO=${ATTN_DROP_RATIO}"
    echo "SAMPLE_FRAC=${SAMPLE_FRAC} (250 samples)"
    echo "SAMPLE_SEED=${SAMPLE_SEED}"

    mkdir -p "${LOG_DIR}" "${RESULT_DIR}/tmp"

    RUN_NAME="${RUN_NAME}" \
    CKPT_PATH="${CKPT_PATH}" \
    LOG_DIR="${LOG_DIR}" \
    USE_SUBSET=1 \
    SAMPLE_FRAC="${SAMPLE_FRAC}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    SAMPLED_TASK_CSV="${SAMPLED_TASK_CSV}" \
    HIST_ENABLED="${HIST_ENABLED}" \
    HIST_THRESHOLD="${HIST_THRESHOLD}" \
    DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF}" \
    HIST_MIN_KEEP_FRAMES="${HIST_MIN_KEEP_FRAMES}" \
    HIST_MAX_INTERVAL="${HIST_MAX_INTERVAL}" \
    ENABLE_ATTN_V2="${ENABLE_ATTN_V2}" \
    ATTN_DROP_RATIO="${ATTN_DROP_RATIO}" \
    bash eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_streamingbench_hdiff_parallel.sh

done
