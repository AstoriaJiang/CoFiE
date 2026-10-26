#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# "RUN_NAME|CKPT_PATH|LOG_DIR|HIST_ENABLED|HIST_THRESHOLD|ENABLE_ATTN_V2|ATTN_DROP_RATIO"
EXPERIMENTS=(
    "8B_hdiff_20_attn_37.5_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|log-full|1|0.2|1|0.375"
    # "8B_hdiff_50_attn_60_T0.7|${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct|log-full|1|0.5|1|0.60"
    # "ckpt3000_hdiff_20_attn_37.5_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|log-full|1|0.2|1|0.375"
    # "ckpt3000_hdiff_50_attn_60_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|log-full|1|0.5|1|0.60"
    # "ckpt3000_hdiff_60_attn_75_T0.7|${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926-7_dataset/checkpoint-3000|log-full|1|0.6|1|0.75"
)

NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"

run_eval() {
    local name="$1"
    local suffix="$2"
    local script="$3"
    local output_path="$4"

    echo "========== ${name} =========="
    RUN_NAME="${RUN_NAME}_${suffix}" \
    CKPT_PATH="${CKPT_PATH}" \
    OUTPUT_PATH="${output_path}" \
    ENABLE_HDIFF="${HIST_ENABLED}" \
    HIST_THRESHOLD="${HIST_THRESHOLD}" \
    ENABLE_ATTN_V2="${ENABLE_ATTN_V2}" \
    ATTN_DROP_RATIO="${ATTN_DROP_RATIO}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    bash "${script}"
}

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r RUN_NAME CKPT_PATH LOG_DIR HIST_ENABLED HIST_THRESHOLD ENABLE_ATTN_V2 ATTN_DROP_RATIO <<< "${EXP}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    echo "========== 4bench hdiff sweep =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "HIST_ENABLED=${HIST_ENABLED}"
    echo "HIST_THRESHOLD=${HIST_THRESHOLD}"
    echo "ENABLE_ATTN_V2=${ENABLE_ATTN_V2}"
    echo "ATTN_DROP_RATIO=${ATTN_DROP_RATIO}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "BATCH_SIZE=${BATCH_SIZE}"

    run_eval "MVBench" "mvbench" "eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_mvbench_hdiff_parellel.sh" "eval_qwen3vl/result_mvbench/${LOG_DIR}"
    run_eval "MLVU" "mlvu" "eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_mlvu_hdiff_parellel.sh" "eval_qwen3vl/result_mlvu/${LOG_DIR}"
    run_eval "LongVideoBench" "longvideobench" "eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_longvideobench_hdiff_parellel.sh" "eval_qwen3vl/result_longvideobench/${LOG_DIR}"
    run_eval "EgoSchema" "egoschema" "eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_egoschema_hdiff_parellel.sh" "eval_qwen3vl/result_egoschema/${LOG_DIR}"
done
