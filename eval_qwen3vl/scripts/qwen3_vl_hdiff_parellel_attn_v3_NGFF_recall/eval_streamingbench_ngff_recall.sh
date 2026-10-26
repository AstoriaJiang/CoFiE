#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/eval_qwen3vl/.venv/bin/python}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
SOURCE_TASK_CSV="${REPO_ROOT}/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="${REPO_ROOT}/datasets/StreamingBench/Real-Time_Visual_Understanding"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/eval_qwen3vl/result_streamingbench/ngff_recall}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_qwen3vl/result_streamingbench/log-NGFF_recall}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLED_TASK_CSV="${REPO_ROOT}/eval_qwen3vl/result_streamingbench/tmp/streamingbench_10pct_seed${SAMPLE_SEED}.csv"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SOURCE_TASK_CSV SAMPLED_TASK_CSV SAMPLE_SEED

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_TASK_CSV}" ]]; then
    echo "StreamingBench CSV not found: ${SOURCE_TASK_CSV}" >&2
    exit 1
fi
if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "StreamingBench video directory not found: ${VIDEO_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${SAMPLED_TASK_CSV}")" "${RESULT_DIR}" "${LOG_DIR}"

if [[ ! -f "${SAMPLED_TASK_CSV}" ]]; then
    "${PYTHON_BIN}" - <<'PY_SAMPLE'
import os
import pandas as pd

source_csv = os.environ["SOURCE_TASK_CSV"]
sampled_csv = os.environ["SAMPLED_TASK_CSV"]
seed = int(os.environ["SAMPLE_SEED"])

df = pd.read_csv(source_csv)
groups = list(df.groupby("task_type", sort=True))
target_total = int(round(len(df) * 0.10))
if target_total != 250:
    raise ValueError(f"Expected a 2,500-row source and a 250-row subset, got target={target_total}")
base, remainder = divmod(target_total, len(groups))
quotas = {name: min(len(group), base) for name, group in groups}
for name, group in sorted(groups, key=lambda item: (-len(item[1]), str(item[0]))):
    if remainder <= 0:
        break
    if quotas[name] < len(group):
        quotas[name] += 1
        remainder -= 1

sampled = pd.concat(
    [
        group.sample(n=quotas[name], random_state=seed)
        for name, group in groups
        if quotas[name] > 0
    ],
    axis=0,
).sample(frac=1, random_state=seed).reset_index(drop=True)
sampled.to_csv(sampled_csv, index=False)
print(f"Saved fixed 10% StreamingBench subset: {sampled_csv} ({len(sampled)} rows)")
PY_SAMPLE
fi

SAMPLE_COUNT="$("${PYTHON_BIN}" -c 'import pandas as pd, sys; print(len(pd.read_csv(sys.argv[1])))' "${SAMPLED_TASK_CSV}")"
if [[ "${SAMPLE_COUNT}" != "250" ]]; then
    echo "Expected the fixed 10% subset to contain 250 rows, found ${SAMPLE_COUNT}" >&2
    exit 1
fi

echo "========== Qwen3-VL NGFF recall analysis =========="
echo "Checkpoint: ${CKPT_PATH}"
echo "Subset: ${SAMPLED_TASK_CSV} (${SAMPLE_COUNT} rows)"
echo "NGFF drop ratios: 20%, 50%"
echo "Methods: NGFF, Uniform, Random (5 seeds)"
echo "Metrics: Attention Preservation, QER Top-Evidence Recall@20%"
echo "Result dir: ${RESULT_DIR}"
echo "Log dir: ${LOG_DIR}"

"${PYTHON_BIN}" \
    eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel_attn_v3_NGFF_recall/streamingbench_ngff_recall.py \
    --run_name "Qwen3VL_8B_NGFF_recall_drop20_50" \
    --ckpt_path "${CKPT_PATH}" \
    --task_csv "${SAMPLED_TASK_CSV}" \
    --video_dir "${VIDEO_DIR}" \
    --result_dir "${RESULT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --fps "${FPS:-1.0}" \
    --min_frames "${MIN_FRAMES:-4}" \
    --max_frames "${MAX_FRAMES:-1000}" \
    --hist_bins "${HIST_BINS:-32}" \
    --hist_min_keep_frames "${HIST_MIN_KEEP_FRAMES:-4}" \
    --hist_max_interval "${HIST_MAX_INTERVAL:-15}" \
    --attn_layer_idx "${ATTN_LAYER_IDX:--1}" \
    --scorer_chunk_size "${SCORER_CHUNK_SIZE:-512}" \
    --random_seeds 5 \
    --bootstrap_iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --num_workers "${NUM_WORKERS:-2}" \
    --prefetch_factor "${PREFETCH_FACTOR:-2}" \
    --pin_memory \
    --persistent_workers
