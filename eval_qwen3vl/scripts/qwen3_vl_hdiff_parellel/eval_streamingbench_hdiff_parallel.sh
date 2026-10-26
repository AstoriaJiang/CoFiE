#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR="${QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR:-1}"

export TASK_CSV="${TASK_CSV:-${REPO_ROOT}/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/datasets/StreamingBench/Real-Time_Visual_Understanding}"

RUN_NAME="${RUN_NAME:-8B_hdiff_50_attn_60_T0.7}"
CKPT_PATH="${CKPT_PATH:-ckpt/Qwen3-VL-8B-Instruct}"
RESULT_DIR="eval_qwen3vl/result_streamingbench"
LOG_DIR="${LOG_DIR:-eval_qwen3vl/result_streamingbench/log-NGFF_ablation}"

TEMPERATURE=0.7
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}" # 1 时使用相对筛选：按相邻帧 hist 差排序，保留 top k；HIST_THRESHOLD 表示丢弃比例。

BATCH_SIZE="${BATCH_SIZE:-1}"
# NUM_WORKERS="${NUM_WORKERS:-1}"
# PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
# PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

HIST_ENABLED="${HIST_ENABLED:-1}"
HIST_BINS=32
HIST_THRESHOLD="${HIST_THRESHOLD:-0.5}"
HIST_MIN_KEEP_FRAMES="${HIST_MIN_KEEP_FRAMES:-4}"
HIST_MAX_INTERVAL="${HIST_MAX_INTERVAL:-30}"

USE_SUBSET="${USE_SUBSET:-0}"
export SAMPLE_FRAC="${SAMPLE_FRAC:-0.05}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export SAMPLED_TASK_CSV="${SAMPLED_TASK_CSV:-${RESULT_DIR}/tmp/streamingbench_5pct_seed${SAMPLE_SEED}.csv}"
export SUBSET_STRATIFY_COL="${SUBSET_STRATIFY_COL:-task_type}"
TASK_CSV_EVAL="$TASK_CSV"

DROP_METHOD=none
DROP_THRESHOLD=0

ENABLE_ATTN_V2="${ENABLE_ATTN_V2:-1}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.60}"
TOP_K_RATIO="${TOP_K_RATIO:-$(awk "BEGIN {print 1 - $ATTN_DROP_RATIO}")}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"

ATTN_ARGS=()
if [[ "${ENABLE_ATTN_V2}" == "1" || "${ENABLE_ATTN_V2}" == "true" || "${ENABLE_ATTN_V2}" == "True" ]]; then
    DROP_METHOD=attention
    ATTN_ARGS+=(
        --attn_layer_idx "$ATTN_LAYER_IDX"
        --top_k_ratio "$TOP_K_RATIO"
        --scorer_chunk_size "$SCORER_CHUNK_SIZE"
    )
fi

HIST_ARGS=()
if [[ "${HIST_ENABLED}" == "1" ]]; then
    HIST_ARGS+=(--hist_enabled)
fi

DROP_ARGS=()
if [[ "${DROP_RELATIVE_HDIFF}" == "1" || "${DROP_RELATIVE_HDIFF}" == "true" || "${DROP_RELATIVE_HDIFF}" == "True" ]]; then
    DROP_ARGS+=(--drop_relative_hdiff)
fi

# LOADER_ARGS=(
#     --batch_size "$BATCH_SIZE"
#     --num_workers "$NUM_WORKERS"
#     --prefetch_factor "$PREFETCH_FACTOR"
# )
LOADER_ARGS=(--batch_size "$BATCH_SIZE" --num_workers 0)
if [[ "${PIN_MEMORY}" == "1" || "${PIN_MEMORY}" == "true" || "${PIN_MEMORY}" == "True" ]]; then
    LOADER_ARGS+=(--pin_memory)
fi
# if [[ "${PERSISTENT_WORKERS}" == "1" || "${PERSISTENT_WORKERS}" == "true" || "${PERSISTENT_WORKERS}" == "True" ]]; then
#     LOADER_ARGS+=(--persistent_workers)
# fi

mkdir -p "${RESULT_DIR}/tmp"

if [[ "${USE_SUBSET}" == "1" || "${USE_SUBSET}" == "true" || "${USE_SUBSET}" == "True" ]]; then
    python - <<'PY_SAMPLE'
import os
import pandas as pd

task_csv = os.environ["TASK_CSV"]
sampled_task_csv = os.environ["SAMPLED_TASK_CSV"]
sample_frac = float(os.environ["SAMPLE_FRAC"])
sample_seed = int(os.environ["SAMPLE_SEED"])
stratify_col = os.environ.get("SUBSET_STRATIFY_COL", "task_type")

df = pd.read_csv(task_csv)
target_total = max(1, int(round(len(df) * sample_frac)))

if stratify_col and stratify_col in df.columns:
    groups = list(df.groupby(stratify_col, sort=True))
    target_total = max(target_total, len(groups))
    base = target_total // len(groups)
    remainder = target_total % len(groups)
    quotas = {name: min(len(group), base) for name, group in groups}

    for name, _ in sorted(groups, key=lambda item: (-len(item[1]), str(item[0]))):
        if remainder <= 0:
            break
        if quotas[name] < len(dict(groups)[name]):
            quotas[name] += 1
            remainder -= 1

    remaining = target_total - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for name, group in sorted(groups, key=lambda item: (-len(item[1]), str(item[0]))):
            if quotas[name] < len(group):
                quotas[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    sampled_parts = [
        group.sample(n=quotas[name], random_state=sample_seed)
        for name, group in groups
        if quotas[name] > 0
    ]
    sampled_df = pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=sample_seed).reset_index(drop=True)
    print(f"Stratified balanced subset by column: {stratify_col}")
    print("Original distribution:")
    print(df[stratify_col].value_counts().sort_index().to_string())
    print("Subset distribution:")
    print(sampled_df[stratify_col].value_counts().sort_index().to_string())
else:
    sampled_df = df.sample(n=target_total, random_state=sample_seed).reset_index(drop=True)
    print(f"Stratify column not found, fallback to random subset: {stratify_col}")

sampled_df.to_csv(sampled_task_csv, index=False)

print(f"Saved sampled csv: {sampled_task_csv}")
print(f"Sampled rows: {len(sampled_df)} / {len(df)} ({len(sampled_df)/len(df):.2%})")
PY_SAMPLE
    TASK_CSV_EVAL="$SAMPLED_TASK_CSV"
else
    echo "Use full StreamingBench csv: ${TASK_CSV}"
fi

python eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/streamingbench_hdiff_parallel.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV_EVAL" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --log_dir "$LOG_DIR" \
    --temperature "$TEMPERATURE" \
    "${LOADER_ARGS[@]}" \
    "${DROP_ARGS[@]}" \
    "${HIST_ARGS[@]}" \
    --hist_bins $HIST_BINS \
    --hist_threshold $HIST_THRESHOLD \
    --hist_min_keep_frames $HIST_MIN_KEEP_FRAMES \
    --hist_max_interval $HIST_MAX_INTERVAL \
    "${ATTN_ARGS[@]}"
