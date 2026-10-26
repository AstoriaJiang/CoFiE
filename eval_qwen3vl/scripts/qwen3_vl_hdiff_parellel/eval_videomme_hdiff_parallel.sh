#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
export VIDEO_DIR="./datasets/Video-MME/data/"
CKPT_PATH="${CKPT_PATH:-ckpt/Qwen3-VL-8B-Instruct}"
RESULT_DIR="eval_qwen3vl/result_videomme"

RUN_NAME="${RUN_NAME:-8B_T0_hdiff_b32_35_attn_70}"
LOG_DIR="${LOG_DIR:-eval_qwen3vl/result_videomme/log-hdiff_vs_attn_drop_ratio}"

TEMPERATURE="${TEMPERATURE:-0.7}"
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}" # 1 时使用 hdiff 相对筛选：按相邻帧 hist 差排序，保留 top k；HIST_THRESHOLD 表示丢弃比例。

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-5}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

HIST_ENABLED="${HIST_ENABLED:-1}"
HIST_BINS=32 # 直方图分桶数。值越大，比较越细，对亮度/颜色分布差异越敏感，但也更容易受噪声影响。
HIST_THRESHOLD="${HIST_THRESHOLD:-0.35}" # 帧差异阈值[0, 1]。当前帧和相邻帧/上次保留帧的直方图差异超过该值时保留；值越大，删帧越激进。
HIST_MIN_KEEP_FRAMES=4 # 每段视频至少保留的帧数，避免筛得太少。
HIST_MAX_INTERVAL=30 # 最长间隔保护。如果连续很多帧都被判为相似，最多隔 30 帧也会强制保留一帧，避免长时间段完全没有采样。

DROP_METHOD=none # "feature" or "pixel" or "none"
DROP_THRESHOLD=0

ENABLE_ATTN_V2="${ENABLE_ATTN_V2:-1}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.7}"
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

# quick test sampling
USE_SUBDATASET=0
export SAMPLE_FRAC=0.05
export SAMPLE_SEED=42
export SAMPLED_TASK_PARQUET="${RESULT_DIR}/tmp/videomme_test_5pct_seed${SAMPLE_SEED}.parquet"

mkdir -p "${RESULT_DIR}/tmp"

if [[ "${USE_SUBDATASET}" == "1" ]]; then
python - <<'PY'
import os
import pandas as pd

task_parquet = os.environ["TASK_PARQUET"]
sampled_task_parquet = os.environ["SAMPLED_TASK_PARQUET"]
sample_frac = float(os.environ["SAMPLE_FRAC"])
sample_seed = int(os.environ["SAMPLE_SEED"])

df = pd.read_parquet(task_parquet)
if "duration" in df.columns:
    sampled_df = (
        df.groupby("duration", group_keys=False)
        .apply(lambda x: x.sample(n=max(1, int(round(len(x) * sample_frac))), random_state=sample_seed))
        .reset_index(drop=True)
    )
else:
    sampled_df = df.sample(frac=sample_frac, random_state=sample_seed).reset_index(drop=True)

sampled_df.to_parquet(sampled_task_parquet, index=False)
print(f"Saved sampled parquet: {sampled_task_parquet}")
print(f"Sampled rows: {len(sampled_df)} / {len(df)} ({len(sampled_df)/len(df):.2%})")
PY
    TASK_PARQUET_FOR_EVAL="$SAMPLED_TASK_PARQUET"
else
    TASK_PARQUET_FOR_EVAL="$TASK_PARQUET"
fi

HIST_ARGS=()
if [[ "${HIST_ENABLED}" == "1" ]]; then
    HIST_ARGS+=(
        --hist_enabled
        --hist_bins "$HIST_BINS"
        --hist_threshold "$HIST_THRESHOLD"
        --hist_min_keep_frames "$HIST_MIN_KEEP_FRAMES"
        --hist_max_interval "$HIST_MAX_INTERVAL"
    )
fi

DROP_ARGS=()
if [[ "${DROP_RELATIVE_HDIFF}" == "1" || "${DROP_RELATIVE_HDIFF}" == "true" || "${DROP_RELATIVE_HDIFF}" == "True" ]]; then
    DROP_ARGS+=(--drop_relative_hdiff)
fi

LOG_ARGS=()
if [[ -n "${LOG_DIR}" ]]; then
    LOG_ARGS+=(--log_dir "$LOG_DIR")
fi

LOADER_ARGS=(
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
)
if [[ "${PIN_MEMORY}" == "1" || "${PIN_MEMORY}" == "true" || "${PIN_MEMORY}" == "True" ]]; then
    LOADER_ARGS+=(--pin_memory)
fi
if [[ "${PERSISTENT_WORKERS}" == "1" || "${PERSISTENT_WORKERS}" == "true" || "${PERSISTENT_WORKERS}" == "True" ]]; then
    LOADER_ARGS+=(--persistent_workers)
fi

python eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/videomme_hdiff_parallel.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET_FOR_EVAL" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    "${LOG_ARGS[@]}" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE" \
    "${LOADER_ARGS[@]}" \
    "${DROP_ARGS[@]}" \
    "${HIST_ARGS[@]}" \
    "${ATTN_ARGS[@]}"
