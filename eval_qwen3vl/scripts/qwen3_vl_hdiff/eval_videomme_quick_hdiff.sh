#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

export TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
# export TASK_PARQUET="eval_qwen3vl/result_videomme/tmp/videomme_test_video_id_549.parquet"

export VIDEO_DIR="./datasets/Video-MME/data/"


# RUN_NAME=8B_hdiff_1_1_2sample_2resize
RUN_NAME=8B_hdiff_r30_1sample_1resize_2_2
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme"

DROP_METHOD=none # "feature" or "pixel" or "none"
DROP_THRESHOLD=0

TEMPERATURE="${TEMPERATURE:-0}"
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}" # 1 时使用 hdiff 相对筛选：按相邻帧 hist 差排序，保留 top k；HIST_THRESHOLD 表示丢弃比例。

HIST_ENABLED=1
HIST_BINS=64 # 直方图分桶数。值越大，比较越细，对亮度/颜色分布差异越敏感，但也更容易受噪声影响。
HIST_THRESHOLD=0.3 # 帧差异阈值[0, 1]。当前帧和相邻帧/上次保留帧的直方图差异超过该值时保留；值越大，删帧越激进。
HIST_MIN_KEEP_FRAMES=4 # 每段视频至少保留的帧数，避免筛得太少。
HIST_MAX_INTERVAL=30 # 最长间隔保护。如果连续很多帧都被判为相似，最多隔 30 帧也会强制保留一帧，避免长时间段完全没有采样。

# quick test sampling
USE_SUBDATASET=1
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

python eval_qwen3vl/scripts/qwen3_vl_hdiff/videomme_hdiff.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET_FOR_EVAL" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE" \
    "${DROP_ARGS[@]}" \
    "${HIST_ARGS[@]}" \
