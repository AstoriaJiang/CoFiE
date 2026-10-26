#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="./datasets/Video-MME/data/"

USE_SUBDATASET=1

RUN_NAME=8B_fps_quick
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme"

# quick eval sampling config
export SAMPLE_FRAC=0.05
export SAMPLE_SEED=42
export SAMPLED_TASK_PARQUET="${RESULT_DIR}/tmp/videomme_test_5pct_seed${SAMPLE_SEED}.parquet"
mkdir -p "${RESULT_DIR}/tmp"

# FPS Frame Selection arguments
FPS_ENABLED=1  # set to 1 to enable, 0 to disable pixel-diff frame selection
TAU_ADJ=0 # 相邻帧差分阈值（d_adj）。当前帧和前一帧像素平均差超过它，才算“有明显变化”。
TAU_REF=0 # 参考帧差分阈值（d_ref）。当前帧和参考帧像素平均差超过它，才算“有明显变化”。
CONSECUTIVE_THRESHOLD=1 # 连续变化判定长度。需要连续多少帧都满足 d_adj > TAU_ADJ 才触发保留，避免单帧噪声误判。
MAX_INTERVAL=10 # 最大无保留间隔。距离上次保留帧达到这个值时，强制保留一帧，防止时间轴断裂。

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"



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

FPS_ARGS=""
if [ "$FPS_ENABLED" -eq 1 ]; then
    FPS_ARGS="--fps_enabled --tau_adj $TAU_ADJ --tau_ref $TAU_REF --consecutive_threshold $CONSECUTIVE_THRESHOLD --max_interval $MAX_INTERVAL"
fi

python eval_qwen3vl/scripts/qwen3_vl_fps/videomme_fps.py \
    --run_name $RUN_NAME \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET_FOR_EVAL" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --attn_implementation "$ATTN_IMPL"\
    --max_frames 1000 \
    --fps 2 \
    $FPS_ARGS