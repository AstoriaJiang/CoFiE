#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

export TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
export VIDEO_DIR="./datasets/Video-MME/data/"

RUN_NAME=attn_50
CKPT_PATH="ckpt/Qwen3-VL-4B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme"

# ---- Attention frame-drop ----
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
TOP_K_RATIO="${TOP_K_RATIO:-0.5}"

# sampling config
ENABLE_SAMPLING="${ENABLE_SAMPLING:-0}"
export SAMPLE_FRAC=0.05
export SAMPLE_SEED=42
export SAMPLED_TASK_PARQUET="${RESULT_DIR}/tmp/videomme_test_5pct_seed${SAMPLE_SEED}.parquet"
EVAL_TASK_PARQUET="${TASK_PARQUET}"


if [[ "$ENABLE_SAMPLING" == "1" ]]; then
python - <<PY
import os, pandas as pd
task_parquet = "${TASK_PARQUET}"
sampled_task_parquet = "${SAMPLED_TASK_PARQUET}"
sample_frac = ${SAMPLE_FRAC}
sample_seed = ${SAMPLE_SEED}
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
EVAL_TASK_PARQUET="${SAMPLED_TASK_PARQUET}"
echo "Sampling enabled: using ${EVAL_TASK_PARQUET}"
else
echo "Sampling disabled: using ${EVAL_TASK_PARQUET}"
fi


python eval_qwen3vl/scripts/videomme_attn_frame_drop.py \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$EVAL_TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --attn_layer_idx "$ATTN_LAYER_IDX" \
    --top_k_ratio "$TOP_K_RATIO" \
    --max_frames 1000 \
    --fps 1
