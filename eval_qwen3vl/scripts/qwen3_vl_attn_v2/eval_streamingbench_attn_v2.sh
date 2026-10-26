#!/usr/bin/env bash
set -euo pipefail

_EVAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${_EVAL_SH_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0

export TASK_CSV="./datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
export VIDEO_DIR="./datasets/StreamingBench/Real-Time_Visual_Understanding"

RUN_NAME=8B_attn_v2_30
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_streamingbench"

# ---- Attention frame-drop (v2: chunked scorer, flash/sdpa) ----
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
TOP_K_RATIO="${TOP_K_RATIO:-0.3}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# sampling config
ENABLE_SAMPLING="${ENABLE_SAMPLING:-0}"
export SAMPLE_FRAC=0.05
export SAMPLE_SEED=42
export SAMPLED_TASK_CSV="${RESULT_DIR}/tmp/streamingbench_test_5pct_seed${SAMPLE_SEED}.csv"
EVAL_TASK_CSV="${TASK_CSV}"

mkdir -p "${RESULT_DIR}/tmp"

if [[ "$ENABLE_SAMPLING" == "1" ]]; then
python - <<PY
import os, pandas as pd
task_csv = "${TASK_CSV}"
sampled_task_csv = "${SAMPLED_TASK_CSV}"
sample_frac = ${SAMPLE_FRAC}
sample_seed = ${SAMPLE_SEED}
df = pd.read_csv(task_csv)
if "task_type" in df.columns:
    sampled_df = (
        df.groupby("task_type", group_keys=False)
        .apply(lambda x: x.sample(n=max(1, int(round(len(x) * sample_frac))), random_state=sample_seed))
        .reset_index(drop=True)
    )
else:
    sampled_df = df.sample(frac=sample_frac, random_state=sample_seed).reset_index(drop=True)
sampled_df.to_csv(sampled_task_csv, index=False)
print(f"Saved sampled csv: {sampled_task_csv}")
print(f"Sampled rows: {len(sampled_df)} / {len(df)} ({len(sampled_df)/len(df):.2%})")
PY
EVAL_TASK_CSV="${SAMPLED_TASK_CSV}"
echo "Sampling enabled: using ${EVAL_TASK_CSV}"
else
echo "Sampling disabled: using ${EVAL_TASK_CSV}"
fi


args=(
    --run_name "$RUN_NAME"
    --ckpt_path "$CKPT_PATH"
    --task_csv "$EVAL_TASK_CSV"
    --video_dir "$VIDEO_DIR"
    --result_dir "$RESULT_DIR"
    --attn_layer_idx "$ATTN_LAYER_IDX"
    --top_k_ratio "$TOP_K_RATIO"
    --scorer_chunk_size "$SCORER_CHUNK_SIZE"
    --attn_implementation "$ATTN_IMPL"
    --max_frames 1000
    --fps 1
)
python "${_EVAL_SH_DIR}/streamingbench_attn_v2.py" "${args[@]}"
