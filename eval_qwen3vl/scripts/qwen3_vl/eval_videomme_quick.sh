export CUDA_VISIBLE_DEVICES=0
export TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="./datasets/Video-MME/data/"

RUN_NAME=8B_ckpt_v21_20k
CKPT_PATH="ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme"

# quick eval sampling config
export SAMPLE_FRAC=0.05
export SAMPLE_SEED=42
export SAMPLED_TASK_PARQUET="${RESULT_DIR}/tmp/videomme_test_5pct_seed${SAMPLE_SEED}.parquet"

# DTD arguments
DROP_METHOD=none   # "feature" or "pixel" or "none"
DROP_THRESHOLD=0

TEMPERATURE=0

mkdir -p "${RESULT_DIR}/tmp"

# Stratified sampling by duration for a more uniform 20% subset.
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

python eval_qwen3vl/scripts/qwen3_vl/videomme.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_relative \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$SAMPLED_TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    --temperature "$TEMPERATURE"
