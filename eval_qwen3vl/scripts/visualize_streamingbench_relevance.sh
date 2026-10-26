#!/usr/bin/env bash

set -euo pipefail

QUESTION_ID="${QUESTION_ID:-${1:-}}"
TASK_CSV="${TASK_CSV:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding}"
CKPT_PATH="${CKPT_PATH:-ckpt/Qwen2.5-VL-7B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-eval_qwen2.5vl/vis_streamingbench_relevance}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MAX_FRAMES="${MAX_FRAMES:-1016}"
DROP_THRESHOLD="${DROP_THRESHOLD:-0.75}"
GRID_COLS="${GRID_COLS:-4}"

if [[ -z "$QUESTION_ID" ]]; then
  echo "请提供 QUESTION_ID，方式：QUESTION_ID=xxx bash eval_qwen2.5vl/scripts/visualize_streamingbench_relevance.sh"
  exit 1
fi

QUESTION_ID_SUFFIX="$QUESTION_ID"
if [[ "$QUESTION_ID" == *"sample_"* ]]; then
  QUESTION_ID_SUFFIX="${QUESTION_ID##*sample_}"
fi
QUESTION_ID_SUFFIX="${QUESTION_ID_SUFFIX// /_}"
QUESTION_ID_SUFFIX="${QUESTION_ID_SUFFIX//\//_}"

RUN_NAME="${RUN_NAME:-query_feature_sample${QUESTION_ID_SUFFIX}}"
RESULT_DIR="${RESULT_ROOT}/query_feature"
SIM_JSONL="${SIM_JSONL:-${RESULT_DIR}/similarity/${RUN_NAME}.jsonl}"
VIS_DIR="${VIS_DIR:-${RESULT_DIR}/visual/${RUN_NAME}}"

mkdir -p "$RESULT_DIR" "$(dirname "$SIM_JSONL")" "$VIS_DIR"

python eval_qwen2.5vl/scripts/streamingbench_query.py \
  --run_name "$RUN_NAME" \
  --drop_method query_feature \
  --drop_threshold "$DROP_THRESHOLD" \
  --drop_relative \
  --ckpt_path "$CKPT_PATH" \
  --task_csv "$TASK_CSV" \
  --video_dir "$VIDEO_DIR" \
  --result_dir "$RESULT_DIR" \
  --question_id "$QUESTION_ID" \
  --sim_save_path "$SIM_JSONL" \
  --min_pixels "$MIN_PIXELS" \
  --max_pixels "$MAX_PIXELS" \
  --min_frames "$MIN_FRAMES" \
  --max_frames "$MAX_FRAMES"

python eval_qwen3vl/scripts/visualize_streamingbench_relevance.py \
  --sim_jsonl "$SIM_JSONL" \
  --output_dir "$VIS_DIR" \
  --task_csv "$TASK_CSV" \
  --video_dir "$VIDEO_DIR" \
  --question_id "$QUESTION_ID" \
  --score_key query_relevance_map \
  --min_pixels "$MIN_PIXELS" \
  --max_pixels "$MAX_PIXELS" \
  --min_frames "$MIN_FRAMES" \
  --max_frames "$MAX_FRAMES" \
  --grid_cols "$GRID_COLS"

echo "similarity json: $SIM_JSONL"
echo "visual output: $VIS_DIR"

# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_streamingbench_relevance.sh

# 单独可视化已有相似度 json：
# python3 eval_qwen2.5vl/scripts/visualize_streamingbench_relevance.py \
#   --sim_jsonl your_similarity.jsonl \
#   --output_dir your_output_dir \
#   --task_csv "$TASK_CSV" \
#   --video_dir "$VIDEO_DIR" \
#   --question_id "$QUESTION_ID" \
#   --score_key query_relevance_map
