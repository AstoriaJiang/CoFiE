#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

METHOD="${1:-feature}"
QUESTION_ID="${QUESTION_ID:-${2:-}}"
TASK_CSV="${TASK_CSV:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding}"
CKPT_PATH="${CKPT_PATH:-ckpt/TimeChatOnline-7B}"
RESULT_ROOT="${RESULT_ROOT:-eval_qwen2.5vl/vis_streamingbench_similarity}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MAX_FRAMES="${MAX_FRAMES:-1016}"
SAMPLE_INDEX="${SAMPLE_INDEX:--1}"
GENERATE_RELEVANCE="${GENERATE_RELEVANCE:-0}" # query_feature 推理时生成 relevance 图
DINO_PRETRAINED_PATH="${DINO_PRETRAINED_PATH:-$REPO_ROOT/ckpt/dinov2-small}"
DINO_INPUT_SIZE="${DINO_INPUT_SIZE:-518}"
GRID_COLS="${GRID_COLS:-4}"

if [[ -z "$QUESTION_ID" ]]; then
  echo "请提供 QUESTION_ID，方式：QUESTION_ID=xxx bash eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.sh [method]"
  exit 1
fi

QUESTION_ID_SUFFIX="$QUESTION_ID"
if [[ "$QUESTION_ID" == *"sample_"* ]]; then
  QUESTION_ID_SUFFIX="${QUESTION_ID##*sample_}"
fi
QUESTION_ID_SUFFIX="${QUESTION_ID_SUFFIX// /_}"
QUESTION_ID_SUFFIX="${QUESTION_ID_SUFFIX//\//_}"

RUN_NAME="${RUN_NAME:-${METHOD}_sample${QUESTION_ID_SUFFIX}}"
RESULT_DIR="${RESULT_ROOT}/${METHOD}"
SIM_JSONL="${SIM_JSONL:-${RESULT_DIR}/similarity/${RUN_NAME}.jsonl}"
VIS_DIR="${VIS_DIR:-${RESULT_DIR}/visual/${RUN_NAME}}"
RELEVANCE_VIS_DIR="${RELEVANCE_VIS_DIR:-eval_qwen2.5vl/vis_streamingbench_relevance/query_feature/visual/${RUN_NAME}}"

mkdir -p "$RESULT_DIR" "$(dirname "$SIM_JSONL")" "$VIS_DIR"

case "$METHOD" in
  feature)
    DROP_THRESHOLD="${DROP_THRESHOLD:-0.75}"
    python eval_qwen2.5vl/scripts/streamingbench.py \
      --run_name "$RUN_NAME" \
      --drop_method feature \
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
    ;;
  query_feature)
    DROP_THRESHOLD="${DROP_THRESHOLD:-0.75}"
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
    ;;
  dynamic_feature)
    DROP_THRESHOLD="${DROP_THRESHOLD:-0.5}"
    SCENE_THRESHOLD="${SCENE_THRESHOLD:-0.6}"
    HIGH_SIM_THRESHOLD="${HIGH_SIM_THRESHOLD:-0.8}"
    HIGH_DROP_RATIO="${HIGH_DROP_RATIO:-0.8}"
    LOW_DROP_RATIO="${LOW_DROP_RATIO:-0.5}"
    python eval_qwen2.5vl/scripts/streamingbench_dynamic_feature.py \
      --run_name "$RUN_NAME" \
      --drop_method dynamic_feature \
      --drop_threshold "$DROP_THRESHOLD" \
      --drop_relative \
      --scene_threshold "$SCENE_THRESHOLD" \
      --high_sim_threshold "$HIGH_SIM_THRESHOLD" \
      --high_drop_ratio "$HIGH_DROP_RATIO" \
      --low_drop_ratio "$LOW_DROP_RATIO" \
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
    ;;
  dino_feature)
    DROP_THRESHOLD="${DROP_THRESHOLD:-0.5}"
    python eval_qwen2.5vl/scripts/streamingbench_dinov2.py \
      --run_name "$RUN_NAME" \
      --drop_method dino_feature \
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
      --max_frames "$MAX_FRAMES" \
      --dino_pretrained_path "$DINO_PRETRAINED_PATH" \
      --dino_input_size "$DINO_INPUT_SIZE"
    ;;
  *)
    echo "METHOD 仅支持: feature | query_feature | dynamic_feature | dino_feature"
    exit 1
    ;;
esac

python eval_qwen3vl/scripts/visualize_streamingbench_similarity.py \
  --method "$METHOD" \
  --sim_jsonl "$SIM_JSONL" \
  --output_dir "$VIS_DIR" \
  --sample_index "$SAMPLE_INDEX" \
  --task_csv "$TASK_CSV" \
  --video_dir "$VIDEO_DIR" \
  --question_id "$QUESTION_ID" \
  --min_pixels "$MIN_PIXELS" \
  --max_pixels "$MAX_PIXELS" \
  --min_frames "$MIN_FRAMES" \
  --max_frames "$MAX_FRAMES" \
  --grid_cols "$GRID_COLS"

if [[ "$METHOD" == "query_feature" && "$GENERATE_RELEVANCE" == "1" ]]; then
  mkdir -p "$RELEVANCE_VIS_DIR"
  python eval_qwen3vl/scripts/visualize_streamingbench_relevance.py \
    --sim_jsonl "$SIM_JSONL" \
    --output_dir "$RELEVANCE_VIS_DIR" \
    --sample_index "$SAMPLE_INDEX" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --question_id "$QUESTION_ID" \
    --method query_feature \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --min_frames "$MIN_FRAMES" \
    --max_frames "$MAX_FRAMES" \
    --grid_cols "$GRID_COLS"
fi

echo "similarity json: $SIM_JSONL"
echo "visual output: $VIS_DIR"
if [[ "$METHOD" == "query_feature" && "$GENERATE_RELEVANCE" == "1" ]]; then
  echo "relevance visual output: $RELEVANCE_VIS_DIR"
fi

# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.sh feature
# QUESTION_ID=你的question_id bash eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.sh query_feature

# QUESTION_ID=你的question_id GENERATE_RELEVANCE=1 bash eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.sh query_feature

# QUESTION_ID=你的question_id bash eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.sh dynamic_feature

# 单独可视化已有相似度 json：
# python3 eval_qwen2.5vl/scripts/visualize_streamingbench_similarity.py \
#   --method query_feature \
#   --sim_jsonl your_similarity.jsonl \
#   --output_dir your_output_dir \
#   --task_csv "$TASK_CSV" \
#   --video_dir "$VIDEO_DIR" \
#   --question_id "$QUESTION_ID"