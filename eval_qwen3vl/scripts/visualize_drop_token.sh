#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

METHOD="${1:-feature}"
QUESTION_ID="${QUESTION_ID:-${2:-}}"
TASK_CSV="${TASK_CSV:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding}"
CKPT_PATH="${CKPT_PATH:-ckpt/Qwen2.5-VL-7B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-eval_qwen2.5vl/vis_drop_token}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MAX_FRAMES="${MAX_FRAMES:-1016}"
DIM_FACTOR="${DIM_FACTOR:-0.3}"
SAMPLE_INDEX="${SAMPLE_INDEX:--1}"
DINO_PRETRAINED_PATH="${DINO_PRETRAINED_PATH:-$REPO_ROOT/ckpt/dinov2-small}"
DINO_INPUT_SIZE="${DINO_INPUT_SIZE:-518}"
GRID_COLS="${GRID_COLS:-4}"

if [[ -z "$QUESTION_ID" ]]; then
  echo "请提供 QUESTION_ID，方式：QUESTION_ID=xxx bash eval_qwen2.5vl/scripts/visualize_drop_token.sh [method]"
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
DP_JSONL="${DP_JSONL:-${RESULT_DIR}/drop_pos/${RUN_NAME}.jsonl}"
VIS_DIR="${VIS_DIR:-${RESULT_DIR}/visual/${RUN_NAME}}"

mkdir -p "$RESULT_DIR" "$(dirname "$SIM_JSONL")" "$(dirname "$DP_JSONL")" "$VIS_DIR"

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
      --dp_save_path "$DP_JSONL" \
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
      --dp_save_path "$DP_JSONL" \
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
      --dp_save_path "$DP_JSONL" \
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
      --dp_save_path "$DP_JSONL" \
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

python eval_qwen3vl/scripts/visualize_drop_token.py \
  --method "$METHOD" \
  --dp_jsonl "$DP_JSONL" \
  --sim_jsonl "$SIM_JSONL" \
  --output_dir "$VIS_DIR" \
  --sample_index "$SAMPLE_INDEX" \
  --dim_factor "$DIM_FACTOR" \
  --task_csv "$TASK_CSV" \
  --video_dir "$VIDEO_DIR" \
  --question_id "$QUESTION_ID" \
  --min_pixels "$MIN_PIXELS" \
  --max_pixels "$MAX_PIXELS" \
  --min_frames "$MIN_FRAMES" \
  --max_frames "$MAX_FRAMES" \
  --grid_cols "$GRID_COLS"

echo "similarity json: $SIM_JSONL"
echo "drop positions json: $DP_JSONL"
echo "drop token visual: $VIS_DIR"

# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_drop_token.sh feature
# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_drop_token.sh query_feature
# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_drop_token.sh dynamic_feature
# QUESTION_ID="Real-Time Visual Understanding_sample_1_1" bash eval_qwen2.5vl/scripts/visualize_drop_token.sh dino_feature

# 默认读取最后一条记录，可用 SAMPLE_INDEX=0 指定第一条

# 单独可视化已有 dp/sim json（跳过推理）：
# python eval_qwen2.5vl/scripts/visualize_drop_token.py \
#   --method feature \
#   --dp_jsonl path/to/drop_pos.jsonl \
#   --sim_jsonl path/to/similarity.jsonl \
#   --output_dir output/ \
#   --video_path path/to/video.mp4
