#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash $0 <question_id> [<question_id2> ...]" >&2
  exit 1
fi

TASK_CSV="${TASK_CSV:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${VIDEO_DIR:-/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding}"
CKPT_PATH="${CKPT_PATH:-/data/ckpts/Qwen3-VL-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_qwen3vl/vis_attention_map}"

ALPHA="${ALPHA:-0.5}"
CMAP="${CMAP:-jet}"

MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MAX_FRAMES="${MAX_FRAMES:-1016}"
FPS="${FPS:-1}"
GRID_COLS="${GRID_COLS:-4}"

DROP_ARGS=""

TOTAL=$#
IDX=0

for QUESTION_ID in "$@"; do
  IDX=$((IDX + 1))
  echo "========================================="
  echo "[$IDX/$TOTAL] Processing: $QUESTION_ID"
  echo "========================================="

  QID_SUFFIX="$QUESTION_ID"
  if [[ "$QUESTION_ID" == *"sample_"* ]]; then
    QID_SUFFIX="${QUESTION_ID##*sample_}"
  fi
  QID_SUFFIX="${QID_SUFFIX// /_}"
  QID_SUFFIX="${QID_SUFFIX//\//_}"

  OUTPUT_DIR="${OUTPUT_ROOT}/sample_${QID_SUFFIX}"

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  OUTPUT_JSONL="${OUTPUT_DIR}/feature_0_${TIMESTAMP}.jsonl"

  python eval_qwen3vl/scripts/visualize_attention_map.py \
      --question_id "$QUESTION_ID" \
      --ckpt_path "$CKPT_PATH" \
      --task_csv "$TASK_CSV" \
      --video_dir "$VIDEO_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --alpha "$ALPHA" \
      --cmap "$CMAP" \
      --min_pixels "$MIN_PIXELS" \
      --max_pixels "$MAX_PIXELS" \
      --min_frames "$MIN_FRAMES" \
      --max_frames "$MAX_FRAMES" \
      --fps "$FPS" \
      --grid_cols "$GRID_COLS" \
      --output_jsonl "$OUTPUT_JSONL" \
      $DROP_ARGS

  echo "Output dir:   $OUTPUT_DIR"
  echo "Output JSONL: $OUTPUT_JSONL"
done

echo "========================================="
echo "All $TOTAL sample(s) done."
echo "========================================="

# Single sample:
# bash eval_qwen3vl/scripts/visualize_attention_map.sh "Real-Time Visual Understanding_sample_1_1"
#
# Batch:
# bash eval_qwen3vl/scripts/visualize_attention_map.sh "Real-Time Visual Understanding_sample_1_1" "Real-Time Visual Understanding_sample_20_2"
