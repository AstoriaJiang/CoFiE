export CUDA_VISIBLE_DEVICES=0

TASK_PARQUET="./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="./datasets/Video-MME/data/"

RUN_NAME=fps_baseline
CKPT_PATH="ckpt/Qwen3-VL-4B-Instruct"
RESULT_DIR="eval_qwen3vl/result_videomme_fps"

# FPS Frame Selection arguments
FPS_ENABLED=1  # set to true to enable pixel-diff frame selection
TAU_ADJ=5.0
TAU_REF=10.0
CONSECUTIVE_THRESHOLD=3
MAX_INTERVAL=30

FPS_ARGS=""
if [ "$FPS_ENABLED" -eq 1 ]; then
    FPS_ARGS="--fps_enabled --tau_adj $TAU_ADJ --tau_ref $TAU_REF --consecutive_threshold $CONSECUTIVE_THRESHOLD --max_interval $MAX_INTERVAL"
fi

python eval_qwen3vl/scripts/qwen3_vl_fps/videomme_fps.py \
    --run_name $RUN_NAME \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 1000 \
    --fps 1 \
    $FPS_ARGS
