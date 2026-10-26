export CUDA_VISIBLE_DEVICES=0

TASK_CSV="./datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding_5pct.csv"
VIDEO_DIR="/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding"

RUN_NAME=fps_quick
CKPT_PATH="ckpt/Qwen3-VL-4B-Instruct"
RESULT_DIR="eval_qwen3vl/result_streamingbench_fps"

# FPS Frame Selection arguments
FPS_ENABLED=1  # set to 1 to enable, 0 to disable pixel-diff frame selection
TAU_ADJ=5.0
TAU_REF=10.0
CONSECUTIVE_THRESHOLD=2
MAX_INTERVAL=10

FPS_ARGS=""
if [ "$FPS_ENABLED" -eq 1 ]; then
    FPS_ARGS="--fps_enabled --tau_adj $TAU_ADJ --tau_ref $TAU_REF --consecutive_threshold $CONSECUTIVE_THRESHOLD --max_interval $MAX_INTERVAL"
fi

python eval_qwen3vl/scripts/qwen3_vl_fps/streamingbench_fps.py \
    --run_name $RUN_NAME \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    $FPS_ARGS