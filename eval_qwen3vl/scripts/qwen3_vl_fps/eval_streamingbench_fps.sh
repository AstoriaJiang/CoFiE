export CUDA_VISIBLE_DEVICES=0

TASK_CSV="/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding"

RUN_NAME=fps
CKPT_PATH="ckpt/Qwen3-VL-4B-Instruct"
RESULT_DIR="eval_qwen3vl/result_streamingbench_fps"

# FPS Frame Selection arguments
FPS_ENABLED=1  # set to true to enable pixel-diff frame selection
TAU_ADJ=5.0 # 相邻帧差分阈值（d_adj）。当前帧和前一帧像素平均差超过它，才算“有明显变化”。
TAU_REF=10.0 # 参考帧差分阈值（d_ref）。当前帧和参考帧像素平均差超过它，才算“有明显变化”。
CONSECUTIVE_THRESHOLD=2 # 连续变化判定长度。需要连续多少帧都满足 d_adj > TAU_ADJ 才触发保留，避免单帧噪声误判。
MAX_INTERVAL=10 # 最大无保留间隔。距离上次保留帧达到这个值时，强制保留一帧，防止时间轴断裂。

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
