#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export TASK_JSON="${TASK_JSON:-${REPO_ROOT}/datasets/OVO-Bench/ovo_bench_new.json}"
VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/datasets/OVO-Bench/src_videos}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
RESULT_DIR="${RESULT_DIR:-eval_qwen3vl/result_ovobench}"
LOG_DIR="${LOG_DIR:-eval_qwen3vl/result_ovobench/log-preencoder-drop20}"

METHODS="${METHODS:-ahash dhash phash ssim frame_diff lite_flow scene_change}"
DROP_RATIO="${DROP_RATIO:-0.2}"
FRAME_SELECT_SIZE="${FRAME_SELECT_SIZE:-32}"
FRAME_SELECT_MIN_KEEP_FRAMES="${FRAME_SELECT_MIN_KEEP_FRAMES:-4}"

TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
FPS="${FPS:-1}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-200704}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

USE_SUBSET="${USE_SUBSET:-0}"
export SAMPLE_FRAC="${SAMPLE_FRAC:-0.05}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export SAMPLED_TASK_JSON="${RESULT_DIR}/tmp/ovobench_5pct_seed${SAMPLE_SEED}.json"

OVO_TASK_GROUP="${OVO_TASK_GROUP:-realtime}"
export OVO_TASK_GROUP
export FILTERED_TASK_JSON="${RESULT_DIR}/tmp/ovobench_${OVO_TASK_GROUP}_preencoder_eval.json"
TASK_JSON_EVAL="$TASK_JSON"

if [[ ! -d "$CKPT_PATH" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}/tmp" "$LOG_DIR"

if [[ "${USE_SUBSET}" == "1" || "${USE_SUBSET}" == "true" || "${USE_SUBSET}" == "True" ]]; then
    python - <<'PY_SAMPLE'
import json
import os
import random

task_json = os.environ["TASK_JSON"]
sampled_task_json = os.environ["SAMPLED_TASK_JSON"]
sample_frac = float(os.environ["SAMPLE_FRAC"])
sample_seed = int(os.environ["SAMPLE_SEED"])

with open(task_json, "r") as f:
    data = json.load(f)

rng = random.Random(sample_seed)
sample_size = max(1, round(len(data) * sample_frac))
sampled_data = rng.sample(data, sample_size)
sampled_data.sort(key=lambda x: x.get("id", 0))

with open(sampled_task_json, "w") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"Saved sampled json: {sampled_task_json}")
print(f"Sampled rows: {len(sampled_data)} / {len(data)} ({len(sampled_data)/len(data):.2%})")
PY_SAMPLE
    TASK_JSON_EVAL="$SAMPLED_TASK_JSON"
else
    echo "Use full OVO-Bench json: ${TASK_JSON}"
fi

if [[ "${OVO_TASK_GROUP}" != "all" ]]; then
    export TASK_JSON_EVAL
    python - <<'PY_FILTER'
import json
import os

task_groups = {
    "backward": {"EPM", "ASI", "HLD"},
    "realtime": {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"},
    "forward": {"REC", "SSR", "CRR"},
}

group = os.environ["OVO_TASK_GROUP"].strip().lower()
if group not in task_groups:
    raise SystemExit(f"Unknown OVO_TASK_GROUP={group}, expected one of all/backward/realtime/forward")

with open(os.environ["TASK_JSON_EVAL"], "r") as f:
    data = json.load(f)

filtered = [item for item in data if item.get("task") in task_groups[group]]
with open(os.environ["FILTERED_TASK_JSON"], "w") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Saved filtered json: {os.environ['FILTERED_TASK_JSON']}")
print(f"Filtered rows: {len(filtered)} / {len(data)} ({group})")
PY_FILTER
    TASK_JSON_EVAL="$FILTERED_TASK_JSON"
fi

LOADER_ARGS=(
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
)
if [[ "${PIN_MEMORY}" == "1" || "${PIN_MEMORY}" == "true" || "${PIN_MEMORY}" == "True" ]]; then
    LOADER_ARGS+=(--pin_memory)
fi
if [[ "${PERSISTENT_WORKERS}" == "1" || "${PERSISTENT_WORKERS}" == "true" || "${PERSISTENT_WORKERS}" == "True" ]]; then
    LOADER_ARGS+=(--persistent_workers)
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OVO_TASK_GROUP=${OVO_TASK_GROUP}"
echo "Task json for eval: ${TASK_JSON_EVAL}"
echo "Methods: ${METHODS}"

for METHOD in ${METHODS}; do
    RUN_NAME="${RUN_NAME_PREFIX:-8B_preencoder_drop20}_${METHOD}_T${TEMPERATURE}"
    echo "========== OVO-Bench pre-encoder experiment =========="
    echo "RUN_NAME=${RUN_NAME}"
    echo "METHOD=${METHOD}"
    echo "DROP_RATIO=${DROP_RATIO}"
    echo "GPU=${CUDA_VISIBLE_DEVICES}"

    python eval_qwen3vl/scripts/qwen3_vl_clip/ovobench_hdiff_parallel.py \
        --run_name "$RUN_NAME" \
        --drop_method none \
        --drop_threshold 0 \
        --ckpt_path "$CKPT_PATH" \
        --task_json "$TASK_JSON_EVAL" \
        --video_dir "$VIDEO_DIR" \
        --result_dir "$RESULT_DIR" \
        --log_dir "$LOG_DIR" \
        --min_pixels "$MIN_PIXELS" \
        --max_pixels "$MAX_PIXELS" \
        --max_frames "$MAX_FRAMES" \
        --fps "$FPS" \
        --temperature "$TEMPERATURE" \
        "${LOADER_ARGS[@]}" \
        --frame_select_enabled \
        --frame_select_method "$METHOD" \
        --frame_select_drop_ratio "$DROP_RATIO" \
        --frame_select_size "$FRAME_SELECT_SIZE" \
        --frame_select_min_keep_frames "$FRAME_SELECT_MIN_KEEP_FRAMES"
done
