#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export TASK_JSON="${REPO_ROOT}/datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR="${REPO_ROOT}/datasets/OVO-Bench/src_videos"

RUN_NAME="${RUN_NAME:-8B_feature_0.5_T0.7}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
RESULT_DIR="eval_qwen3vl/result_ovobench"
LOG_DIR="${LOG_DIR:-eval_qwen3vl/result_ovobench/log-full}"


TEMPERATURE=0.7
DROP_RELATIVE_HDIFF="${DROP_RELATIVE_HDIFF:-1}" # 1 时使用相对筛选：按相邻帧 hist 差排序，保留 top k；HIST_THRESHOLD 表示丢弃比例。

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

OVO_TASK_GROUP="${OVO_TASK_GROUP:-realtime}" # all/backward/realtime/forward
export OVO_TASK_GROUP
export FILTERED_TASK_JSON="${RESULT_DIR}/tmp/ovobench_${OVO_TASK_GROUP}_eval.json"
TASK_JSON_EVAL="$TASK_JSON"

HIST_ENABLED="${HIST_ENABLED:-0}"
HIST_BINS=32
HIST_THRESHOLD="${HIST_THRESHOLD:-0.2}"
HIST_MIN_KEEP_FRAMES=4
HIST_MAX_INTERVAL=30

DROP_METHOD=feature
DROP_THRESHOLD=0.5
DROP_RELATIVE_FEATURE="${DROP_RELATIVE_FEATURE:-1}"

ENABLE_ATTN_V2="${ENABLE_ATTN_V2:-0}"
ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.375}"
TOP_K_RATIO="${TOP_K_RATIO:-$(awk "BEGIN {print 1 - $ATTN_DROP_RATIO}")}"
SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"

ATTN_ARGS=()
if [[ "${ENABLE_ATTN_V2}" == "1" || "${ENABLE_ATTN_V2}" == "true" || "${ENABLE_ATTN_V2}" == "True" ]]; then
    DROP_METHOD=attention
    DROP_THRESHOLD=0
    ATTN_ARGS+=(
        --attn_layer_idx "$ATTN_LAYER_IDX"
        --top_k_ratio "$TOP_K_RATIO"
        --scorer_chunk_size "$SCORER_CHUNK_SIZE"
    )
fi

HIST_ARGS=()
if [[ "${HIST_ENABLED}" == "1" ]]; then
    HIST_ARGS+=(--hist_enabled)
fi

DROP_ARGS=()
if [[ "${DROP_RELATIVE_FEATURE}" == "1" || "${DROP_RELATIVE_FEATURE}" == "true" || "${DROP_RELATIVE_FEATURE}" == "True" ]]; then
    DROP_ARGS+=(--drop_relative)
fi
if [[ "${DROP_RELATIVE_HDIFF}" == "1" || "${DROP_RELATIVE_HDIFF}" == "true" || "${DROP_RELATIVE_HDIFF}" == "True" ]]; then
    DROP_ARGS+=(--drop_relative_hdiff)
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

LOG_ARGS=()
if [[ -n "${LOG_DIR}" ]]; then
    LOG_ARGS+=(--log_dir "$LOG_DIR")
fi

mkdir -p "${RESULT_DIR}/tmp"

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

echo "OVO_TASK_GROUP=${OVO_TASK_GROUP}"
echo "Task json for eval: ${TASK_JSON_EVAL}"

python eval_qwen3vl/scripts/qwen3_vl_clip/ovobench_hdiff_parallel.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON_EVAL" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    "${LOG_ARGS[@]}" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --max_frames "$MAX_FRAMES" \
    --fps "$FPS" \
    --temperature "$TEMPERATURE" \
    "${LOADER_ARGS[@]}" \
    "${DROP_ARGS[@]}" \
    "${HIST_ARGS[@]}" \
    --hist_bins $HIST_BINS \
    --hist_threshold $HIST_THRESHOLD \
    --hist_min_keep_frames $HIST_MIN_KEEP_FRAMES \
    --hist_max_interval $HIST_MAX_INTERVAL \
    "${ATTN_ARGS[@]}"
