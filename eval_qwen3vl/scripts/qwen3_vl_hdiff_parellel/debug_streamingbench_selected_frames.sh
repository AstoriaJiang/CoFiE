#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR="${QWENVL_SKIP_TORCHVISION_FALLBACK_ON_VIDEO_ERROR:-1}"
export CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/ckpt/Qwen3-VL-8B-Instruct}"
export TASK_CSV="${TASK_CSV:-${REPO_ROOT}/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
export VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/datasets/StreamingBench/Real-Time_Visual_Understanding}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/selected_streamingbench_videos}"

export QUESTION_ID="${QUESTION_ID:-"Real-Time Visual Understanding_sample_22_1"}"
export SAMPLE_ID="${SAMPLE_ID:-1}"
export BASE_FPS="${BASE_FPS:-1.0}"
export MIN_PIXELS="${MIN_PIXELS:-200704}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export MIN_FRAMES="${MIN_FRAMES:-4}"
export MAX_FRAMES="${MAX_FRAMES:-1000}"
export HIST_BINS="${HIST_BINS:-32}"
export HIST_THRESHOLD="${HIST_THRESHOLD:-0.2}"
export HIST_MIN_KEEP_FRAMES="${HIST_MIN_KEEP_FRAMES:-4}"
export HIST_MAX_INTERVAL="${HIST_MAX_INTERVAL:-30}"

export ATTN_LAYER_IDX="${ATTN_LAYER_IDX:--1}"
export ATTN_DROP_RATIO="${ATTN_DROP_RATIO:-0.375}"
export TOP_K_RATIO="${TOP_K_RATIO:-$(awk "BEGIN {print 1 - $ATTN_DROP_RATIO}")}"
export SCORER_CHUNK_SIZE="${SCORER_CHUNK_SIZE:-512}"

mkdir -p "${OUTPUT_ROOT}"

python - <<'PY'
import importlib.util
import json
import os
import os.path as osp
import re
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

repo = os.getcwd()
sys.path.append(osp.join(repo, "eval_qwen3vl"))
from qwen3_vl_hdiff_parellel import Qwen3VLForConditionalGenerationDTD

PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""


def resolve_processor_path(ckpt_path):
    if osp.isabs(ckpt_path) and not osp.exists(ckpt_path):
        raise FileNotFoundError(f"CKPT_PATH does not exist: {ckpt_path}")
    adapter_config_path = osp.join(ckpt_path, "adapter_config.json")
    if osp.isfile(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        return adapter_config.get("base_model_name_or_path") or ckpt_path
    return ckpt_path


def load_hdiff_module():
    package_name = "transformers.models.qwen3_vl_hdiff"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [osp.join(repo, "eval_qwen3vl", "qwen3_vl_hdiff_parellel")]
        sys.modules[package_name] = pkg
    module_name = f"{package_name}.video_processing_qwen3_vl"
    file_path = osp.join(repo, "eval_qwen3vl", "qwen3_vl_hdiff_parellel", "video_processing_qwen3_vl.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def time_to_seconds(time_str):
    fmt = "%M:%S" if len(str(time_str)) == 5 else "%H:%M:%S"
    time_obj = datetime.strptime(str(time_str), fmt)
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


def streaming_fps(time_stamp, base_fps):
    time_stamp_sec = time_to_seconds(time_stamp)
    if 300 < time_stamp_sec <= 600:
        return base_fps * 0.5
    if time_stamp_sec > 600:
        return base_fps * 0.2
    return base_fps


def to_image(frame):
    if torch.is_tensor(frame):
        frame = frame.detach().cpu()
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        arr = frame.numpy()
    else:
        arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.size and np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr).convert("RGB")


def safe_name(value):
    value = str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def fps_name(fps):
    return str(fps).replace(".", "_")


def save_frames(frames, indices, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for order, local_idx in enumerate(indices):
        image_path = osp.join(out_dir, f"{prefix}_{order:04d}_local_{local_idx:04d}.jpg")
        to_image(frames[local_idx]).save(image_path, quality=95)
        saved.append(image_path)
    return saved


def video_path_for(row, video_dir):
    question_id = row.question_id
    return osp.join(video_dir, f"sample_{question_id.split('_')[-2]}", "video.mp4")


def build_messages(row, fps):
    time_stamp_sec = time_to_seconds(row.time_stamp)
    video_path = video_path_for(row, video_dir)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "fps": fps,
                    "video_start": 0,
                    "video_end": time_stamp_sec,
                },
                {"type": "text", "text": PROMPT.format(row.question, "\n".join(eval(row.options)))},
            ],
        }
    ]


def decode_sampled_frames(processor, row, fps):
    messages = build_messages(row, fps)
    image_inputs, video_inputs_with_metadata = process_vision_info(
        messages,
        return_video_metadata=True,
        image_patch_size=processor.video_processor.patch_size,
    )
    video_tensor, metadata = video_inputs_with_metadata[0]
    return messages, image_inputs, video_tensor, metadata


def run_attention(processor, model, messages, image_inputs, video_tensor, top_k_ratio, capture):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=[video_tensor],
        padding=True,
        return_tensors="pt",
        do_sample_frames=True,
        num_frames=video_tensor.shape[0],
        fps=None,
        do_resize=False,
    ).to(torch.device("cuda"))
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            drop_method="attention",
            drop_threshold=0,
            drop_absolute=True,
            attn_layer_idx=attn_layer_idx,
            top_k_ratio=top_k_ratio,
            scorer_chunk_size=scorer_chunk_size,
        )
    return capture.pop("keep_indices", list(range(video_tensor.shape[0])))


ckpt_path = os.environ["CKPT_PATH"]
processor_path = resolve_processor_path(ckpt_path)
task_csv = os.environ["TASK_CSV"]
video_dir = os.environ["VIDEO_DIR"]
output_root = os.environ["OUTPUT_ROOT"]
question_id = os.environ.get("QUESTION_ID", "").strip()
sample_id = os.environ.get("SAMPLE_ID", "").strip()
base_fps = float(os.environ["BASE_FPS"])
min_pixels = int(os.environ["MIN_PIXELS"])
max_pixels = int(os.environ["MAX_PIXELS"])
min_frames = int(os.environ["MIN_FRAMES"])
max_frames = int(os.environ["MAX_FRAMES"])
attn_layer_idx = int(os.environ["ATTN_LAYER_IDX"])
top_k_ratio = float(os.environ["TOP_K_RATIO"])
scorer_chunk_size = int(os.environ["SCORER_CHUNK_SIZE"])

df = pd.read_csv(task_csv)
if question_id:
    df = df[df["question_id"] == question_id]
elif sample_id:
    df = df[df["question_id"].astype(str).str.contains(f"_sample_{sample_id}_", regex=False)]
else:
    raise ValueError("Please set QUESTION_ID or SAMPLE_ID")
if df.empty:
    raise ValueError(f"No StreamingBench rows found for QUESTION_ID={question_id!r}, SAMPLE_ID={sample_id!r}")

hdiff_module = load_hdiff_module()
hist_config = hdiff_module.HistogramFrameSelectorConfig(
    enabled=True,
    bins=int(os.environ["HIST_BINS"]),
    threshold=float(os.environ["HIST_THRESHOLD"]),
    relative=True,
    min_keep_frames=int(os.environ["HIST_MIN_KEEP_FRAMES"]),
    max_interval=int(os.environ["HIST_MAX_INTERVAL"]),
)
frame_selector = hdiff_module.HistogramFrameSelector(hist_config)

processor = AutoProcessor.from_pretrained(
    processor_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)
attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map="auto",
)

capture = {}
orig_token_drop = model.token_drop


def capture_token_drop(*args, **kwargs):
    if kwargs.get("method") == "attention" and kwargs.get("frame_weights") is not None:
        frame_weights = kwargs["frame_weights"][0]
        n_frames = len(frame_weights)
        if n_frames > 0:
            n_keep = max(1, min(n_frames, int(round(n_frames * float(kwargs.get("top_k_ratio", top_k_ratio))))))
            weights = torch.tensor(frame_weights)
            _, topk_idx = torch.topk(weights, n_keep, largest=True)
            capture["keep_indices"] = sorted(int(i) for i in topk_idx.tolist())
    return orig_token_drop(*args, **kwargs)


model.token_drop = capture_token_drop
summary = []
for _, row in df.iterrows():
    fps = streaming_fps(row.time_stamp, base_fps)
    messages, image_inputs, video_tensor, metadata = decode_sampled_frames(processor, row, fps)
    frames = list(video_tensor)
    all_indices = list(range(len(frames)))
    keep_hdiff = frame_selector.select_frames(frames)
    capture.clear()
    keep_attn = run_attention(processor, model, messages, image_inputs, video_tensor, top_k_ratio, capture)

    case_name = safe_name(row.question_id)
    base_dir = osp.join(output_root, case_name, f"fps_{fps_name(fps)}")
    save_frames(frames, all_indices, osp.join(base_dir, "raw"), "raw")
    save_frames(frames, keep_hdiff, osp.join(base_dir, "hdiff"), "hdiff")
    save_frames(frames, keep_attn, osp.join(base_dir, "attn"), "attn")
    record = {
        "question_id": row.question_id,
        "task_type": row.task_type,
        "video": video_path_for(row, video_dir),
        "question": row.question,
        "time_stamp": row.time_stamp,
        "answer": row.answer,
        "options": eval(row.options),
        "frames_required": row.frames_required,
        "temporal_clue_type": row.temporal_clue_type,
        "fps": fps,
        "video_metadata": dict(metadata),
        "raw_count": len(all_indices),
        "hdiff_count": len(keep_hdiff),
        "attn_count": len(keep_attn),
        "hdiff_keep_indices": keep_hdiff,
        "attn_keep_indices": keep_attn,
        "hist_config": hist_config.__dict__,
        "attn_layer_idx": attn_layer_idx,
        "top_k_ratio": top_k_ratio,
    }
    with open(osp.join(base_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    summary.append(record)
    print(f"{row.question_id} fps={fps}: raw={len(all_indices)} hdiff={len(keep_hdiff)} attn={len(keep_attn)} -> {base_dir}")

with open(osp.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Saved debug frames to: {output_root}")
PY
