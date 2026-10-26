#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source eval_qwen3vl/.venv/bin/activate

# export CUDA_VISIBLE_DEVICES=0

export TASK_PARQUET="${TASK_PARQUET:-eval_qwen3vl/result_videomme_debug/videomme_none_T_hdiff_F.parquet}"
export VIDEO_DIR="${VIDEO_DIR:-./datasets/Video-MME/data/}"
export CKPT_PATH="${CKPT_PATH:-ckpt/Qwen3-VL-8B-Instruct}"
export DEBUG_DIR="${DEBUG_DIR:-/root/autodl-tmp/CoFiE/eval_qwen3vl/result_videomme_debug}"
export DEBUG_MAX_ROWS="${DEBUG_MAX_ROWS:-21}"

export FPS="${FPS:-1}"
export MIN_FRAMES="${MIN_FRAMES:-4}"
export MAX_FRAMES="${MAX_FRAMES:-1000}"
export MIN_PIXELS="${MIN_PIXELS:-200704}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"

# Model generation temperature
export TEMPERATURE="${TEMPERATURE:-0}"

# Seed mode
export SEED_MODE="${SEED_MODE:-per_sample_time}" # global or per_sample_time
# Global seed
export GLOBAL_SEED="${GLOBAL_SEED:-1234}"

# Histogram configuration
export HIST_BINS="${HIST_BINS:-64}"
export HIST_THRESHOLD="${HIST_THRESHOLD:-0.5}"
export HIST_MIN_KEEP_FRAMES="${HIST_MIN_KEEP_FRAMES:-4}"
export HIST_MAX_INTERVAL="${HIST_MAX_INTERVAL:-30}"

export RUN_NAME="${RUN_NAME:-hdiff_frames_T_${TEMPERATURE}_${SEED_MODE}_$(date +%Y%m%d)_$(date +%H%M%S)}"

mkdir -p "${DEBUG_DIR}"

python - <<'PY'
import importlib.util
import json
import os
import os.path as osp
import sys
import time
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
from qwen3_vl import Qwen3VLForConditionalGenerationDTD

package_name = "transformers.models.qwen3_vl_hdiff"
if package_name not in sys.modules:
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [osp.join(repo, "eval_qwen3vl", "qwen3_vl_hdiff")]
    sys.modules[package_name] = pkg

module_name = f"{package_name}.video_processing_qwen3_vl"
file_path = osp.join(repo, "eval_qwen3vl", "qwen3_vl_hdiff", "video_processing_qwen3_vl.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
Qwen3VLVideoProcessor = module.Qwen3VLVideoProcessor

task_parquet = os.environ["TASK_PARQUET"]
video_dir = os.environ["VIDEO_DIR"]
ckpt_path = os.environ["CKPT_PATH"]
debug_dir = os.environ["DEBUG_DIR"]
max_rows = int(os.environ["DEBUG_MAX_ROWS"])
fps = int(os.environ["FPS"])
min_frames = int(os.environ["MIN_FRAMES"])
max_frames = int(os.environ["MAX_FRAMES"])
min_pixels = int(os.environ["MIN_PIXELS"])
max_pixels = int(os.environ["MAX_PIXELS"])
temperature = float(os.environ["TEMPERATURE"])
seed_mode = os.environ["SEED_MODE"]
global_seed = int(os.environ["GLOBAL_SEED"])
if seed_mode not in {"global", "per_sample_time"}:
    raise ValueError("SEED_MODE must be global or per_sample_time")

hist_config = {
    "enabled": True,
    "bins": int(os.environ["HIST_BINS"]),
    "threshold": float(os.environ["HIST_THRESHOLD"]),
    "min_keep_frames": int(os.environ["HIST_MIN_KEEP_FRAMES"]),
    "max_interval": int(os.environ["HIST_MAX_INTERVAL"]),
}

run_dir = osp.join(debug_dir, os.environ["RUN_NAME"])
frames_root = osp.join(run_dir, "frames")
os.makedirs(frames_root, exist_ok=True)
log_path = osp.join(run_dir, "selected_frames.jsonl")
infer_path = osp.join(run_dir, "inference_results.jsonl")
summary_path = osp.join(run_dir, "summary_stats.log")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(global_seed)
attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    ckpt_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)
original_size = processor.video_processor.size
processor.video_processor = Qwen3VLVideoProcessor.from_pretrained(ckpt_path, frame_selector_config=hist_config)
processor.video_processor.size = original_size
df = pd.read_parquet(task_parquet).head(max_rows)

prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""

def extract_answer(text):
    text = text.strip()
    for answer_prefix in (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ):
        text = text.replace(answer_prefix, "")
    for char in text:
        if char in "ABCD":
            return char
    return ""

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

summary_records = []
generation_kwargs = {"do_sample": False}
generation_mode = "greedy"
if temperature != 0:
    generation_kwargs = {"do_sample": True, "temperature": temperature}
    generation_mode = "sampling"

with open(log_path, "w", encoding="utf-8") as log_file:
    for row_idx, row in enumerate(df.itertuples(index=False)):
        video_id = getattr(row, "video_id", str(row_idx))
        videoID = getattr(row, "videoID", video_id)
        video_path = osp.join(video_dir, f"{videoID}.mp4")
        frame_dir = osp.join(frames_root, str(video_id))
        os.makedirs(frame_dir, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "max_frames": max_frames,
                        "min_frames": min_frames,
                        "fps": fps,
                    },
                    {"type": "text", "text": prompt_template.format(row.question, "\n".join(row.options.tolist()))},
                ],
            }
        ]
        image_inputs, video_inputs_with_metadata = process_vision_info(
            messages,
            return_video_metadata=True,
            image_patch_size=processor.video_processor.patch_size,
        )
        video_inputs = []
        process_video_metadata = []
        for video_input in video_inputs_with_metadata or []:
            video_tensor, video_metadata = video_input
            video_inputs.append(video_tensor)
            process_video_metadata.append(video_metadata)

        state = {}
        orig_select_frames = processor.video_processor.frame_selector.select_frames
        orig_sample_frames = processor.video_processor.sample_frames

        def debug_select_frames(frames, _orig=orig_select_frames):
            keep_local_indices = _orig(frames)
            state["sampled_frames"] = frames
            state["keep_local_indices"] = [int(i) for i in keep_local_indices]
            return keep_local_indices

        def debug_sample_frames(*args, **kwargs):
            metadata = kwargs.get("metadata")
            if metadata is None and args:
                metadata = args[0]
            total_num_frames = metadata.total_num_frames
            metadata_fps = metadata.fps or 24
            n = int(total_num_frames / metadata_fps * fps)
            n = min(max(n, min_frames), max_frames, total_num_frames)
            state["processor_uniform_indices"] = np.linspace(0, total_num_frames - 1, n).round().astype(int).tolist()
            indices = _orig_sample_frames(*args, **kwargs)
            state["processor_sample_indices"] = [int(i) for i in indices]
            state["total_num_frames"] = int(total_num_frames)
            state["metadata_fps"] = float(metadata_fps)
            return indices

        _orig_sample_frames = orig_sample_frames
        processor.video_processor.frame_selector.select_frames = debug_select_frames
        processor.video_processor.sample_frames = debug_sample_frames
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                do_sample_frames=True,
                num_frames=video_inputs[0].shape[0],
                fps=None,
                do_resize=False,
            )
        finally:
            processor.video_processor.frame_selector.select_frames = orig_select_frames
            processor.video_processor.sample_frames = orig_sample_frames

        inputs = inputs.to(torch.device("cuda"))
        sample_seed = global_seed if seed_mode == "global" else time.time_ns() % (2 ** 32)
        set_seed(sample_seed)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            **generation_kwargs,
            drop_method="none",
            drop_threshold=0,
            drop_absolute=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        pred = extract_answer(response)
        answer = getattr(row, "answer", None)
        is_correct = bool(pred and answer and pred == answer)

        sampled_frames = state.get("sampled_frames", [])
        keep_local_indices = state.get("keep_local_indices", [])
        processor_uniform_indices = state.get("processor_uniform_indices", [])
        processor_sample_indices = state.get("processor_sample_indices", [])
        process_frame_indices = process_video_metadata[0].get("frames_indices", []) if process_video_metadata else []
        process_frame_indices = [int(i) for i in process_frame_indices]
        original_frame_count = int(process_video_metadata[0].get("total_num_frames")) if process_video_metadata else state.get("total_num_frames")
        decoded_sampled_frame_count = len(sampled_frames)
        kept_frame_count = len(keep_local_indices)
        kept_frame_indices = []
        saved_frames = []
        for order, local_idx in enumerate(keep_local_indices):
            processor_idx = processor_sample_indices[local_idx] if local_idx < len(processor_sample_indices) else local_idx
            frame_index = process_frame_indices[processor_idx] if processor_idx < len(process_frame_indices) else processor_idx
            kept_frame_indices.append(int(frame_index))
            image_path = osp.join(frame_dir, f"kept_{order:03d}_frame_{frame_index:06d}.jpg")
            to_image(sampled_frames[local_idx]).save(image_path, quality=95)
            saved_frames.append(image_path)

        record = {
            "row_idx": row_idx,
            "video_id": video_id,
            "videoID": videoID,
            "video_path": video_path,
            "total_num_frames": original_frame_count,
            "decoded_sampled_frame_count": decoded_sampled_frame_count,
            "kept_frame_count": kept_frame_count,
            "is_correct": is_correct,
            "metadata_fps": float(process_video_metadata[0].get("fps")) if process_video_metadata else None,
            "process_vision_info_frame_indices": process_frame_indices,
            "processor_frame_indices": processor_uniform_indices,
            "processor_sample_indices": processor_sample_indices,
            "hdiff_frame_indices": keep_local_indices,
            # "kept_local_indices": keep_local_indices,
            "kept_origin_frame_abs_indices": kept_frame_indices,
            # "saved_frames": saved_frames,
            "hist_config": hist_config,
            "temperature": temperature,
            "generation_mode": generation_mode,
            "seed_mode": seed_mode,
            "seed": sample_seed,
        }
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(infer_path, "a", encoding="utf-8") as infer_file:
            infer_file.write(json.dumps({
                "row_idx": row_idx,
                "video_id": video_id,
                "videoID": videoID,
                "question_id": getattr(row, "question_id", None),
                "question": row.question,
                "options": row.options.tolist(),
                "answer": answer,
                "response": response,
                "pred": pred,
                "is_correct": is_correct,
                "temperature": temperature,
                "generation_mode": generation_mode,
                "seed_mode": seed_mode,
                "seed": sample_seed,
            }, ensure_ascii=False) + "\n")
        summary_records.append({
            "row_idx": row_idx,
            "question_id": getattr(row, "question_id", None),
            "video_id": video_id,
            "videoID": videoID,
            "original_frame_count": original_frame_count,
            "decoded_sampled_frame_count": decoded_sampled_frame_count,
            "kept_frame_count": kept_frame_count,
            "answer": answer,
            "pred": pred,
            "is_correct": is_correct,
            "seed": sample_seed,
        })
        print(f"[{row_idx + 1}/{len(df)}] {videoID}: kept {len(keep_local_indices)} frames -> {frame_dir}")
        print(f"prediction: {pred} | answer: {answer} | response: {response}")

total_original_frames = sum(r["original_frame_count"] or 0 for r in summary_records)
total_decoded_sampled_frames = sum(r["decoded_sampled_frame_count"] for r in summary_records)
total_kept_frames = sum(r["kept_frame_count"] for r in summary_records)
correct_count = sum(1 for r in summary_records if r["is_correct"])
compression_rate = total_kept_frames / total_decoded_sampled_frames if total_decoded_sampled_frames else 0.0
compression_percent = compression_rate * 100
accuracy = correct_count / len(summary_records) * 100 if summary_records else 0.0

with open(summary_path, "w", encoding="utf-8") as summary_file:
    summary_file.write(
        f"temperature={temperature}\tgeneration_mode={generation_mode}\t"
        f"seed_mode={seed_mode}\tglobal_seed={global_seed}\n"
    )
    for r in summary_records:
        sample_compression_rate = (
            r["kept_frame_count"] / r["decoded_sampled_frame_count"]
            if r["decoded_sampled_frame_count"]
            else 0.0
        )
        sample_compression_percent = sample_compression_rate * 100
        summary_file.write(
            "question_id={question_id}\tvideo_id={video_id}\tvideoID={videoID}\t"
            "original_frames={original_frame_count}\tdecoded_sampled_frames={decoded_sampled_frame_count}\t"
            "kept_frames={kept_frame_count}\t"
            "compression_rate={sample_compression_percent:.2f}%\tanswer={answer}\t"
            "pred={pred}\tcorrect={is_correct}\tseed={seed}\n".format(
                sample_compression_percent=sample_compression_percent,
                **r,
            )
        )
    summary_file.write(
        "OVERALL\tsamples={samples}\ntotal_original_frames={total_original_frames}\t"
        "total_decoded_sampled_frames={total_decoded_sampled_frames}\t"
        "total_kept_frames={total_kept_frames}\tcompression_rate={compression_percent:.2f}%\t"
        "correct={correct_count}\taccuracy={accuracy:.2f}%\n".format(
            samples=len(summary_records),
            total_original_frames=total_original_frames,
            total_decoded_sampled_frames=total_decoded_sampled_frames,
            total_kept_frames=total_kept_frames,
            compression_percent=compression_percent,
            correct_count=correct_count,
            accuracy=accuracy,
        )
    )

print(f"Log saved to: {log_path}")
print(f"Inference saved to: {infer_path}")
print(f"Summary saved to: {summary_path}")
print(f"Frames saved to: {frames_root}")
PY

exit 0
