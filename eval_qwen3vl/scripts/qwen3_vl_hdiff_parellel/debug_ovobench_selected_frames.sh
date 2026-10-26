#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/eval_qwen3vl/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/train/logs/ckpt/qwen3_vl_8b/ours/v28-20260517-000926/checkpoint-8500}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/selected_ovo_videos}"

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
import sys
import types

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

repo = os.getcwd()
sys.path.append(osp.join(repo, "eval_qwen3vl"))
from qwen3_vl_hdiff_parellel import Qwen3VLForConditionalGenerationDTD


def resolve_processor_path(ckpt_path):
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


def safe_name(path):
    return osp.splitext(osp.basename(path))[0]


def fps_name(fps):
    return str(fps).replace(".", "_")


def save_frames(frames, indices, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for order, local_idx in enumerate(indices):
        frame = frames[local_idx]
        image_path = osp.join(out_dir, f"{prefix}_{order:04d}_local_{local_idx:04d}.jpg")
        to_image(frame).save(image_path, quality=95)
        saved.append(image_path)
    return saved


def build_prompt(question, options):
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{question}\n"
        f"Options: A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "The best answer is:"
    )


def build_messages(case, fps):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": case["video"],
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "fps": fps,
                },
                {"type": "text", "text": build_prompt(case["question"], case["options"])},
            ],
        }
    ]


def decode_sampled_frames(processor, case, fps):
    messages = build_messages(case, fps)
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
output_root = os.environ["OUTPUT_ROOT"]
min_pixels = int(os.environ["MIN_PIXELS"])
max_pixels = int(os.environ["MAX_PIXELS"])
min_frames = int(os.environ["MIN_FRAMES"])
max_frames = int(os.environ["MAX_FRAMES"])
attn_layer_idx = int(os.environ["ATTN_LAYER_IDX"])
top_k_ratio = float(os.environ["TOP_K_RATIO"])
scorer_chunk_size = int(os.environ["SCORER_CHUNK_SIZE"])

cases = [
    {
        "video": osp.join(repo, "selected_temporal_query_videos", "5wLv3pCqZ9o.mp4"),
        "question": "What was the next step taken by the operator after filling the bottle with oil?",
        "options": ["Applied glue to cork.", "Added blue food dye.", "Added 1/3 water.", "Put the cork on the bottle."],
        "answer": "A",
    },
    {
        "video": osp.join(repo, "selected_temporal_query_videos", "eno3UMEMQJI.mp4"),
        "question": "What did the male protagonist in the video do immediately after finishing his personal report at the meeting?",
        "options": ["Play games.", "Watch a friend unbox a watch.", "Have lunch.", "Chat with many friends."],
        "answer": "C",
    },
    {
        "video": osp.join(repo, "selected_temporal_query_videos", "3E4Zca1p64s.mp4"),
        "question": "What does the person in the video do after brushing his teeth?",
        "options": ["Driving a car.", "Spraying perfume.", "Drinking water.", "Charging the cell phone."],
        "answer": "B",
    },
]
for case in cases:
    if not osp.exists(case["video"]):
        case["video"] = osp.join(repo, "datasets", "Video-MME", "data", osp.basename(case["video"]))

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
for case_idx, case in enumerate(cases):
    case_name = f"{case_idx:02d}_{safe_name(case['video'])}"
    for fps in (1.0, 0.5):
        messages, image_inputs, video_tensor, metadata = decode_sampled_frames(processor, case, fps)
        frames = list(video_tensor)
        all_indices = list(range(len(frames)))
        keep_hdiff = frame_selector.select_frames(frames)
        capture.clear()
        keep_attn = run_attention(processor, model, messages, image_inputs, video_tensor, top_k_ratio, capture)

        base_dir = osp.join(output_root, case_name, f"fps_{fps_name(fps)}")
        save_frames(frames, all_indices, osp.join(base_dir, "raw"), "raw")
        save_frames(frames, keep_hdiff, osp.join(base_dir, "hdiff"), "hdiff")
        save_frames(frames, keep_attn, osp.join(base_dir, "attn"), "attn")
        record = {
            "case": case_name,
            "video": case["video"],
            "question": case["question"],
            "answer": case["answer"],
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
        print(f"{case_name} fps={fps}: raw={len(all_indices)} hdiff={len(keep_hdiff)} attn={len(keep_attn)} -> {base_dir}")

with open(osp.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Saved debug frames to: {output_root}")
PY
