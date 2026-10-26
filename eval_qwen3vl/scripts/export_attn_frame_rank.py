"""
Export per-frame attention weights and frame ranking from a specified LM layer.

Supports both single-sample (``--question_id``) and batch modes.
Outputs ``t2v_attention.json``, ``frame_rank.txt``, ``keep_frames.json``
into per-sample sub-directories under ``--output_dir``.
"""

import argparse
import inspect
import json
import os
import os.path as osp
import sys
from datetime import datetime

import ffmpeg
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl import Qwen3VLForConditionalGenerationDTD
from qwen3_vl_attn import LayerAttentionCollector, select_topk_frames, save_results

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016

PROMPT = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a "
    "multiple-choice question related to the video. Your task is to "
    "carefully analyze the video and provide the best answer to question, "
    "choosing from the four options provided. Respond with only the letter "
    "(A, B, C, or D) of the correct option.\n\n"
    "Question: {}\n\nOptions:\n{}\n\nThe best option is:"
)


def time_to_seconds(time_str):
    if len(time_str) == 5:
        t = datetime.strptime(time_str, '%M:%S')
    else:
        t = datetime.strptime(time_str, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


def split_video(video_file, start_time, end_time):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        return output_file
    (
        ffmpeg.input(video_file, ss=int(start_time))
        .output(output_file, t=int(end_time) - int(start_time),
                vcodec='libx264', acodec='aac')
        .global_args('-nostdin')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return output_file


def process_vision_info_with_video_kwargs(messages):
    sig = inspect.signature(process_vision_info)
    if "return_video_kwargs" in sig.parameters:
        return process_vision_info(messages, return_video_kwargs=True)
    img, vid = process_vision_info(messages)
    return img, vid, {"do_sample_frames": False}


def make_sample_dir(output_root, question_id):
    qid_suffix = question_id.split("sample_")[-1]
    qid_suffix = qid_suffix.replace(" ", "_").replace("/", "_")
    return osp.join(output_root, f"sample_{qid_suffix}")


def process_one(row, model, processor, collector, args):
    question_id = row.question_id
    video_path = osp.join(
        args.video_dir,
        f"sample_{question_id.split('_')[-2]}",
        "video.mp4",
    )
    ts = time_to_seconds(row.time_stamp)
    video_path = split_video(video_path, 0, ts)

    fps = 1
    if 300 < ts <= 600:
        fps = 0.5
    elif ts > 600:
        fps = 0.2
    if args.fps is not None:
        fps = args.fps

    options_list = eval(row.options) if isinstance(row.options, str) else row.options
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
             "max_frames": args.max_frames, "min_frames": args.min_frames,
             "fps": fps},
            {"type": "text",
             "text": PROMPT.format(row.question, '\n'.join(options_list))},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = \
        process_vision_info_with_video_kwargs(messages)
    video_kwargs = dict(video_kwargs)
    video_kwargs.setdefault("do_sample_frames", False)
    fv = video_kwargs.get("fps")
    if isinstance(fv, (list, tuple)) and fv:
        video_kwargs["fps"] = fv[0]
    elif fv is None:
        video_kwargs["fps"] = fps

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to('cuda')

    collector.reset()
    collector.parse_from_input_ids(
        inputs['input_ids'], inputs.get('video_grid_thw'), model.config)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    frame_weights = collector.get_frame_weights()
    keep_indices, drop_indices = select_topk_frames(
        frame_weights, top_k_ratio=args.top_k_ratio)

    sample_dir = make_sample_dir(args.output_dir, question_id)
    save_results(
        sample_dir, frame_weights, keep_indices, drop_indices,
        layer_idx=args.attn_layer_idx,
        top_k_ratio=args.top_k_ratio,
        question_id=question_id,
    )
    print(f"[{question_id}] {len(frame_weights)} frames -> "
          f"keep {len(keep_indices)}, drop {len(drop_indices)} | "
          f"saved to {sample_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Export T2V attention frame ranking from a specified LM layer.")
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--task_csv', type=str, required=True)
    p.add_argument('--video_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--question_id', type=str, default=None,
                   help='Process only this question_id; omit for batch mode')
    p.add_argument('--attn_layer_idx', type=int, default=-1,
                   help='LM decoder layer index for attention (supports negative)')
    p.add_argument('--top_k_ratio', type=float, default=0.5,
                   help='Fraction of frames to KEEP (e.g. 0.5 = keep top 50%%)')
    p.add_argument('--fps', type=float, default=None,
                   help='Override fps (default: auto based on video length)')
    p.add_argument('--min_pixels', type=int, default=MIN_PIXELS)
    p.add_argument('--max_pixels', type=int, default=MAX_PIXELS)
    p.add_argument('--min_frames', type=int, default=MIN_FRAMES)
    p.add_argument('--max_frames', type=int, default=MAX_FRAMES)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(1234)
    print(f"Loading model from {args.ckpt_path} (eager attention) ...")
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        args.ckpt_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto")
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    collector = LayerAttentionCollector(layer_idx=args.attn_layer_idx)
    collector.register(model)

    task_df = pd.read_csv(args.task_csv)
    if args.question_id is not None:
        task_df = task_df[task_df['question_id'] == args.question_id]
        if task_df.empty:
            print(f"ERROR: question_id '{args.question_id}' not found in CSV")
            return

    for row in task_df.itertuples():
        try:
            process_one(row, model, processor, collector, args)
        except Exception as e:
            print(f"ERROR processing {row.question_id}: {e}")

    collector.remove()
    print(f"Done. Results saved under {args.output_dir}")


if __name__ == '__main__':
    main()
