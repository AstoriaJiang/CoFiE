"""
StreamingBench evaluation with attention-based whole-frame dropping.

Single-pass pipeline (no re-prefill):
  model.generate() with drop_method='attention' triggers:
    Phase 1  – layers 0..attn_layer_idx on full sequence, capture T2V attention
    Phase 2  – drop entire low-weight frames at the token level
    Phase 3  – layers attn_layer_idx+1..end on pruned sequence, then decode

Requires ``attn_implementation="eager"`` so that the target layer can produce
attention weights via hooks.
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import ffmpeg
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_attn import Qwen3VLForConditionalGenerationDTD

# ---------------------------------------------------------------------------
RUN_NAME = "attn_frame_drop_50"
CKPT_PATH = "wyccccc/TimeChatOnline-7B"
TASK_CSV = "/home/gaohuan03/yaolinli/datasets/StreamingBench/annos/Real_Time_Visual_Understanding.csv"
VIDEO_DIR = "/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding"
RESULT_DIR = "eval_qwen3vl/result_streamingbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 10000

PROMPT = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a "
    "multiple-choice question related to the video. Your task is to "
    "carefully analyze the video and provide the best answer to question, "
    "choosing from the four options provided. Respond with only the letter "
    "(A, B, C, or D) of the correct option.\n\n"
    "Question: {}\n\nOptions:\n{}\n\nThe best option is:"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def time_to_seconds(time_str):
    if len(time_str) == 5:
        t = datetime.strptime(time_str, '%M:%S')
    else:
        t = datetime.strptime(time_str, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


def extract_characters_regex(s):
    s = s.strip()
    for prefix in [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    m = re.search(r"[ABCD]", s)
    return m[0] if m else ""


def split_video(video_file, start_time, end_time):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        return output_file
    try:
        (
            ffmpeg.input(video_file, ss=int(start_time))
            .output(output_file, t=int(end_time) - int(start_time),
                    vcodec='libx264', acodec='aac')
            .global_args('-nostdin')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="StreamingBench with single-pass attention frame dropping.")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--task_csv", type=str, default=TASK_CSV)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--attn_layer_idx", type=int, default=-1,
                        help="LM layer for attention extraction (supports negative)")
    parser.add_argument("--top_k_ratio", type=float, default=0.5,
                        help="Fraction of frames to KEEP (0.5 = keep 50%%)")
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir
    run_name = args.run_name
    log_path = osp.join(result_dir, LOG_PATH.format(
        run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(
        run_name=run_name, curr_time=curr_time))

    for d in [result_dir, osp.join(result_dir, 'output'),
              osp.join(result_dir, 'log')]:
        os.makedirs(d, exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Run: {run_name}")
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Attention layer: {args.attn_layer_idx}")
    logger.info(f"Top-k ratio (keep): {args.top_k_ratio}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Output JSONL: {output_jsonl}")

    # ---- Load model (eager attention required for attention capture) ----
    torch.manual_seed(1234)
    logger.info("Loading model with eager attention ...")
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        args.ckpt_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto")
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    # ---- Load data ----
    task_df = pd.read_csv(args.task_csv)
    if args.question_id is not None:
        task_df = task_df[task_df["question_id"] == args.question_id]

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    profiled_samples = 0

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    # ---- Inference loop ----
    t_start = time.time()
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            question_id = row.question_id
            video_path = osp.join(
                args.video_dir,
                f"sample_{question_id.split('_')[-2]}", "video.mp4")
            ts = time_to_seconds(row.time_stamp)
            video_path = split_video(video_path, 0, ts)

            fps = 1
            if 300 < ts <= 600:
                fps = 0.5
            elif ts > 600:
                fps = 0.2

            options_list = eval(row.options) if isinstance(
                row.options, str) else row.options
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "min_pixels": args.min_pixels,
                     "max_pixels": args.max_pixels,
                     "max_frames": args.max_frames,
                     "min_frames": args.min_frames,
                     "fps": fps},
                    {"type": "text",
                     "text": PROMPT.format(
                         row.question, '\n'.join(options_list))},
                ],
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            vd_t0 = time.perf_counter()
            image_inputs, video_inputs = process_vision_info(messages)
            stage_totals["video_decode"] += time.perf_counter() - vd_t0

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt", do_sample_frames=False,
            ).to('cuda')

            model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            # Single-pass generate: attention capture + frame drop + decode
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    drop_method='attention',
                    attn_layer_idx=args.attn_layer_idx,
                    top_k_ratio=args.top_k_ratio,
                )
            model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            if model_timing_before and model_timing_after:
                for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
                    stage_totals[key] += max(
                        0.0,
                        model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
                    )

            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            td_t0 = time.perf_counter()
            response = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            stage_totals["text_decode"] += time.perf_counter() - td_t0

            output_dict = {
                'question_id': question_id,
                'task_type': row.task_type,
                'question': row.question,
                'time_stamp': row.time_stamp,
                'answer': row.answer,
                'options': options_list,
                'frames_required': row.frames_required,
                'temporal_clue_type': row.temporal_clue_type,
                'response': response,
            }
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
            profiled_samples += 1

        except Exception as e:
            logger.error(f"Error processing {row.question_id}: {e}")

    # ---- Print results ----
    t_cost = int(time.time() - t_start)
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    if osp.exists(output_jsonl):
        with open(output_jsonl) as f:
            lines = f.readlines()
    else:
        lines = []
    for line in lines:
        item = json.loads(line)
        cnt_total['overall'] += 1
        cnt_total[item['task_type']] += 1
        if extract_characters_regex(item['response']) == item['answer']:
            cnt_correct['overall'] += 1
            cnt_correct[item['task_type']] += 1

    task_types = [
        'Object Perception', 'Causal Reasoning', 'Clips Summarize',
        'Attribute Perception', 'Event Understanding',
        'Text-Rich Understanding', 'Prospective Reasoning',
        'Spatial Understanding', 'Action Perception', 'Counting',
    ]
    for tt in task_types:
        if cnt_total[tt] == 0:
            logger.info(f"- {tt}: no samples")
        else:
            logger.info(
                f"- {tt}: {cnt_correct[tt]}/{cnt_total[tt]} = "
                f"{100 * cnt_correct[tt] / cnt_total[tt]:.2f}%")
    if cnt_total['overall']:
        logger.info(
            f"Total: {cnt_total['overall']}, "
            f"Correct: {cnt_correct['overall']}, "
            f"Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%")
    else:
        logger.info("No samples processed")

    stage_keys = ["video_decode", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode"]
    stage_name_map = {
        "video_decode": "Frame Extraction",
        "visual_encode": "Vision Encoder",
        "token_drop": "Token Drop",
        "llm_prefill": "LLM Prefill",
        "llm_decode": "LLM Decode",
            "text_decode": "Text Detokenize (processor.batch_decode)",
    }
    measured_stage_total = sum(stage_totals.values())
    logger.info("Stage time breakdown:")
    for key in stage_keys:
        sec = stage_totals[key]
        pct_total = (100.0 * sec / t_cost) if t_cost > 0 else 0.0
        pct_profiled = (100.0 * sec / measured_stage_total) if measured_stage_total > 0 else 0.0
        logger.info(
            f"- {stage_name_map[key]}: {sec:.2f}s | {pct_total:.2f}% of end-to-end | {pct_profiled:.2f}% inside profiled stages"
        )
    other_time = max(0.0, t_cost - measured_stage_total)
    other_pct = (100.0 * other_time / t_cost) if t_cost > 0 else 0.0
    logger.info(f"- Other/uncaptured: {other_time:.2f}s | {other_pct:.2f}% of end-to-end")
    if measured_stage_total > 0:
        bottleneck_key = max(stage_totals, key=stage_totals.get)
        logger.info(
            f"Profiled-stage bottleneck: {stage_name_map[bottleneck_key]} ({100.0 * stage_totals[bottleneck_key] / measured_stage_total:.2f}% of profiled stages)"
        )
    if profiled_samples > 0:
        logger.info(f"Profiled samples: {profiled_samples}")
        for key in stage_keys:
            logger.info(f"- Avg {stage_name_map[key]} per sample: {stage_totals[key] / profiled_samples:.4f}s")

    logger.info(
        f"Time: {t_cost // 3600}h {(t_cost % 3600) // 60}m {t_cost % 60}s")


if __name__ == '__main__':
    main()
