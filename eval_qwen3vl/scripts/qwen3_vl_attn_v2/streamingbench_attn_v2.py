"""StreamingBench evaluation with chunked-scorer attention-based whole-frame dropping (v2)."""

import argparse
import gc
import json
import logging
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_attn_v2 import Qwen3VLForConditionalGenerationDTD

RUN_NAME = "attn_v2_50"
CKPT_PATH = "ckpt/Qwen3-VL-4B-Instruct"
TASK_CSV = "/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR = "/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding"
RESULT_DIR = "eval_qwen3vl/result_streamingbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 10000
MIN_FRAMES = 4
FPS = 1

prompt = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


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


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "CUDA out of memory" in str(exc)
    return False


def run_single_sample(
    model, processor, video_path, question, options, args, max_frames, fps,
    video_start, video_end, stage_totals=None,
):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
             "max_frames": max_frames, "min_frames": args.min_frames,
             "fps": fps, "video_start": video_start, "video_end": video_end},
            {"type": "text",
             "text": prompt.format(question, '\n'.join(options))},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    video_decode_t0 = time.perf_counter()
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
        do_sample_frames=False,
    ).to('cuda')
    if stage_totals is not None:
        stage_totals["video_decode"] += time.perf_counter() - video_decode_t0

    with torch.no_grad():
        model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            drop_method='attention',
            attn_layer_idx=args.attn_layer_idx,
            top_k_ratio=args.top_k_ratio,
            scorer_chunk_size=args.scorer_chunk_size,
        )
        model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
    if stage_totals is not None and model_timing_before and model_timing_after:
        for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
            stage_totals[key] += max(
                0.0,
                model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
            )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    text_decode_t0 = time.perf_counter()
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    if stage_totals is not None:
        stage_totals["text_decode"] += time.perf_counter() - text_decode_t0
    return response


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--top_k_ratio", type=float, default=0.5)
    parser.add_argument("--scorer_chunk_size", type=int, default=512)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir
    run_name = args.run_name
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=run_name, curr_time=curr_time))

    for d in [result_dir, osp.join(result_dir, 'output'), osp.join(result_dir, 'log')]:
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
    logger.info(f"Scorer chunk size: {args.scorer_chunk_size}")
    logger.info(f"Attn implementation: {args.attn_implementation}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Output JSONL: {output_jsonl}")

    torch.manual_seed(1234)
    selected_attn_impl = args.attn_implementation
    logger.info(f"Loading model with {selected_attn_impl} ...")
    try:
        model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=selected_attn_impl,
            device_map="auto",
        )
    except ImportError as e:
        if selected_attn_impl == "flash_attention_2":
            logger.warning("flash_attention_2 unavailable, fallback to sdpa")
            selected_attn_impl = "sdpa"
            model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=selected_attn_impl,
                device_map="auto",
            )
        else:
            raise
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    task_df = pd.read_csv(args.task_csv)
    if args.question_id is not None:
        task_df = task_df[task_df["question_id"] == args.question_id]

    t_start = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    profiled_samples = 0

    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            question_id, task_type, question, time_stamp, answer, options = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options
            frames_required, temporal_clue_type = row.frames_required, row.temporal_clue_type

            video_path = osp.join(args.video_dir, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)

            fps = args.fps
            if 300 < time_stamp_sec <= 600:
                fps = args.fps * 0.5
            elif time_stamp_sec > 600:
                fps = args.fps * 0.2

            options_list = eval(options)
            curr_max_frames = max(args.min_frames, args.max_frames)
            response = None

            while True:
                try:
                    response = run_single_sample(
                        model=model,
                        processor=processor,
                        video_path=video_path,
                        question=question,
                        options=options_list,
                        args=args,
                        max_frames=curr_max_frames,
                        fps=fps,
                        video_start=0,
                        video_end=time_stamp_sec,
                        stage_totals=stage_totals,
                    )
                    break
                except Exception as e:
                    if not is_cuda_oom(e):
                        raise
                    next_max_frames = max(args.min_frames, curr_max_frames // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if next_max_frames >= curr_max_frames:
                        raise
                    logger.warning(
                        f"OOM on {question_id} with max_frames={curr_max_frames}, "
                        f"retry with max_frames={next_max_frames}"
                    )
                    curr_max_frames = next_max_frames

            cnt_total['overall'] += 1
            cnt_total[task_type] += 1
            profiled_samples += 1
            if extract_characters_regex(response) == answer:
                cnt_correct['overall'] += 1
                cnt_correct[task_type] += 1

            output_dict = {
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'time_stamp': time_stamp,
                'answer': answer,
                'options': options_list,
                'frames_required': frames_required,
                'temporal_clue_type': temporal_clue_type,
                'response': response,
            }
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')

        except Exception as e:
            logger.error(f"Error processing {row.question_id}: {e}")

    end_time = time.time()
    t_cost = int(end_time - t_start)

    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception',
                  'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning',
                  'Spatial Understanding', 'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = "
                        f"{100 * cnt_correct[task_type] / cnt_total[task_type]:.2f}%")

    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, "
            f"Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%")

    cost_time_sec = end_time - t_start
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
        pct_total = (100.0 * sec / cost_time_sec) if cost_time_sec > 0 else 0.0
        pct_profiled = (100.0 * sec / measured_stage_total) if measured_stage_total > 0 else 0.0
        logger.info(
            f"- {stage_name_map[key]}: {sec:.2f}s | {pct_total:.2f}% of end-to-end | {pct_profiled:.2f}% inside profiled stages"
        )

    other_time = max(0.0, cost_time_sec - measured_stage_total)
    other_pct = (100.0 * other_time / cost_time_sec) if cost_time_sec > 0 else 0.0
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
        f"Inference cost time: {t_cost // 3600}h {(t_cost % 3600) // 60}m {t_cost % 60}s")


if __name__ == '__main__':
    main()
