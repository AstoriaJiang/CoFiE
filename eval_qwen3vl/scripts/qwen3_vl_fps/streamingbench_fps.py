from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
import importlib.util
from collections import defaultdict
import argparse
import sys
_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.append(osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_fps import Qwen3VLForConditionalGenerationFPS, Qwen3VLVideoProcessor

RUN_NAME = "fps_baseline"
CKPT_PATH = "ckpt/Qwen3-VL-4B-Instruct"
TASK_CSV = "/root/autodl-tmp/CoFiE/datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR = "/root/autodl-tmp/CoFiE/datasets/StreamingBench/Real-Time_Visual_Understanding"
RESULT_DIR = "eval_qwen3vl/result_streamingbench_fps"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 10000

TAU_ADJ = 5.0
TAU_REF = 10.0
CONSECUTIVE_THRESHOLD = 3
MAX_INTERVAL = 30
FPS_ENABLED = False

prompt = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

def attach_frame_selector_hooks(video_processor):
    frame_selector = video_processor.frame_selector
    if not hasattr(frame_selector, "_orig_select_frames"):
        frame_selector._orig_select_frames = frame_selector.select_frames

        def _timed_select_frames(frames, _orig=frame_selector._orig_select_frames):
            t0 = time.perf_counter()
            keep_indices, stats = _orig(frames)
            elapsed = time.perf_counter() - t0
            records = getattr(video_processor, "_frame_selection_records", None)
            if records is None:
                records = []
                video_processor._frame_selection_records = records
            records.append(
                {
                    "raw_frames": int(stats.total_frames),
                    "kept_frames": int(stats.kept_frames),
                    "compression_ratio": float(stats.compression_ratio),
                    "elapsed_sec": float(elapsed),
                }
            )
            return keep_indices, stats

        frame_selector.select_frames = _timed_select_frames


def configure_frame_selector(video_processor, tau_adj, tau_ref, consecutive_threshold, max_interval, enabled):
    cfg = video_processor.frame_selector.config
    cfg.tau_adj = tau_adj
    cfg.tau_ref = tau_ref
    cfg.consecutive_threshold = consecutive_threshold
    cfg.max_interval = max_interval
    cfg.enabled = bool(enabled)


def pop_frame_selector_summary(video_processor):
    records = getattr(video_processor, "_frame_selection_records", [])
    video_processor._frame_selection_records = []
    if not records:
        return None

    raw_frames = sum(r["raw_frames"] for r in records)
    kept_frames = sum(r["kept_frames"] for r in records)
    elapsed_sec = sum(r["elapsed_sec"] for r in records)
    compression_ratio = (kept_frames / raw_frames) if raw_frames > 0 else 1.0
    return {
        "num_videos": len(records),
        "raw_frames": raw_frames,
        "kept_frames": kept_frames,
        "compression_ratio": compression_ratio,
        "elapsed_sec": elapsed_sec,
    }

if __name__ == "__main__":
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
    parser.add_argument("--tau_adj", type=float, default=TAU_ADJ)
    parser.add_argument("--tau_ref", type=float, default=TAU_REF)
    parser.add_argument("--consecutive_threshold", type=int, default=CONSECUTIVE_THRESHOLD)
    parser.add_argument("--max_interval", type=int, default=MAX_INTERVAL)
    parser.add_argument("--fps_enabled", action="store_true")
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_CSV = args.task_csv
    VIDEO_DIR = args.video_dir
    QUESTION_ID = args.question_id
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    TAU_ADJ = args.tau_adj
    TAU_REF = args.tau_ref
    CONSECUTIVE_THRESHOLD = args.consecutive_threshold
    MAX_INTERVAL = args.max_interval
    FPS_ENABLED = args.fps_enabled

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"Running {RUN_NAME} on StreamingBench (FPS Frame Selection)")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task csv: {TASK_CSV}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Question id: {QUESTION_ID}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS Frame Selection enabled: {FPS_ENABLED}")

    logger.info("FPS selector params")
    logger.info(f"  enabled={FPS_ENABLED}")
    logger.info(f"  tau_adj={TAU_ADJ}")
    logger.info(f"  tau_ref={TAU_REF}")
    logger.info(f"  consecutive_threshold={CONSECUTIVE_THRESHOLD}")
    logger.info(f"  max_interval={MAX_INTERVAL}")

    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = Qwen3VLForConditionalGenerationFPS.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    if FPS_ENABLED and not hasattr(processor.video_processor, "frame_selector"):
        logger.info("Replacing default video processor with Qwen3VLVideoProcessor (with frame selector support).")
        processor.video_processor = Qwen3VLVideoProcessor.from_pretrained(CKPT_PATH)
        processor.video_processor.size = {
            "shortest_edge": MIN_PIXELS,
            "longest_edge": MAX_PIXELS,
        }

    video_processor = getattr(processor, "video_processor", None)
    if FPS_ENABLED:
        if video_processor is None or not hasattr(video_processor, "frame_selector"):
            raise RuntimeError(
                "fps_enabled=True but processor.video_processor has no frame_selector. "
                "Cannot guarantee pixel-diff frame selection."
            )
        configure_frame_selector(
            video_processor,
            tau_adj=TAU_ADJ,
            tau_ref=TAU_REF,
            consecutive_threshold=CONSECUTIVE_THRESHOLD,
            max_interval=MAX_INTERVAL,
            enabled=True,
        )
        attach_frame_selector_hooks(video_processor)
        logger.info("Frame selector is force-enabled and runtime hook is installed.")

    logger.info(f"Load model and processor from {CKPT_PATH}")

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    task_df = pd.read_csv(TASK_CSV)
    if QUESTION_ID is not None:
        task_df = task_df[task_df["question_id"] == QUESTION_ID]

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    frame_selector_totals = {
        "samples": 0,
        "videos": 0,
        "raw_frames": 0,
        "kept_frames": 0,
        "elapsed_sec": 0.0,
    }
    profiled_samples = 0

    start_time = time.time()
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
            video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)
            fps = 1
            if 300 < time_stamp_sec <= 600:
                fps = 0.5
            elif time_stamp_sec > 600:
                fps = 0.2
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": MIN_PIXELS,
                            "max_pixels": MAX_PIXELS,
                            "max_frames": MAX_FRAMES,
                            "min_frames": MIN_FRAMES,
                            "fps": fps,
                            "video_start": 0,
                            "video_end": time_stamp_sec,
                        },
                        {
                            "type": "text",
                            "text": prompt.format(question, '\n'.join(eval(options)))
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            video_decode_t0 = time.perf_counter()
            image_inputs, video_inputs = process_vision_info(messages)
            stage_totals["video_decode"] += time.perf_counter() - video_decode_t0

            if FPS_ENABLED:
                video_processor._frame_selection_records = []

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if FPS_ENABLED:
                fs_summary = pop_frame_selector_summary(video_processor)
                if fs_summary is None:
                    logger.warning(
                        f"Frame selector enabled but no stats captured for question_id={question_id}."
                    )
                else:
                    frame_selector_totals["samples"] += 1
                    frame_selector_totals["videos"] += fs_summary["num_videos"]
                    frame_selector_totals["raw_frames"] += fs_summary["raw_frames"]
                    frame_selector_totals["kept_frames"] += fs_summary["kept_frames"]
                    frame_selector_totals["elapsed_sec"] += fs_summary["elapsed_sec"]
                    logger.info(
                        "Frame selector | "
                        f"question_id={question_id} | videos={fs_summary['num_videos']} | "
                        f"raw_frames={fs_summary['raw_frames']} | kept_frames={fs_summary['kept_frames']} | "
                        f"compression_ratio={fs_summary['compression_ratio']:.2%} | "
                        f"runtime={fs_summary['elapsed_sec']:.4f}s"
                    )

            inputs = inputs.to(torch.device('cuda'))

            model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )
            model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            if model_timing_before and model_timing_after:
                for key in ("visual_encode", "llm_prefill", "llm_decode"):
                    stage_totals[key] += max(
                        0.0,
                        model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
                    )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            text_decode_t0 = time.perf_counter()
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            stage_totals["text_decode"] += time.perf_counter() - text_decode_t0

            response = output_text[0]

            output_dict = {
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'time_stamp': time_stamp,
                'answer': answer,
                'options': eval(options),
                'frames_required': frames_required,
                'temporal_clue_type': temporal_clue_type,
                'response': response
            }
            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
            profiled_samples += 1
        except Exception as e:
            logger.error(f"Error in processing {row}: {e}")
    end_time = time.time()
    cost_time_sec = end_time - start_time
    cost_time = int(end_time - start_time)

    try:
        cnt_total = defaultdict(int)
        cnt_correct = defaultdict(int)
        with open(OUTPUT_JSONL, 'r') as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            cnt_total['overall'] += 1
            cnt_total[item['task_type']] += 1
            if extract_characters_regex(item['response']) == item['answer']:
                cnt_correct['overall'] += 1
                cnt_correct[item['task_type']] += 1
        task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception', 'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning', 'Spatial Understanding', 'Action Perception', 'Counting']
        for task_type in task_types:
            if cnt_total[task_type] == 0:
                logger.info(f"- {task_type}: No question processed")
            else:
                logger.info(f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100*cnt_correct[task_type]/cnt_total[task_type]:.2f}%")
        if cnt_total['overall'] == 0:
            logger.info("No question processed")
        else:
            logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.2f}%")

        stage_keys = ["video_decode", "visual_encode", "llm_prefill", "llm_decode", "text_decode"]
        stage_name_map = {
            "video_decode": "Frame Extraction",
            "visual_encode": "Vision Encoder",
            "llm_prefill": "LLM Prefill",
            "llm_decode": "LLM Decode",
            "text_decode": "Text Detokenize",
        }
        measured_stage_total = sum(stage_totals.values())
        logger.info("Stage time breakdown:")
        for key in stage_keys:
            sec = stage_totals[key]
            pct_total = (100.0 * sec / cost_time_sec) if cost_time_sec > 0 else 0.0
            pct_profiled = (100.0 * sec / measured_stage_total) if measured_stage_total > 0 else 0.0
            logger.info(f"- {stage_name_map[key]}: {sec:.2f}s | {pct_total:.2f}% of end-to-end | {pct_profiled:.2f}% inside profiled stages")

        other_time = max(0.0, cost_time_sec - measured_stage_total)
        other_pct = (100.0 * other_time / cost_time_sec) if cost_time_sec > 0 else 0.0
        logger.info(f"- Other/uncaptured: {other_time:.2f}s | {other_pct:.2f}% of end-to-end")

        if measured_stage_total > 0:
            bottleneck_key = max(stage_totals, key=stage_totals.get)
            logger.info(f"Profiled-stage bottleneck: {stage_name_map[bottleneck_key]} ({100.0 * stage_totals[bottleneck_key] / measured_stage_total:.2f}% of profiled stages)")

        if profiled_samples > 0:
            logger.info(f"Profiled samples: {profiled_samples}")
            for key in stage_keys:
                logger.info(f"- Avg {stage_name_map[key]} per sample: {stage_totals[key] / profiled_samples:.4f}s")

        if FPS_ENABLED:
            if frame_selector_totals["samples"] > 0:
                total_raw = frame_selector_totals["raw_frames"]
                total_kept = frame_selector_totals["kept_frames"]
                total_runtime = frame_selector_totals["elapsed_sec"]
                total_ratio = (total_kept / total_raw) if total_raw > 0 else 1.0
                avg_runtime = total_runtime / frame_selector_totals["samples"]
                logger.info("Frame selector summary:")
                logger.info(f"- Samples: {frame_selector_totals['samples']}")
                logger.info(f"- Videos processed: {frame_selector_totals['videos']}")
                logger.info(f"- Raw frames: {total_raw}")
                logger.info(f"- Kept frames: {total_kept}")
                logger.info(f"- Compression ratio: {total_ratio:.2%}")
                logger.info(f"- Runtime: {total_runtime:.4f}s (avg {avg_runtime:.4f}s/sample)")
            else:
                logger.warning("Frame selector is enabled but no frame selection stats were captured.")
    finally:
        logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
