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
TASK_PARQUET = "./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "./datasets/Video-MME/data/"
RESULT_DIR = "eval_qwen3vl/result_videomme_fps"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MAX_FRAMES = 900
MIN_FRAMES = 4
FPS = 1

TAU_ADJ = 5.0
TAU_REF = 10.0
CONSECUTIVE_THRESHOLD = 3
MAX_INTERVAL = 30
FPS_ENABLED = False

prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

def time_to_seconds(time_str):
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
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--tau_adj", type=float, default=TAU_ADJ)
    parser.add_argument("--tau_ref", type=float, default=TAU_REF)
    parser.add_argument("--consecutive_threshold", type=int, default=CONSECUTIVE_THRESHOLD)
    parser.add_argument("--max_interval", type=int, default=MAX_INTERVAL)
    parser.add_argument("--fps_enabled", action="store_true")
    parser.add_argument("--attn_implementation", type=str, choices=["flash_attention_2", "sdpa"], default=None)
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_PARQUET = args.task_parquet
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MAX_FRAMES = args.max_frames
    MIN_FRAMES = args.min_frames
    FPS = args.fps
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

    logger.info(f"Running {RUN_NAME} on Video-MME (FPS Frame Selection)")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task parquet: {TASK_PARQUET}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS: {FPS}")
    logger.info(f"FPS Frame Selection enabled: {FPS_ENABLED}")

    logger.info("FPS selector params")
    logger.info(f"  enabled={FPS_ENABLED}")
    logger.info(f"  tau_adj={TAU_ADJ}")
    logger.info(f"  tau_ref={TAU_REF}")
    logger.info(f"  consecutive_threshold={CONSECUTIVE_THRESHOLD}")
    logger.info(f"  max_interval={MAX_INTERVAL}")

    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    if args.attn_implementation is not None:
        attn_impl = args.attn_implementation
    else:
        attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    logger.info(f"Attention implementation: {attn_impl}")
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
        original_size = processor.video_processor.size
        logger.info("Replacing default video processor with Qwen3VLVideoProcessor (with frame selector support).")
        processor.video_processor = Qwen3VLVideoProcessor.from_pretrained(CKPT_PATH)
        processor.video_processor.size = original_size
        logger.info(f"Preserved original video_processor.size: {original_size}")

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

    task_df = pd.read_parquet(TASK_PARQUET)

    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    frame_selector_totals = {
        "samples": 0,
        "videos": 0,
        "raw_frames": 0,
        "kept_frames": 0,
        "elapsed_sec": 0.0,
    }
    frame_selector_by_duration = defaultdict(lambda: {
        "samples": 0,
        "videos": 0,
        "raw_frames": 0,
        "kept_frames": 0,
        "elapsed_sec": 0.0,
    })
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            video_id, duration, domain, sub_category, url, videoID, question_id, task_type, question, options, answer = \
                row.video_id, row.duration, row.domain, row.sub_category, row.url, row.videoID, row.question_id, row.task_type, row.question, row.options, row.answer
            video_path = osp.join(VIDEO_DIR, videoID + '.mp4')
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
                            "fps": FPS
                        },
                        {
                            "type": "text",
                            "text": prompt.format(question, '\n'.join(options.tolist()))
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

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
                    frame_selector_by_duration[duration]["samples"] += 1
                    frame_selector_by_duration[duration]["videos"] += fs_summary["num_videos"]
                    frame_selector_by_duration[duration]["raw_frames"] += fs_summary["raw_frames"]
                    frame_selector_by_duration[duration]["kept_frames"] += fs_summary["kept_frames"]
                    frame_selector_by_duration[duration]["elapsed_sec"] += fs_summary["elapsed_sec"]
                    logger.info(
                        "Frame selector | "
                        f"question_id={question_id} | videos={fs_summary['num_videos']} | "
                        f"raw_frames={fs_summary['raw_frames']} | kept_frames={fs_summary['kept_frames']} | "
                        f"compression_ratio={fs_summary['compression_ratio']:.2%} | "
                        f"runtime={fs_summary['elapsed_sec']:.4f}s"
                    )

            inputs = inputs.to(torch.device('cuda'))
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            cnt_total['overall'] += 1
            cnt_total[duration] += 1
            if extract_characters_regex(response) == answer:
                cnt_correct['overall'] += 1
                cnt_correct[duration] += 1

            output_dict = {
                'video_id': video_id,
                'duration': duration,
                'domain': domain,
                'sub_category': sub_category,
                'url': url,
                'videoID': videoID,
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'options': options.tolist(),
                'answer': answer,
                'response': response
            }
            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing {row}: {e}")
    end_time = time.time()
    cost_time = int(end_time - start_time)

    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.1f}%")
    for duration in task_df.duration.unique():
        if cnt_total[duration] == 0:
            continue
        logger.info(f"Duration {duration}: {cnt_correct[duration]}/{cnt_total[duration]} = {100*cnt_correct[duration]/cnt_total[duration]:.1f}%")

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
            logger.info(f"- Compression ratio: {(1 - total_ratio):.2%}")
            for duration in sorted(frame_selector_by_duration.keys()):
                dur_stats = frame_selector_by_duration[duration]
                dur_raw = dur_stats["raw_frames"]
                dur_kept = dur_stats["kept_frames"]
                dur_ratio = (dur_kept / dur_raw) if dur_raw > 0 else 1.0
                logger.info(f"- Compression ratio ({duration}): {(1 - dur_ratio):.2%}")
            logger.info(f"- Runtime: {total_runtime:.4f}s (avg {avg_runtime:.4f}s/sample)")
        else:
            logger.warning("Frame selector is enabled but no frame selection stats were captured.")

    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
