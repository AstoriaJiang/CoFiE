import atexit
import argparse
import importlib.util
import json
import logging
import os
import os.path as osp
import re
import sys
import time
import types
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s | %(message)s")


_repo = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
sys.path.append(osp.join(_repo, "eval_qwen3vl"))
from qwen3_vl import Qwen3VLForConditionalGenerationDTD


RUN_NAME = "feature_0d5"
DROP_METHOD = "feature"
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"
TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval_qwen3vl/result_videomme"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
SIM_SAVE_PATH = "similarity/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 900
MIN_FRAMES = 4
FPS = 1
PROMPT = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""
FMT = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def _load_hdiff_video_processor():
    package_name = "transformers.models.qwen3_vl_hdiff"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [osp.join(_repo, "eval_qwen3vl", "qwen3_vl_hdiff")]
        sys.modules[package_name] = pkg
    module_name = f"{package_name}.video_processing_qwen3_vl"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        file_path = osp.join(_repo, "eval_qwen3vl", "qwen3_vl_hdiff", "video_processing_qwen3_vl.py")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.Qwen3VLVideoProcessor


def _parse_hdiff_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hist_enabled", action="store_true")
    parser.add_argument("--hist_bins", type=int, default=32)
    parser.add_argument("--hist_threshold", type=float, default=0.15)
    parser.add_argument("--drop_relative_hdiff", action="store_true")
    parser.add_argument("--hist_min_keep_frames", type=int, default=4)
    parser.add_argument("--hist_max_interval", type=int, default=30)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


_HIST_FRAME_STATS = {
    "samples": 0,
    "raw_frames": 0,
    "kept_frames": 0,
    "loggers": [],
    "atexit_registered": False,
    "summary_logged": False,
}


def _log_frame_compression_summary():
    if _HIST_FRAME_STATS["summary_logged"] or _HIST_FRAME_STATS["samples"] == 0:
        return
    _HIST_FRAME_STATS["summary_logged"] = True
    raw_frames = _HIST_FRAME_STATS["raw_frames"]
    kept_frames = _HIST_FRAME_STATS["kept_frames"]
    compression_ratio = 1 - kept_frames / raw_frames if raw_frames > 0 else 0.0
    target_loggers = _HIST_FRAME_STATS["loggers"] or [logger]
    logged = set()
    for logger_obj in target_loggers:
        if id(logger_obj) in logged:
            continue
        logged.add(id(logger_obj))
        logger_obj.info("Histogram frame selection summary:")
        logger_obj.info("- Samples: %s", _HIST_FRAME_STATS["samples"])
        logger_obj.info("- Raw frames: %s", raw_frames)
        logger_obj.info("- Kept frames: %s", kept_frames)
        logger_obj.info("- Video frame compression ratio: %.2f%%", 100 * compression_ratio)


def _install_frame_selector_stats_hook(video_processor):
    frame_selector = getattr(video_processor, "frame_selector", None)
    if frame_selector is None or hasattr(frame_selector, "_hdiff_orig_select_frames"):
        return

    frame_selector._hdiff_orig_select_frames = frame_selector.select_frames

    def select_frames(frames, _orig=frame_selector._hdiff_orig_select_frames):
        keep_indices = _orig(frames)
        _HIST_FRAME_STATS["samples"] += 1
        _HIST_FRAME_STATS["raw_frames"] += len(frames)
        _HIST_FRAME_STATS["kept_frames"] += len(keep_indices)
        return keep_indices

    frame_selector.select_frames = select_frames

    if not _HIST_FRAME_STATS["atexit_registered"]:
        atexit.register(_log_frame_compression_summary)
        _HIST_FRAME_STATS["atexit_registered"] = True


def _patch_runtime(args):
    qwen3_video_processor_cls = _load_hdiff_video_processor()
    original_from_pretrained = AutoProcessor.from_pretrained

    def from_pretrained(*from_args, **kwargs):
        processor = original_from_pretrained(*from_args, **kwargs)
        if args.hist_enabled:
            ckpt_path = from_args[0] if from_args else kwargs.get("pretrained_model_name_or_path")
            original_size = processor.video_processor.size
            processor.video_processor = qwen3_video_processor_cls.from_pretrained(
                ckpt_path,
                frame_selector_config={
                    "enabled": True,
                    "bins": args.hist_bins,
                    "threshold": args.hist_threshold,
                    "relative": args.drop_relative_hdiff,
                    "min_keep_frames": args.hist_min_keep_frames,
                    "max_interval": args.hist_max_interval,
                },
            )
            processor.video_processor.size = original_size
            _install_frame_selector_stats_hook(processor.video_processor)
        return processor

    AutoProcessor.from_pretrained = from_pretrained


def _install_hist_logging_hook(args):
    if not args.hist_enabled:
        return

    original_add_handler = logging.Logger.addHandler

    def add_handler(logger_obj, handler):
        original_add_handler(logger_obj, handler)
        if isinstance(handler, logging.FileHandler) and not getattr(logger_obj, "_hdiff_hist_logged", False):
            logger_obj._hdiff_hist_logged = True
            _HIST_FRAME_STATS["loggers"].append(logger_obj)
            logger_obj.info("Histogram frame selection config enabled")
            logger_obj.info("hist_bins: %s", args.hist_bins)
            logger_obj.info("hist_threshold: %s", args.hist_threshold)
            logger_obj.info("hist_relative_hdiff: %s", args.drop_relative_hdiff)
            logger_obj.info("hist_min_keep_frames: %s", args.hist_min_keep_frames)
            logger_obj.info("hist_max_interval: %s", args.hist_max_interval)

    logging.Logger.addHandler = add_handler


def _extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def _run_videomme(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sim_save_path", type=str, default=None)
    parser.add_argument("--dp_save_path", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    drop_method = args.drop_method
    drop_threshold = args.drop_threshold
    drop_absolute = not args.drop_relative
    ckpt_path = args.ckpt_path
    result_dir = args.result_dir
    task_parquet = args.task_parquet
    video_dir = args.video_dir
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=run_name, curr_time=curr_time))
    dr_save_path = osp.join(result_dir, DR_SAVE_PATH.format(run_name=run_name, curr_time=curr_time))
    sim_save_path = args.sim_save_path or osp.join(
        result_dir, SIM_SAVE_PATH.format(run_name=run_name, curr_time=curr_time)
    )
    dp_save_path = args.dp_save_path or osp.join(
        result_dir, "drop_pos/{run_name}_{curr_time}.jsonl".format(run_name=run_name, curr_time=curr_time)
    )
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels
    max_frames = args.max_frames
    min_frames = args.min_frames
    fps = args.fps
    temperature = args.temperature

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(osp.join(result_dir, "output"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "drop"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "drop_pos"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "similarity"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "log"), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(FMT)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Running %s on Video-MME", run_name)
    logger.info("Drop method: %s", drop_method)
    logger.info("Drop threshold: %s", drop_threshold)
    logger.info("Drop absolute" if drop_absolute else "Drop relative")
    logger.info("Checkpoint path: %s", ckpt_path)
    logger.info("Result dir: %s", result_dir)
    logger.info("Task parquet: %s", task_parquet)
    logger.info("Video dir: %s", video_dir)
    logger.info("Output jsonl: %s", output_jsonl)
    logger.info("Drop ratio info save path: %s", dr_save_path)
    logger.info("Drop positions save path: %s", dp_save_path)
    logger.info("Similarity info save path: %s", sim_save_path)
    logger.info("Min pixels: %s", min_pixels)
    logger.info("Max pixels: %s", max_pixels)
    logger.info("Max frames: %s", max_frames)
    logger.info("Min frames: %s", min_frames)
    logger.info("FPS: %s", fps)
    logger.info("Temperature (T): %s", temperature)
    logger.info("Generation mode: %s", "greedy" if temperature is None or temperature == 0 else "sampling")

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")
    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    logger.info("Attention implementation: %s", attn_impl)
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
    logger.info("Load model and processor from %s", ckpt_path)

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    task_df = pd.read_parquet(task_parquet)

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    profiled_samples = 0
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            video_id, duration, domain, sub_category, url, videoID, question_id, task_type, question, options, answer = (
                row.video_id,
                row.duration,
                row.domain,
                row.sub_category,
                row.url,
                row.videoID,
                row.question_id,
                row.task_type,
                row.question,
                row.options,
                row.answer,
            )
            video_path = osp.join(video_dir, videoID + ".mp4")
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
                        {
                            "type": "text",
                            "text": PROMPT.format(question, "\n".join(options.tolist())),
                        },
                    ],
                }
            ]

            video_decode_t0 = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs = process_vision_info(
                messages,
                image_patch_size=processor.video_processor.patch_size,
            )
            orig_video_preprocess = processor.video_processor._preprocess

            def _timed_video_preprocess(*preprocess_args, **preprocess_kwargs):
                return orig_video_preprocess(*preprocess_args, **preprocess_kwargs)

            processor.video_processor._preprocess = _timed_video_preprocess
            try:
                processor_num_frames = video_inputs[0].shape[0] if video_inputs else None
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    do_sample_frames=True,
                    num_frames=processor_num_frames,
                    fps=None,
                    do_resize=False,
                )
            finally:
                processor.video_processor._preprocess = orig_video_preprocess
            inputs = inputs.to(torch.device("cuda"))
            stage_totals["video_decode"] += time.perf_counter() - video_decode_t0
            
            # -------------- #

            # use original video preprocess to process video
            # image_inputs = None
            # video_inputs = [video_path]
            # video_decode_t0 = time.perf_counter()
            # orig_video_preprocess = processor.video_processor._preprocess

            # def _timed_video_preprocess(*preprocess_args, **preprocess_kwargs):
            #     return orig_video_preprocess(*preprocess_args, **preprocess_kwargs)

            # processor.video_processor._preprocess = _timed_video_preprocess
            # try:
            #     inputs = processor(
            #         text=[text],
            #         images=image_inputs,
            #         videos=video_inputs,
            #         padding=True,
            #         return_tensors="pt",
            #         do_sample_frames=True,
            #         do_resize=True,
            #         fps=fps,
            #         min_frames=min_frames,
            #         max_frames=max_frames,
            #         min_pixels=min_pixels,
            #         max_pixels=max_pixels,
            #     )
            # finally:
            #     processor.video_processor._preprocess = orig_video_preprocess
            # inputs = inputs.to(torch.device("cuda"))
            # stage_totals["video_decode"] += time.perf_counter() - video_decode_t0

            model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            _gen_kwargs = {}
            if temperature is None or temperature == 0:
                _gen_kwargs["do_sample"] = False
            else:
                _gen_kwargs["do_sample"] = True
                _gen_kwargs["temperature"] = temperature
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                **_gen_kwargs,
                drop_method=drop_method,
                drop_threshold=drop_threshold,
                drop_absolute=drop_absolute,
                dr_save_path=dr_save_path,
                dp_save_path=dp_save_path,
                sim_save_path=sim_save_path,
            )
            model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            if model_timing_before and model_timing_after:
                for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
                    stage_totals[key] += max(
                        0.0,
                        model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
                    )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            text_decode_t0 = time.perf_counter()
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            stage_totals["text_decode"] += time.perf_counter() - text_decode_t0

            response = output_text[0]
            profiled_samples += 1
            cnt_total["overall"] += 1
            cnt_total[duration] += 1
            if _extract_characters_regex(response) == answer:
                cnt_correct["overall"] += 1
                cnt_correct[duration] += 1

            output_dict = {
                "video_id": video_id,
                "duration": duration,
                "domain": domain,
                "sub_category": sub_category,
                "url": url,
                "videoID": videoID,
                "question_id": question_id,
                "task_type": task_type,
                "question": question,
                "options": options.tolist(),
                "answer": answer,
                "response": response,
            }
            with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w") as f:
                f.write(json.dumps(output_dict) + "\n")
        except Exception as e:
            logger.error("Error in processing %s: %s", row, e)

    end_time = time.time()
    cost_time = int(end_time - start_time)

    if cnt_total["overall"] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            "Total: %s, Correct: %s, Accuracy: %.1f%%",
            cnt_total["overall"],
            cnt_correct["overall"],
            100 * cnt_correct["overall"] / cnt_total["overall"],
        )
    for duration in task_df.duration.unique():
        if cnt_total[duration] == 0:
            continue
        logger.info(
            "Duration %s: %s/%s = %.1f%%",
            duration,
            cnt_correct[duration],
            cnt_total[duration],
            100 * cnt_correct[duration] / cnt_total[duration],
        )

    if (
        drop_method is not None
        and str(drop_method).lower() != "none"
        and dr_save_path is not None
        and osp.exists(dr_save_path)
    ):
        drop_list, total_list, ratio_list = [], [], []
        with open(dr_save_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info["drop"])
            total_list.append(drop_ratio_info["total"])
            ratio_list.append(drop_ratio_info["ratio"])
        if drop_list and total_list and sum(total_list) > 0:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info("Total drop ratio (weighted drop ratio): %.1f%%", 100 * total_dr)
            logger.info("Average drop ratio (unweighted drop ratio): %.1f%%", 100 * avg_dr)

    cost_time_sec = end_time - start_time
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
            "- %s: %.2fs | %.2f%% of end-to-end | %.2f%% inside profiled stages",
            stage_name_map[key],
            sec,
            pct_total,
            pct_profiled,
        )

    other_time = max(0.0, cost_time_sec - measured_stage_total)
    other_pct = (100.0 * other_time / cost_time_sec) if cost_time_sec > 0 else 0.0
    logger.info("- Other/uncaptured: %.2fs | %.2f%% of end-to-end", other_time, other_pct)

    if measured_stage_total > 0:
        bottleneck_key = max(stage_totals, key=stage_totals.get)
        logger.info(
            "Profiled-stage bottleneck: %s (%.2f%% of profiled stages)",
            stage_name_map[bottleneck_key],
            100.0 * stage_totals[bottleneck_key] / measured_stage_total,
        )

    if profiled_samples > 0:
        logger.info("Profiled samples: %s", profiled_samples)
        for key in stage_keys:
            logger.info("- Avg %s per sample: %.4fs", stage_name_map[key], stage_totals[key] / profiled_samples)

    logger.info("Inference cost time: %sh %sm %ss", cost_time // 3600, (cost_time % 3600) // 60, cost_time % 60)
    _log_frame_compression_summary()


if __name__ == "__main__":
    hdiff_args, remaining_argv = _parse_hdiff_args(sys.argv[1:])
    _install_hist_logging_hook(hdiff_args)
    _patch_runtime(hdiff_args)
    _run_videomme(remaining_argv)
