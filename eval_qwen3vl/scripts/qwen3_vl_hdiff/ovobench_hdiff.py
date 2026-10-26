import atexit
import argparse
import json
import importlib.util
import logging
import os
import os.path as osp
import re
import sys
import time
import types
from collections import defaultdict
from datetime import datetime

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
CKPT_PATH = "ckpt/Qwen3-VL-8B-Instruct"
TASK_JSON = "./datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR = "./datasets/OVO-Bench/src_videos"
RESULT_DIR = "eval_qwen3vl/result_ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
SIM_SAVE_PATH = "similarity/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1000
FPS = 1
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]
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
}


def _log_frame_compression_summary():
    if _HIST_FRAME_STATS["samples"] == 0:
        return
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


def _build_prompt(task, question, options, anno, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        return f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    if task == "REC":
        activity = anno["activity"]
        question = "How many times did they " + activity + "?"
        return f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    if task == "SSR":
        step = anno["test_info"][index]["step"]
        return f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    if task == "CRR":
        question = anno["question"]
        return f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return ""


def _get_response(prompt, video_path, model, processor, run_config, video_start=None, video_end=None):
    video_content = {
        "type": "video",
        "video": video_path,
        "min_pixels": run_config["min_pixels"],
        "max_pixels": run_config["max_pixels"],
        "min_frames": run_config["min_frames"],
        "max_frames": run_config["max_frames"],
        "fps": run_config["fps"],
    }
    if video_start is not None:
        video_content["video_start"] = video_start
    if video_end is not None:
        video_content["video_end"] = video_end
    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {
                    "type": "text",
                    "text": prompt,
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

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
        do_resize=False,
    )
    inputs = inputs.to(torch.device("cuda"))
    vd_time = time.perf_counter() - video_decode_t0

    model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
    gen_kwargs = {}
    if run_config["temperature"] is None or run_config["temperature"] == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = run_config["temperature"]
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        **gen_kwargs,
        drop_method=run_config["drop_method"],
        drop_threshold=run_config["drop_threshold"],
        drop_absolute=run_config["drop_absolute"],
        dr_save_path=run_config["dr_save_path"],
        dp_save_path=run_config["dp_save_path"],
        sim_save_path=run_config["sim_save_path"],
    )
    model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    td_t0 = time.perf_counter()
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    td_time = time.perf_counter() - td_t0

    timing = {"video_decode": vd_time, "text_decode": td_time}
    if model_timing_before and model_timing_after:
        for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
            timing[key] = max(0.0, model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0))
    return output_text[0], timing


def _score(results):
    def calculate_score_backward_realtime(items):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {}
        for item in items:
            item["score"] = get_score(item["response"], item["ground_truth"])
            scores.setdefault(item["task"], []).append(item["score"])
        return items, scores

    def calculate_score_forward(items):
        def get_score_rec(response, gt):
            if response is None:
                return 0
            response = "".join(re.findall(r"\d+", response))
            return response == str(gt)

        def get_score_ssr_crr(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {task: [] for task in set(result["task"] for result in items)}
        for result in items:
            if result["task"] == "REC":
                cnt_correct = 0
                for test_info in result["test_info"]:
                    cnt_correct += get_score_rec(test_info["response"], test_info["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "SSR":
                cnt_correct = 0
                for test_info in result["test_info"]:
                    if (test_info["response"] == "N" and test_info["type"] == 0) or (
                        test_info["response"] == "Y" and test_info["type"] == 1
                    ):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info["type"] == 0 else "Yes"
                    cnt_correct += get_score_ssr_crr(test_info["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "CRR":
                cnt_correct = 0
                for test_info in result["test_info"]:
                    if (test_info["response"] == "N" and test_info["type"] == 0) or (
                        test_info["response"] == "Y" and test_info["type"] == 1
                    ):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info["type"] == 0 else "Yes"
                    cnt_correct += get_score_ssr_crr(test_info["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return items, scores

    avg_scores = {"backward": [], "realtime": [], "forward": []}
    if len(results["backward"]) > 0:
        _, backward_scores = calculate_score_backward_realtime(results["backward"])
        for k, v in backward_scores.items():
            logger.info("Task: %s, Acc: %.2f", k, 100 * sum(v) / len(v))
            avg_scores["backward"].append(sum(v) / len(v))
        logger.info("Backward Avg.: %.2f\n", 100 * sum(avg_scores["backward"]) / len(avg_scores["backward"]))

    if len(results["realtime"]) > 0:
        _, realtime_scores = calculate_score_backward_realtime(results["realtime"])
        for k, v in realtime_scores.items():
            logger.info("Task: %s, Acc: %.2f", k, 100 * sum(v) / len(v))
            avg_scores["realtime"].append(sum(v) / len(v))
        logger.info("Realtime Avg.: %.2f\n", 100 * sum(avg_scores["realtime"]) / len(avg_scores["realtime"]))

    if len(results["forward"]) > 0:
        _, forward_scores = calculate_score_forward(results["forward"])
        for k, v in forward_scores.items():
            logger.info("Task: %s, Acc: %.2f", k, 100 * sum(v) / len(v))
            avg_scores["forward"].append(sum(v) / len(v))
        logger.info("Forward Avg.: %.2f\n", 100 * sum(avg_scores["forward"]) / len(avg_scores["forward"]))

    valid_avgs = []
    if len(avg_scores["backward"]) > 0:
        valid_avgs.append(sum(avg_scores["backward"]) / len(avg_scores["backward"]))
    if len(avg_scores["realtime"]) > 0:
        valid_avgs.append(sum(avg_scores["realtime"]) / len(avg_scores["realtime"]))
    if len(avg_scores["forward"]) > 0:
        valid_avgs.append(sum(avg_scores["forward"]) / len(avg_scores["forward"]))
    if len(valid_avgs) > 0:
        logger.info("Total Avg.: %.2f", 100 * sum(valid_avgs) / len(valid_avgs))
    else:
        logger.info("Total Avg.: N/A (no valid results)")


def _run_ovobench(argv, hdiff_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sim_save_path", type=str, default=None)
    parser.add_argument("--dp_save_path", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    drop_method = args.drop_method
    drop_threshold = args.drop_threshold
    drop_absolute = not args.drop_relative
    ckpt_path = args.ckpt_path
    result_dir = args.result_dir
    task_json = args.task_json
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
    drop_relative_hdiff = bool(getattr(hdiff_args, "drop_relative_hdiff", False))
    run_config = {
        "drop_method": drop_method,
        "drop_threshold": drop_threshold,
        "drop_absolute": drop_absolute,
        "dr_save_path": dr_save_path,
        "dp_save_path": dp_save_path,
        "sim_save_path": sim_save_path,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "fps": args.fps,
        "temperature": args.temperature,
    }

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(osp.join(result_dir, "output"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "drop"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "drop_pos"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "similarity"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "log"), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(FMT)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("Running %s on OvOBench", run_name)
    logger.info("Drop method: %s", drop_method)
    logger.info("Drop threshold: %s", drop_threshold)
    logger.info("Drop absolute" if drop_absolute else "Drop relative")
    logger.info("Checkpoint path: %s", ckpt_path)
    logger.info("Result dir: %s", result_dir)
    logger.info("Task json: %s", task_json)
    logger.info("Video dir: %s", video_dir)
    logger.info("Output jsonl: %s", output_jsonl)
    logger.info("Drop ratio info save path: %s", dr_save_path)
    logger.info("Drop positions save path: %s", dp_save_path)
    logger.info("Similarity info save path: %s", sim_save_path)
    logger.info("Min pixels: %s", run_config["min_pixels"])
    logger.info("Max pixels: %s", run_config["max_pixels"])
    logger.info("Max frames: %s", run_config["max_frames"])
    logger.info("Min frames: %s", run_config["min_frames"])
    logger.info("FPS: %s", run_config["fps"])
    logger.info("Temperature (T): %s", run_config["temperature"])
    logger.info("Drop relative hdiff: %s", drop_relative_hdiff)

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
        min_pixels=run_config["min_pixels"],
        max_pixels=run_config["max_pixels"],
    )
    logger.info("Load model and processor from %s", ckpt_path)

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    with open(task_json, "r") as f:
        task_list = json.load(f)

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

    for item in tqdm(task_list):
        try:
            if item["task"] in BACKWARD_TASKS or item["task"] in REALTIME_TASKS:
                item_id, video, task, question, options, realtime, gt = (
                    item["id"],
                    item["video"],
                    item["task"],
                    item["question"],
                    item["options"],
                    item["realtime"],
                    item["gt"],
                )
                prompt = _build_prompt(task=task, question=question, options=options, anno=None, index=None)
                video_path = osp.join(video_dir, video)
                response, timing = _get_response(
                    prompt=prompt,
                    video_path=video_path,
                    model=model,
                    processor=processor,
                    run_config=run_config,
                    video_start=0,
                    video_end=int(realtime),
                )
                output_dict = {
                    "id": item_id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(65 + gt),
                }
            elif item["task"] in FORWARD_TASKS:
                video, task, test_info = item["video"], item["task"], item["test_info"]
                timing = None
                for i in range(len(test_info)):
                    prompt = _build_prompt(task=task, question=None, options=None, anno=item, index=i)
                    realtime = test_info[i]["realtime"]
                    video_path = osp.join(video_dir, video)
                    resp, timing = _get_response(
                        prompt=prompt,
                        video_path=video_path,
                        model=model,
                        processor=processor,
                        run_config=run_config,
                        video_start=0,
                        video_end=int(realtime),
                    )
                    item["test_info"][i]["response"] = resp
                output_dict = item
            else:
                continue

            if timing:
                for k, v in timing.items():
                    if k in stage_totals:
                        stage_totals[k] += v
            profiled_samples += 1

            with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w") as f:
                f.write(json.dumps(output_dict) + "\n")
        except Exception as e:
            logger.error("Error in processing %s: %s", item, e)

    end_time = time.time()
    cost_time_sec = end_time - start_time
    cost_time = int(cost_time_sec)

    try:
        results = defaultdict(list)
        with open(output_jsonl, "r") as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            if item["task"] in BACKWARD_TASKS:
                results["backward"].append(item)
            elif item["task"] in REALTIME_TASKS:
                results["realtime"].append(item)
            elif item["task"] in FORWARD_TASKS:
                results["forward"].append(item)
        _score(results)

        drop_enabled = drop_method is not None and str(drop_method).lower() != "none"
        if drop_enabled and dr_save_path is not None and osp.exists(dr_save_path):
            drop_list, total_list, ratio_list = [], [], []
            with open(dr_save_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                drop_ratio_info = json.loads(line)
                drop_list.append(drop_ratio_info["drop"])
                total_list.append(drop_ratio_info["total"])
                ratio_list.append(drop_ratio_info["ratio"])
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info("Total drop ratio (weighted drop ratio): %.1f%%", 100 * total_dr)
            logger.info("Average drop ratio (unweighted drop ratio): %.1f%%", 100 * avg_dr)
        elif drop_enabled and dr_save_path is not None:
            logger.warning("Drop ratio file not found, skip statistics: %s", dr_save_path)

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
    finally:
        logger.info("Inference cost time: %sh %sm %ss", cost_time // 3600, (cost_time % 3600) // 60, cost_time % 60)


if __name__ == "__main__":
    hdiff_args, remaining_argv = _parse_hdiff_args(sys.argv[1:])
    _install_hist_logging_hook(hdiff_args)
    _patch_runtime(hdiff_args)
    _run_ovobench(remaining_argv, hdiff_args)
