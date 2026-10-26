import atexit
import argparse
import json
import importlib.util
import logging
import math
import os
import os.path as osp
import re
import sys
import time
import types
from collections.abc import Mapping
from collections import defaultdict
from datetime import datetime

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset, get_worker_info
from transformers import AutoProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s | %(message)s")


_repo = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
sys.path.append(osp.join(_repo, "eval_qwen3vl"))
from qwen3_vl_hdiff_parellel import Qwen3VLForConditionalGenerationDTD


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


def resolve_processor_path(ckpt_path):
    adapter_config_path = osp.join(ckpt_path, "adapter_config.json")
    if osp.isfile(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        return adapter_config.get("base_model_name_or_path") or ckpt_path
    return ckpt_path


def _load_hdiff_video_processor():
    package_name = "transformers.models.qwen3_vl_hdiff"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [osp.join(_repo, "eval_qwen3vl", "qwen3_vl_hdiff_parellel")]
        sys.modules[package_name] = pkg
    module_name = f"{package_name}.video_processing_qwen3_vl"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        file_path = osp.join(_repo, "eval_qwen3vl", "qwen3_vl_hdiff_parellel", "video_processing_qwen3_vl.py")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.Qwen3VLVideoProcessor


def _load_clip_frame_selector():
    """Load the optional CLIP selector without changing the default Hdiff path."""
    from qwen3_vl_clip.frame_selector import (
        CLIPPatchFrameSelector,
        CLIPPatchFrameSelectorConfig,
    )

    return CLIPPatchFrameSelector, CLIPPatchFrameSelectorConfig


def _parse_hdiff_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--hist_enabled", action="store_true")
    parser.add_argument("--hist_bins", type=int, default=32)
    parser.add_argument("--hist_threshold", type=float, default=0.15)
    parser.add_argument("--drop_relative_hdiff", action="store_true")
    parser.add_argument("--hist_min_keep_frames", type=int, default=4)
    parser.add_argument("--hist_max_interval", type=int, default=30)
    parser.add_argument("--frame_select_enabled", action="store_true")
    parser.add_argument("--frame_select_method", type=str, default="histogram")
    parser.add_argument("--frame_select_drop_ratio", type=float, default=0.2)
    parser.add_argument("--frame_select_size", type=int, default=32)
    parser.add_argument("--frame_select_min_keep_frames", type=int, default=4)
    parser.add_argument("--clip_enabled", action="store_true")
    parser.add_argument("--clip_model_path", type=str, default="ckpt/clip-vit-base-patch32")
    parser.add_argument("--clip_device", type=str, default="cuda")
    parser.add_argument("--clip_dtype", type=str, default="float16")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--clip_strategy", choices=("relative", "threshold"), default="relative")
    parser.add_argument("--clip_drop_ratio", type=float, default=0.2)
    parser.add_argument("--clip_similarity_threshold", type=float, default=0.95)
    parser.add_argument("--clip_min_keep_frames", type=int, default=4)
    parser.add_argument("--clip_max_interval", type=int, default=30)
    parser.add_argument("--clip_stats_path", type=str, default=None)
    parser.add_argument("--clip_allow_remote", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


_HIST_FRAME_STATS = {
    "samples": 0,
    "raw_frames": 0,
    "kept_frames": 0,
    "selection_time": 0.0,
    "clip_model_load_time": 0.0,
    "clip_preprocess_time": 0.0,
    "clip_forward_time": 0.0,
    "clip_selection_time": 0.0,
    "clip_filter_time": 0.0,
    "selector_kind": None,
    "loggers": [],
    "atexit_registered": False,
    "summary_logged": False,
}

_FRAME_STATS_KEYS = (
    "samples",
    "raw_frames",
    "kept_frames",
    "selection_time",
    "clip_model_load_time",
    "clip_preprocess_time",
    "clip_forward_time",
    "clip_selection_time",
    "clip_filter_time",
)

_CLIP_TIMING_KEYS = {
    "model_load_seconds": "clip_model_load_time",
    "preprocess_seconds": "clip_preprocess_time",
    "forward_seconds": "clip_forward_time",
    "selection_seconds": "clip_selection_time",
    "total_seconds": "clip_filter_time",
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
        logger_obj.info("Frame selection summary:")
        logger_obj.info("- Samples: %s", _HIST_FRAME_STATS["samples"])
        logger_obj.info("- Raw frames: %s", raw_frames)
        logger_obj.info("- Kept frames: %s", kept_frames)
        logger_obj.info("- Video frame compression ratio: %.2f%%", 100 * compression_ratio)
        logger_obj.info("- Frame select time: %.2fs", _HIST_FRAME_STATS["selection_time"])
        if _HIST_FRAME_STATS["selector_kind"] == "CLIP":
            clip_load_time = _HIST_FRAME_STATS["clip_model_load_time"]
            clip_filter_time = _HIST_FRAME_STATS["clip_filter_time"]
            logger_obj.info("CLIP timing summary:")
            logger_obj.info("- CLIP model load time: %.2fs", clip_load_time)
            logger_obj.info("- CLIP frame preprocessing time: %.2fs", _HIST_FRAME_STATS["clip_preprocess_time"])
            logger_obj.info("- CLIP forward time: %.2fs", _HIST_FRAME_STATS["clip_forward_time"])
            logger_obj.info("- CLIP index selection time: %.2fs", _HIST_FRAME_STATS["clip_selection_time"])
            logger_obj.info("- CLIP filtering time (excluding model load): %.2fs", clip_filter_time)
            logger_obj.info("- CLIP end-to-end time (model load + filtering): %.2fs", clip_load_time + clip_filter_time)


def _selector_timing_snapshot(frame_selector):
    get_stats = getattr(frame_selector, "get_stats", None)
    if not callable(get_stats):
        return {}
    return get_stats()


def _install_frame_selector_stats_hook(video_processor, selector_kind):
    frame_selector = getattr(video_processor, "frame_selector", None)
    if frame_selector is None or hasattr(frame_selector, "_hdiff_orig_select_frames"):
        return
    _HIST_FRAME_STATS["selector_kind"] = selector_kind

    frame_selector._hdiff_orig_select_frames = frame_selector.select_frames

    def select_frames(frames, _orig=frame_selector._hdiff_orig_select_frames):
        select_t0 = time.perf_counter()
        timing_before = _selector_timing_snapshot(frame_selector)
        keep_indices = _orig(frames)
        timing_after = _selector_timing_snapshot(frame_selector)
        _HIST_FRAME_STATS["samples"] += 1
        _HIST_FRAME_STATS["raw_frames"] += len(frames)
        _HIST_FRAME_STATS["kept_frames"] += len(keep_indices)
        _HIST_FRAME_STATS["selection_time"] += time.perf_counter() - select_t0
        for selector_key, stats_key in _CLIP_TIMING_KEYS.items():
            _HIST_FRAME_STATS[stats_key] += float(timing_after.get(selector_key, 0.0)) - float(
                timing_before.get(selector_key, 0.0)
            )
        return keep_indices

    frame_selector.select_frames = select_frames

    if get_worker_info() is None and not _HIST_FRAME_STATS["atexit_registered"]:
        atexit.register(_log_frame_compression_summary)
        _HIST_FRAME_STATS["atexit_registered"] = True


def _patch_runtime(args):
    qwen3_video_processor_cls = _load_hdiff_video_processor()
    clip_selector_cls = None
    clip_selector_config_cls = None
    if args.clip_enabled:
        clip_selector_cls, clip_selector_config_cls = _load_clip_frame_selector()
    original_from_pretrained = AutoProcessor.from_pretrained

    def from_pretrained(*from_args, **kwargs):
        processor = original_from_pretrained(*from_args, **kwargs)
        if args.clip_enabled:
            ckpt_path = from_args[0] if from_args else kwargs.get("pretrained_model_name_or_path")
            original_size = processor.video_processor.size
            video_processor = qwen3_video_processor_cls.from_pretrained(ckpt_path)
            video_processor.size = original_size
            video_processor.frame_selector = clip_selector_cls(
                clip_selector_config_cls(
                    enabled=True,
                    model_path=args.clip_model_path,
                    device=args.clip_device,
                    dtype=args.clip_dtype,
                    batch_size=args.clip_batch_size,
                    strategy=args.clip_strategy,
                    drop_ratio=args.clip_drop_ratio,
                    similarity_threshold=args.clip_similarity_threshold,
                    min_keep_frames=args.clip_min_keep_frames,
                    max_interval=args.clip_max_interval,
                    stats_path=args.clip_stats_path,
                    local_files_only=not args.clip_allow_remote,
                )
            )
            processor.video_processor = video_processor
            # The custom video processor applies its own frame selector during
            # preprocessing.  Do not also filter the decoded frames in the
            # OVO dataset path, or a configured 20% CLIP drop becomes 36%.
            processor.video_processor._skip_external_frame_selector = True
            _install_frame_selector_stats_hook(processor.video_processor, selector_kind="CLIP")
        elif args.hist_enabled or args.frame_select_enabled:
            ckpt_path = from_args[0] if from_args else kwargs.get("pretrained_model_name_or_path")
            original_size = processor.video_processor.size
            frame_selector_config = {
                "enabled": True,
                "bins": args.hist_bins,
                "threshold": args.hist_threshold,
                "relative": args.drop_relative_hdiff,
                "min_keep_frames": args.hist_min_keep_frames,
                "max_interval": args.hist_max_interval,
            }
            if args.frame_select_enabled:
                frame_selector_config.update({
                    "method": args.frame_select_method,
                    "drop_ratio": args.frame_select_drop_ratio,
                    "relative": True,
                    "min_keep_frames": args.frame_select_min_keep_frames,
                    "size": args.frame_select_size,
                })
            processor.video_processor = qwen3_video_processor_cls.from_pretrained(
                ckpt_path,
                frame_selector_config=frame_selector_config,
            )
            processor.video_processor.size = original_size
            _install_frame_selector_stats_hook(processor.video_processor, selector_kind="Hdiff")
        return processor

    AutoProcessor.from_pretrained = from_pretrained


def _install_hist_logging_hook(args):
    if not args.hist_enabled and not args.frame_select_enabled and not args.clip_enabled:
        return

    original_add_handler = logging.Logger.addHandler

    def add_handler(logger_obj, handler):
        original_add_handler(logger_obj, handler)
        if isinstance(handler, logging.FileHandler) and not getattr(logger_obj, "_hdiff_hist_logged", False):
            logger_obj._hdiff_hist_logged = True
            _HIST_FRAME_STATS["loggers"].append(logger_obj)
            if args.clip_enabled:
                logger_obj.info("CLIP patch frame selection config enabled")
                logger_obj.info("clip_model_path: %s", args.clip_model_path)
                logger_obj.info("clip_device: %s", args.clip_device)
                logger_obj.info("clip_dtype: %s", args.clip_dtype)
                logger_obj.info("clip_batch_size: %s", args.clip_batch_size)
                logger_obj.info("clip_strategy: %s", args.clip_strategy)
                logger_obj.info("clip_drop_ratio: %s", args.clip_drop_ratio)
                logger_obj.info("clip_similarity_threshold: %s", args.clip_similarity_threshold)
                logger_obj.info("clip_min_keep_frames: %s", args.clip_min_keep_frames)
                logger_obj.info("clip_max_interval: %s", args.clip_max_interval)
                logger_obj.info("clip_stats_path: %s", args.clip_stats_path)
                logger_obj.info("clip_local_files_only: %s", not args.clip_allow_remote)
            else:
                logger_obj.info("Frame selection config enabled")
                logger_obj.info("hist_bins: %s", args.hist_bins)
                logger_obj.info("hist_threshold: %s", args.hist_threshold)
                logger_obj.info("hist_relative_hdiff: %s", args.drop_relative_hdiff)
                logger_obj.info("hist_min_keep_frames: %s", args.hist_min_keep_frames)
                logger_obj.info("hist_max_interval: %s", args.hist_max_interval)
                logger_obj.info("frame_select_enabled: %s", args.frame_select_enabled)
                logger_obj.info("frame_select_method: %s", args.frame_select_method)
                logger_obj.info("frame_select_drop_ratio: %s", args.frame_select_drop_ratio)
                logger_obj.info("frame_select_size: %s", args.frame_select_size)
                logger_obj.info("frame_select_min_keep_frames: %s", args.frame_select_min_keep_frames)

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


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _write_jsonl(path, payload):
    with open(path, "a" if osp.exists(path) else "w") as f:
        f.write(json.dumps(payload, default=_json_default) + "\n")


def _make_requests(task_list, video_dir):
    requests = []
    for item_index, item in enumerate(task_list):
        task = item["task"]
        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            prompt = _build_prompt(
                task=task,
                question=item["question"],
                options=item["options"],
                anno=None,
                index=None,
            )
            requests.append(
                {
                    "prompt": prompt,
                    "video_path": osp.join(video_dir, item["video"]),
                    "video_start": 0,
                    "video_end": math.ceil(float(item["realtime"])),
                    "metadata": {
                        "item_index": item_index,
                        "request_type": "single",
                        "id": item["id"],
                        "task": task,
                        "output_base": {
                            "id": item["id"],
                            "video": item["video"],
                            "task": task,
                            "question": item["question"],
                            "ground_truth": chr(65 + item["gt"]),
                        },
                    },
                }
            )
        elif task in FORWARD_TASKS:
            for sub_index, test_info in enumerate(item["test_info"]):
                prompt = _build_prompt(task=task, question=None, options=None, anno=item, index=sub_index)
                requests.append(
                    {
                        "prompt": prompt,
                        "video_path": osp.join(video_dir, item["video"]),
                        "video_start": 0,
                        "video_end": math.ceil(float(test_info["realtime"])),
                        "metadata": {
                            "item_index": item_index,
                            "request_type": "forward",
                            "sub_index": sub_index,
                            "total_subitems": len(item["test_info"]),
                            "id": item.get("id", item_index),
                            "task": task,
                            "output_base": json.loads(json.dumps(item, default=_json_default)),
                        },
                    }
                )
    return requests


def _build_messages(prompt, video_path, run_config, video_start=None, video_end=None):
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
    return [
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


def _take_video_frames(video, indices):
    if torch.is_tensor(video):
        return video[list(indices)]
    return [video[index] for index in indices]


def _apply_preencoder_frame_selector(video_inputs, video_processor):
    """Apply an enabled Hdiff or CLIP selector to decoded video frames once."""
    if video_inputs is None:
        return None
    if getattr(video_processor, "_skip_external_frame_selector", False):
        return video_inputs
    selector = getattr(video_processor, "frame_selector", None)
    selector_config = getattr(selector, "config", None)
    if selector is None or not getattr(selector_config, "enabled", False):
        return video_inputs
    if torch.is_tensor(video_inputs) and video_inputs.ndim == 4:
        return _take_video_frames(video_inputs, selector.select_frames(list(video_inputs)))
    return [
        _take_video_frames(video, selector.select_frames(list(video)))
        for video in video_inputs
    ]


class OVORequestDataset(Dataset):
    def __init__(self, requests, ckpt_path, run_config):
        self.requests = requests
        self.ckpt_path = ckpt_path
        self.run_config = run_config
        self._processor = None

    def __len__(self):
        return len(self.requests)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.ckpt_path,
                min_pixels=self.run_config["min_pixels"],
                max_pixels=self.run_config["max_pixels"],
            )
        return self._processor

    def __getitem__(self, idx):
        request = self.requests[idx]
        metadata = dict(request["metadata"])
        metadata["idx"] = int(idx)
        metadata["video_path"] = request["video_path"]
        try:
            processor = self._get_processor()
            messages = _build_messages(
                request["prompt"],
                request["video_path"],
                self.run_config,
                request.get("video_start"),
                request.get("video_end"),
            )
            video_decode_t0 = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(
                messages,
                image_patch_size=processor.video_processor.patch_size,
            )
            hist_before = {
                key: _HIST_FRAME_STATS[key]
                for key in _FRAME_STATS_KEYS
            }
            filtered_videos = _apply_preencoder_frame_selector(
                video_inputs,
                processor.video_processor,
            )
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=filtered_videos,
                padding=True,
                return_tensors="pt",
                do_sample_frames=False,
                do_resize=False,
            )
            hist_after = {
                key: _HIST_FRAME_STATS[key]
                for key in _FRAME_STATS_KEYS
            }
            frame_stats = {
                key: hist_after[key] - hist_before[key]
                for key in _FRAME_STATS_KEYS
            }
            frame_stats["from_worker"] = get_worker_info() is not None
            return {
                "inputs": inputs,
                "metadata": metadata,
                "preprocess_time": time.perf_counter() - video_decode_t0,
                "frame_stats": frame_stats,
                "error": None,
            }
        except Exception as e:
            return {
                "inputs": None,
                "metadata": metadata,
                "preprocess_time": 0.0,
                "frame_stats": {
                    **{key: 0.0 for key in _FRAME_STATS_KEYS},
                    "from_worker": get_worker_info() is not None,
                },
                "error": f"{type(e).__name__}: {e}",
            }


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("ovobench_hdiff_parallel.py currently supports batch_size=1 only")
    return batch[0]


def move_to_cuda(obj, device="cuda", non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {key: move_to_cuda(value, device, non_blocking) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(value, device, non_blocking) for value in obj)
    if isinstance(obj, list):
        return [move_to_cuda(value, device, non_blocking) for value in obj]
    return obj


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=False,
    persistent_workers=False,
):
    if batch_size != 1:
        raise ValueError("Only batch_size=1 is supported for variable-length OVO-Bench samples")
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)


def _record_frame_stats(frame_stats):
    if not frame_stats.get("from_worker", False):
        return
    for key in _FRAME_STATS_KEYS:
        if key in ("samples", "raw_frames", "kept_frames"):
            _HIST_FRAME_STATS[key] += int(frame_stats.get(key, 0))
        else:
            _HIST_FRAME_STATS[key] += float(frame_stats.get(key, 0.0))


def run_inference_with_dataloader(
    dataloader_iter,
    total,
    model,
    processor,
    run_config,
    output_jsonl,
    error_jsonl,
    pin_memory,
    exit_on_oom,
    stage_totals,
):
    profiled_samples = 0
    forward_states = {}
    gen_kwargs = {}
    if run_config["temperature"] is None or run_config["temperature"] == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = run_config["temperature"]

    for batch in tqdm(dataloader_iter, total=total):
        metadata = batch["metadata"]
        stage_totals["video_decode"] += float(batch.get("preprocess_time", 0.0))
        stage_totals["frame_select"] += float(batch.get("frame_stats", {}).get("selection_time", 0.0))
        _record_frame_stats(batch.get("frame_stats", {}))

        if batch.get("error"):
            logger.error("Preprocess error idx=%s id=%s: %s", metadata.get("idx"), metadata.get("id"), batch["error"])
            _write_jsonl(error_jsonl, {"stage": "preprocess", "metadata": metadata, "error": batch["error"]})
            continue

        try:
            inputs = move_to_cuda(batch["inputs"], torch.device("cuda"), non_blocking=pin_memory)
            model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    **gen_kwargs,
                    drop_method=run_config["drop_method"],
                    drop_threshold=run_config["drop_threshold"],
                    drop_absolute=run_config["drop_absolute"],
                    attn_layer_idx=run_config["attn_layer_idx"],
                    top_k_ratio=run_config["top_k_ratio"],
                    scorer_chunk_size=run_config["scorer_chunk_size"],
                    dr_save_path=run_config["dr_save_path"],
                    dp_save_path=run_config["dp_save_path"],
                    sim_save_path=run_config["sim_save_path"],
                )
            model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            if model_timing_before and model_timing_after:
                for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
                    stage_totals[key] += max(
                        0.0,
                        model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
                    )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            td_t0 = time.perf_counter()
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            stage_totals["text_decode"] += time.perf_counter() - td_t0
            response = output_text[0]

            if metadata["request_type"] == "single":
                output_dict = dict(metadata["output_base"])
                output_dict["response"] = response
                _write_jsonl(output_jsonl, output_dict)
                profiled_samples += 1
            else:
                item_index = metadata["item_index"]
                if item_index not in forward_states:
                    forward_states[item_index] = {
                        "item": metadata["output_base"],
                        "remaining": metadata["total_subitems"],
                    }
                state = forward_states[item_index]
                state["item"]["test_info"][metadata["sub_index"]]["response"] = response
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    _write_jsonl(output_jsonl, state["item"])
                    profiled_samples += 1
        except RuntimeError as e:
            err_msg = f"{type(e).__name__}: {e}"
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM idx=%s id=%s: %s", metadata.get("idx"), metadata.get("id"), err_msg)
                torch.cuda.empty_cache()
                _write_jsonl(error_jsonl, {"stage": "generate", "metadata": metadata, "error": err_msg})
                if exit_on_oom:
                    raise
                continue
            logger.error("Generate error idx=%s id=%s: %s", metadata.get("idx"), metadata.get("id"), err_msg)
            _write_jsonl(error_jsonl, {"stage": "generate", "metadata": metadata, "error": err_msg})
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            logger.error("Generate error idx=%s id=%s: %s", metadata.get("idx"), metadata.get("id"), err_msg)
            _write_jsonl(error_jsonl, {"stage": "generate", "metadata": metadata, "error": err_msg})
    return profiled_samples


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
    filtered_videos = _apply_preencoder_frame_selector(
        video_inputs,
        processor.video_processor,
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=filtered_videos,
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
        attn_layer_idx=run_config["attn_layer_idx"],
        top_k_ratio=run_config["top_k_ratio"],
        scorer_chunk_size=run_config["scorer_chunk_size"],
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
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--top_k_ratio", type=float, default=0.5)
    parser.add_argument("--scorer_chunk_size", type=int, default=512)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--log_dir", type=str, default=None)
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--exit_on_oom", action="store_true")
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
    log_dir = args.log_dir or osp.join(result_dir, "log")
    log_path = osp.join(log_dir, "{run_name}_{curr_time}.log".format(run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=run_name, curr_time=curr_time))
    error_jsonl = osp.join(result_dir, "error/{run_name}_{curr_time}.jsonl".format(run_name=run_name, curr_time=curr_time))
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
        "attn_layer_idx": args.attn_layer_idx,
        "top_k_ratio": args.top_k_ratio,
        "scorer_chunk_size": args.scorer_chunk_size,
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
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(osp.join(result_dir, "error"), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(FMT)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("Running %s on OvOBench", run_name)
    logger.info("Drop method: %s", drop_method)
    logger.info("Drop threshold: %s", drop_threshold)
    logger.info("Drop absolute" if drop_absolute else "Drop relative")
    if str(drop_method).lower() == "attention":
        logger.info("Attention layer: %s", args.attn_layer_idx)
        logger.info("Top-k ratio (keep): %s", args.top_k_ratio)
        logger.info("Scorer chunk size: %s", args.scorer_chunk_size)
    logger.info("Checkpoint path: %s", ckpt_path)
    logger.info("Result dir: %s", result_dir)
    logger.info("Log dir: %s", log_dir)
    logger.info("Task json: %s", task_json)
    logger.info("Video dir: %s", video_dir)
    logger.info("Output jsonl: %s", output_jsonl)
    logger.info("Error jsonl: %s", error_jsonl)
    logger.info("Drop ratio info save path: %s", dr_save_path)
    logger.info("Drop positions save path: %s", dp_save_path)
    logger.info("Similarity info save path: %s", sim_save_path)
    logger.info("Min pixels: %s", run_config["min_pixels"])
    logger.info("Max pixels: %s", run_config["max_pixels"])
    logger.info("Max frames: %s", run_config["max_frames"])
    logger.info("Min frames: %s", run_config["min_frames"])
    logger.info("FPS: %s", run_config["fps"])
    logger.info("Temperature (T): %s", run_config["temperature"])
    if hdiff_args.clip_enabled:
        logger.info("Frame select enabled: True")
        logger.info("Frame select method: CLIP patch similarity (%s)", hdiff_args.clip_strategy)
        logger.info("Frame select drop ratio: %s", hdiff_args.clip_drop_ratio)
        logger.info("Frame select min keep frames: %s", hdiff_args.clip_min_keep_frames)
        logger.info("Frame select max interval: %s", hdiff_args.clip_max_interval)
        logger.info("CLIP stats jsonl: %s", hdiff_args.clip_stats_path)
    else:
        logger.info("Drop relative hdiff: %s", drop_relative_hdiff)
        logger.info("Frame select enabled: %s", bool(getattr(hdiff_args, "frame_select_enabled", False)))
        logger.info("Frame select method: %s", getattr(hdiff_args, "frame_select_method", "histogram"))
        logger.info("Frame select drop ratio: %s", getattr(hdiff_args, "frame_select_drop_ratio", None))
        logger.info("Frame select size: %s", getattr(hdiff_args, "frame_select_size", None))
        logger.info("Frame select min keep frames: %s", getattr(hdiff_args, "frame_select_min_keep_frames", None))
    logger.info("DataLoader batch_size: %s", args.batch_size)
    logger.info("DataLoader num_workers: %s", args.num_workers)
    logger.info("DataLoader prefetch_factor: %s", args.prefetch_factor)
    logger.info("DataLoader pin_memory: %s", args.pin_memory)
    logger.info("DataLoader persistent_workers: %s", args.persistent_workers)

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")
    with open(task_json, "r") as f:
        task_list = json.load(f)

    processor_path = resolve_processor_path(ckpt_path)

    requests = _make_requests(task_list, video_dir)
    dataset = OVORequestDataset(
        requests=requests,
        ckpt_path=processor_path,
        run_config=run_config,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    dataloader_iter = iter(dataloader)

    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    logger.info("Attention implementation: %s", attn_impl)
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=run_config["min_pixels"],
        max_pixels=run_config["max_pixels"],
    )
    logger.info("Load model from %s", ckpt_path)
    logger.info("Load processor from %s", processor_path)

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
        "frame_select": 0.0,
    }
    start_time = time.time()
    profiled_samples = run_inference_with_dataloader(
        dataloader_iter=dataloader_iter,
        total=len(dataset),
        model=model,
        processor=processor,
        run_config=run_config,
        output_jsonl=output_jsonl,
        error_jsonl=error_jsonl,
        pin_memory=args.pin_memory,
        exit_on_oom=args.exit_on_oom,
        stage_totals=stage_totals,
    )

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
            frame_drop_list, frame_total_list, frame_ratio_list = [], [], []
            with open(dr_save_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                drop_ratio_info = json.loads(line)
                drop_list.append(drop_ratio_info["drop"])
                total_list.append(drop_ratio_info["total"])
                ratio_list.append(drop_ratio_info["ratio"])
                if "frame_drop" in drop_ratio_info and "frame_total" in drop_ratio_info:
                    frame_drop_list.append(drop_ratio_info["frame_drop"])
                    frame_total_list.append(drop_ratio_info["frame_total"])
                    frame_ratio_list.append(drop_ratio_info["frame_ratio"])
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info("Total drop ratio (weighted drop ratio): %.1f%%", 100 * total_dr)
            logger.info("Average drop ratio (unweighted drop ratio): %.1f%%", 100 * avg_dr)
            if frame_total_list:
                total_frame_dr = sum(frame_drop_list) / sum(frame_total_list)
                avg_frame_dr = sum(frame_ratio_list) / len(frame_ratio_list)
                logger.info("Attention frame drop ratio (weighted): %.1f%%", 100 * total_frame_dr)
                logger.info("Attention frame drop ratio (unweighted): %.1f%%", 100 * avg_frame_dr)
        elif drop_enabled and dr_save_path is not None:
            logger.warning("Drop ratio file not found, skip statistics: %s", dr_save_path)

        stage_keys = ["video_decode", "frame_select", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode"]
        non_overlap_stage_keys = ["video_decode", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode"]
        stage_name_map = {
            "video_decode": "Preprocessing Total",
            "frame_select": "Frame Select",
            "visual_encode": "Vision Encoder",
            "token_drop": "Token Drop",
            "llm_prefill": "LLM Prefill",
            "llm_decode": "LLM Decode",
            "text_decode": "Text Detokenize (processor.batch_decode)",
        }
        measured_stage_total = sum(stage_totals[key] for key in non_overlap_stage_keys)
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
            bottleneck_key = max(non_overlap_stage_keys, key=stage_totals.get)
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
        _log_frame_compression_summary()


if __name__ == "__main__":
    hdiff_args, remaining_argv = _parse_hdiff_args(sys.argv[1:])
    _install_hist_logging_hook(hdiff_args)
    _patch_runtime(hdiff_args)
    _run_ovobench(remaining_argv, hdiff_args)
