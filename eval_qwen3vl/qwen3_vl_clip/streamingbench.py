"""StreamingBench evaluation with CLIP first-stage frame filtering.

Unlike the historical StreamingBench runner, this module filters the decoded
frames explicitly before passing them to Qwen's processor.  That makes the
first-stage model-based filtering observable and avoids a second sampling pass
inside the Qwen processor.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from qwen3_vl_hdiff_parellel import Qwen3VLForConditionalGenerationDTD

from .frame_selector import CLIPPatchFrameSelector, CLIPPatchFrameSelectorConfig
from .summarize_latency import load_records, summarize_records


LOG_FORMAT = "%(asctime)s %(levelname)7s | %(message)s"


PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""


def _resolve_processor_path(ckpt_path: str) -> str:
    adapter_config = Path(ckpt_path) / "adapter_config.json"
    if not adapter_config.is_file():
        return ckpt_path
    with adapter_config.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("base_model_name_or_path") or ckpt_path


def _time_to_seconds(value: str) -> int:
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported StreamingBench timestamp: {value!r}")


def _parse_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    # The benchmark CSV stores a Python-style list.  JSON is preferred when
    # available; eval is intentionally avoided for data supplied by a dataset.
    import ast

    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of options, got {type(parsed).__name__}")
    return parsed


def _build_messages(row: pd.Series, args: argparse.Namespace) -> tuple[list[dict], str]:
    seconds = _time_to_seconds(row.time_stamp)
    fps = args.fps
    if 300 < seconds <= 600:
        fps *= 0.5
    elif seconds > 600:
        fps *= 0.2
    video_path = Path(args.video_dir) / f"sample_{str(row.question_id).split('_')[-2]}" / "video.mp4"
    options = _parse_options(row.options)
    return (
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                        "max_frames": args.max_frames,
                        "min_frames": args.min_frames,
                        "fps": fps,
                        "video_start": 0,
                        "video_end": seconds,
                    },
                    {"type": "text", "text": PROMPT.format(row.question, "\n".join(options))},
                ],
            }
        ],
        str(video_path),
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _take_frames(video: Any, indices: Sequence[int]) -> Any:
    if torch.is_tensor(video) or isinstance(video, np.ndarray):
        return video[list(indices)]
    return [video[index] for index in indices]


def _filter_videos(video_inputs: Any, selector: CLIPPatchFrameSelector) -> Any:
    if video_inputs is None:
        return None
    if isinstance(video_inputs, (torch.Tensor, np.ndarray)) and video_inputs.ndim == 4:
        return _take_frames(video_inputs, selector.select_frames(list(video_inputs)))
    return [_take_frames(video, selector.select_frames(list(video))) for video in video_inputs]


def _extract_answer(text: str) -> str:
    cleaned = str(text).strip()
    for prefix in (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The best option is",
        "The correct option is",
    ):
        cleaned = cleaned.replace(prefix, "")
    match = re.search(r"[ABCD]", cleaned)
    return match.group(0) if match else ""


def _write_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total_seconds": 0.0, "mean_seconds_per_video": 0.0, "p50_seconds_per_video": 0.0, "p95_seconds_per_video": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "total_seconds": float(array.sum()),
        "mean_seconds_per_video": float(array.mean()),
        "p50_seconds_per_video": float(np.percentile(array, 50)),
        "p95_seconds_per_video": float(np.percentile(array, 95)),
    }


def _sample_subset(task_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Return an exact-size, deterministic subset stratified by task type."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if fraction == 1.0:
        return task_df.reset_index(drop=True)

    target = max(1, int(round(len(task_df) * fraction)))
    if target >= len(task_df):
        return task_df.reset_index(drop=True)
    if "task_type" not in task_df.columns:
        return task_df.sample(n=target, random_state=seed).reset_index(drop=True)

    groups = list(task_df.groupby("task_type", sort=True, dropna=False))
    ideal = [len(group) * target / len(task_df) for _, group in groups]
    quotas = [min(len(group), int(value)) for value, (_, group) in zip(ideal, groups)]
    remaining = target - sum(quotas)
    order = sorted(
        range(len(groups)),
        key=lambda index: (ideal[index] - quotas[index], len(groups[index][1])),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in order:
            if quotas[index] < len(groups[index][1]):
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break

    sampled = [
        group.sample(n=quota, random_state=seed + index)
        for index, ((_, group), quota) in enumerate(zip(groups, quotas))
        if quota
    ]
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("qwen3_vl_clip.streamingbench")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="8B_clip_drop20")
    parser.add_argument("--ckpt-path", default=str(repo_root / "ckpt/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--task-csv", default=str(repo_root / "datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"))
    parser.add_argument("--video-dir", default=str(repo_root / "datasets/StreamingBench/Real-Time_Visual_Understanding"))
    parser.add_argument("--result-dir", default=str(repo_root / "eval_qwen3vl/result_streamingbench"))
    parser.add_argument("--log-dir")
    parser.add_argument("--question-id")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--drop-method", choices=("none", "attention"), default="none")
    parser.add_argument("--attn-layer-idx", type=int, default=-1)
    parser.add_argument("--top-k-ratio", type=float, default=0.625)
    parser.add_argument("--scorer-chunk-size", type=int, default=512)
    parser.add_argument("--clip-model-path", default=str(repo_root / "ckpt/clip-vit-base-patch32"))
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-dtype", default="float16")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--clip-strategy", choices=("relative", "threshold"), default="relative")
    parser.add_argument("--clip-drop-ratio", type=float, default=0.2)
    parser.add_argument("--clip-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--clip-min-keep-frames", type=int, default=4)
    parser.add_argument("--clip-max-interval", type=int, default=30)
    parser.add_argument("--allow-remote-clip", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("StreamingBench evaluation requires a CUDA device for Qwen3-VL")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir)
    output_dir = result_dir / "output"
    stats_dir = result_dir / "clip_stats"
    summary_dir = result_dir / "summary"
    error_dir = result_dir / "error"
    log_dir = Path(args.log_dir) if args.log_dir else result_dir / "log-NGFF_ablation"
    for directory in (output_dir, stats_dir, summary_dir, error_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run_name}_{timestamp}"
    output_path = output_dir / f"{stem}.jsonl"
    error_path = error_dir / f"{stem}.jsonl"
    stats_path = stats_dir / f"{stem}.jsonl"
    summary_path = summary_dir / f"{stem}.json"
    log_path = log_dir / f"{stem}.log"
    logger = _configure_logger(log_path)

    selector = CLIPPatchFrameSelector(
        CLIPPatchFrameSelectorConfig(
            model_path=args.clip_model_path,
            device=args.clip_device,
            dtype=args.clip_dtype,
            batch_size=args.clip_batch_size,
            strategy=args.clip_strategy,
            drop_ratio=args.clip_drop_ratio,
            similarity_threshold=args.clip_similarity_threshold,
            min_keep_frames=args.clip_min_keep_frames,
            max_interval=args.clip_max_interval,
            stats_path=str(stats_path),
            local_files_only=not args.allow_remote_clip,
        )
    )

    task_df = pd.read_csv(args.task_csv)
    source_rows = len(task_df)
    if args.question_id:
        task_df = task_df[task_df["question_id"] == args.question_id]
    elif args.sample_fraction < 1.0:
        task_df = _sample_subset(task_df, args.sample_fraction, args.sample_seed)
    if task_df.empty:
        raise ValueError("No StreamingBench rows selected")

    logger.info("CLIP patch frame selection config enabled")
    logger.info("clip_model_path: %s", args.clip_model_path)
    logger.info("clip_mode: %s", args.clip_strategy)
    logger.info("clip_drop_ratio: %s", args.clip_drop_ratio)
    logger.info("clip_similarity_threshold: %s", args.clip_similarity_threshold)
    logger.info("clip_min_keep_frames: %s", args.clip_min_keep_frames)
    logger.info("clip_max_interval: %s", args.clip_max_interval)
    logger.info("clip_device: %s", args.clip_device)
    logger.info("clip_dtype: %s", args.clip_dtype)
    logger.info("clip_batch_size: %s", args.clip_batch_size)
    logger.info("clip_local_files_only: %s", not args.allow_remote_clip)
    logger.info("Running %s on StreamingBench", args.run_name)
    logger.info("Drop method: %s", args.drop_method)
    if args.drop_method == "attention":
        logger.info("Attention layer: %s", args.attn_layer_idx)
        logger.info("Attention top-k ratio (keep): %s", args.top_k_ratio)
        logger.info("Attention drop ratio: %s", 1.0 - args.top_k_ratio)
        logger.info("Attention scorer chunk size: %s", args.scorer_chunk_size)
    logger.info("Checkpoint path: %s", args.ckpt_path)
    logger.info("Processor checkpoint path: %s", _resolve_processor_path(args.ckpt_path))
    logger.info("Result dir: %s", result_dir)
    logger.info("Log path: %s", log_path)
    logger.info("Task csv: %s", args.task_csv)
    logger.info("Video dir: %s", args.video_dir)
    logger.info("Question id: %s", args.question_id)
    logger.info("Subset: %s/%s samples (fraction=%s, seed=%s, stratified_by=task_type)", len(task_df), source_rows, args.sample_fraction, args.sample_seed)
    logger.info("Output jsonl: %s", output_path)
    logger.info("Error jsonl: %s", error_path)
    logger.info("CLIP stats jsonl: %s", stats_path)
    logger.info("Summary json: %s", summary_path)
    logger.info("Min pixels: %s", args.min_pixels)
    logger.info("Max pixels: %s", args.max_pixels)
    logger.info("Max frames: %s", args.max_frames)
    logger.info("Min frames: %s", args.min_frames)
    logger.info("FPS: %s", args.fps)
    logger.info("Temperature (T): %s", args.temperature)
    logger.info("Resolved arguments: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

    processor_path = _resolve_processor_path(args.ckpt_path)
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    logger.info("Attention implementation: %s", attn_impl)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        processor_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    ).eval()
    logger.info("Load model and processor time: %.2fs", time.perf_counter() - load_started)
    logger.info("Load model from %s", args.ckpt_path)
    logger.info("Load processor from %s", processor_path)
    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    correct = 0
    totals: defaultdict[str, int] = defaultdict(int)
    correct_by_task: defaultdict[str, int] = defaultdict(int)
    end_to_end_latencies: list[float] = []
    stage_totals: defaultdict[str, float] = defaultdict(float)
    run_started = time.perf_counter()

    for index, row in task_df.reset_index(drop=True).iterrows():
        torch.cuda.synchronize()
        sample_started = time.perf_counter()
        video_path = ""
        try:
            preprocessing_started = time.perf_counter()
            messages, video_path = _build_messages(row, args)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(
                messages, image_patch_size=processor.video_processor.patch_size
            )
            filtered_videos = _filter_videos(video_inputs, selector)
            selector_record = selector.last_stats
            stage_totals["frame_select"] += float(selector_record.get("total_seconds", 0.0))
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=filtered_videos,
                padding=True,
                return_tensors="pt",
                do_sample_frames=False,
                do_resize=False,
            )
            inputs = _move_to_device(inputs, torch.device("cuda"))
            stage_totals["preprocessing_total"] += time.perf_counter() - preprocessing_started
            timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            gen_kwargs: dict[str, Any] = {"max_new_tokens": 128}
            if args.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": args.temperature})
            else:
                gen_kwargs["do_sample"] = False
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                    drop_method=args.drop_method,
                    attn_layer_idx=args.attn_layer_idx,
                    top_k_ratio=args.top_k_ratio,
                    scorer_chunk_size=args.scorer_chunk_size,
                )
            timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            for name in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
                stage_totals[name] += max(0.0, timing_after.get(name, 0.0) - timing_before.get(name, 0.0))
            continuation_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
            ]
            decode_started = time.perf_counter()
            response = processor.batch_decode(
                continuation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            stage_totals["text_decode"] += time.perf_counter() - decode_started
            answer = _extract_answer(response)
            task_type = str(row.task_type)
            is_correct = answer == str(row.answer)
            totals["overall"] += 1
            totals[task_type] += 1
            correct += int(is_correct)
            correct_by_task[task_type] += int(is_correct)
            # Qwen generation is asynchronous on CUDA; synchronize before
            # reporting latency so the per-video figures are comparable with
            # the synchronized CLIP forward timing.
            torch.cuda.synchronize()
            latency = time.perf_counter() - sample_started
            end_to_end_latencies.append(latency)
            _write_jsonl(
                output_path,
                {
                    "index": int(index),
                    "question_id": row.question_id,
                    "task_type": task_type,
                    "video_path": video_path,
                    "answer": str(row.answer),
                    "response": response,
                    "prediction": answer,
                    "correct": is_correct,
                    "end_to_end_seconds": latency,
                    "clip_frames_in": selector_record.get("frames_in"),
                    "clip_frames_out": selector_record.get("frames_out"),
                    "clip_filter_seconds": selector_record.get("total_seconds"),
                },
            )
            logger.info(
                "[%s/%s] %s: %s | %.2fs | frames %s->%s | clip %.3fs",
                index + 1,
                len(task_df),
                row.question_id,
                "correct" if is_correct else "wrong",
                latency,
                selector_record.get("frames_in"),
                selector_record.get("frames_out"),
                selector_record.get("total_seconds", 0.0),
            )
        except Exception as exc:
            _write_jsonl(
                error_path,
                {"index": int(index), "question_id": row.question_id, "video_path": video_path, "error": f"{type(exc).__name__}: {exc}"},
            )
            logger.exception("[%s/%s] %s: %s", index + 1, len(task_df), row.question_id, exc)

    selector_records = load_records(stats_path) if stats_path.exists() else []
    selector_summary = summarize_records(selector_records) if selector_records else {}
    completed = totals["overall"]
    overall_accuracy = correct / completed if completed else 0.0
    summary = {
        "run_name": args.run_name,
        "checkpoint": args.ckpt_path,
        "clip": selector.to_dict(),
        "attention": {
            "drop_method": args.drop_method,
            "attn_layer_idx": args.attn_layer_idx,
            "top_k_ratio": args.top_k_ratio,
        },
        "sampling": {
            "source_rows": source_rows,
            "selected_rows": len(task_df),
            "sample_fraction_requested": args.sample_fraction,
            "sample_seed": args.sample_seed,
            "stratified_by": "task_type" if not args.question_id else None,
        },
        "samples": {
            "attempted": len(task_df),
            "completed": completed,
            "failed": len(task_df) - completed,
            "correct": correct,
        },
        "accuracy": {
            "overall": overall_accuracy,
            "overall_percent": 100.0 * overall_accuracy,
            "by_task_type": {
                name: correct_by_task[name] / total for name, total in sorted(totals.items()) if name != "overall" and total
            },
        },
        "latency": {
            "end_to_end": _percentiles(end_to_end_latencies),
            "clip_filter": selector_summary,
            "qwen_profiled_stage_totals": dict(stage_totals),
            "qwen_profiled_stage_mean_seconds": {
                name: seconds / completed for name, seconds in stage_totals.items()
            }
            if completed
            else {},
            "wall_seconds": time.perf_counter() - run_started,
        },
        "artifacts": {
            "predictions_jsonl": str(output_path),
            "errors_jsonl": str(error_path),
            "clip_stats_jsonl": str(stats_path),
            "log": str(log_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for task_type, total in sorted((name, count) for name, count in totals.items() if name != "overall"):
        logger.info("- %s: %s/%s = %.2f%%", task_type, correct_by_task[task_type], total, 100.0 * correct_by_task[task_type] / total)
    logger.info("Total: %s, Correct: %s, Accuracy: %.2f%%", completed, correct, 100.0 * overall_accuracy)
    stage_keys = ("preprocessing_total", "frame_select", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode")
    stage_names = {
        "preprocessing_total": "Preprocessing Total",
        "frame_select": "Frame Select",
        "visual_encode": "Vision Encoder",
        "token_drop": "Token Drop",
        "llm_prefill": "LLM Prefill",
        "llm_decode": "LLM Decode",
        "text_decode": "Text Detokenize (processor.batch_decode)",
    }
    wall_seconds = summary["latency"]["wall_seconds"]
    non_overlap = ("preprocessing_total", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode")
    measured_total = sum(stage_totals[key] for key in non_overlap)
    logger.info("Stage time breakdown:")
    for key in stage_keys:
        seconds = stage_totals[key]
        logger.info(
            "- %s: %.2fs | %.2f%% of end-to-end | %.2f%% inside profiled stages",
            stage_names[key],
            seconds,
            100.0 * seconds / wall_seconds if wall_seconds else 0.0,
            100.0 * seconds / measured_total if measured_total else 0.0,
        )
    other_seconds = max(0.0, wall_seconds - measured_total)
    logger.info("- Other/uncaptured: %.2fs | %.2f%% of end-to-end", other_seconds, 100.0 * other_seconds / wall_seconds if wall_seconds else 0.0)
    if measured_total:
        bottleneck = max(non_overlap, key=lambda key: stage_totals[key])
        logger.info("Profiled-stage bottleneck: %s (%.2f%% of profiled stages)", stage_names[bottleneck], 100.0 * stage_totals[bottleneck] / measured_total)
    if completed:
        logger.info("Profiled samples: %s", completed)
        for key in stage_keys:
            logger.info("- Avg %s per sample: %.4fs", stage_names[key], stage_totals[key] / completed)
    if selector_summary:
        logger.info("CLIP frame selection summary:")
        logger.info("- Raw frames: %s", selector_summary["frames_in"])
        logger.info("- Kept frames: %s", selector_summary["frames_out"])
        logger.info("- Video frame compression ratio: %.2f%%", 100.0 * (1.0 - selector_summary["keep_ratio"]))
        logger.info("- Avg raw frames per sample: %.2f", selector_summary["frames_in"] / selector_summary["videos"])
        logger.info("- Avg kept frames per sample: %.2f", selector_summary["frames_out"] / selector_summary["videos"])
        logger.info("- Frame select time: %.2fs", selector_summary["clip_filter_total_seconds"])
        logger.info("- CLIP forward time: %.2fs", selector_summary["clip_forward_total_seconds"])
    logger.info("Inference cost time: %sh %sm %ss", int(wall_seconds) // 3600, (int(wall_seconds) % 3600) // 60, int(wall_seconds) % 60)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
