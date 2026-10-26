#!/usr/bin/env python3
"""Counterfactual NGFF recall analysis on the 10% StreamingBench subset.

NGFF produces frame masks without removing frames.  Qwen3-VL sees the full
candidate video once, and its question/options-to-video Attention is reused to
score NGFF, uniform, and random masks at 20% and 50% drop ratios.
"""

import argparse
import ast
import csv
import importlib.util
import json
import logging
import math
import os
import os.path as osp
import sys
import time
import types
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor


SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
REPO_ROOT = osp.abspath(osp.join(SCRIPT_DIR, "..", "..", ".."))
EVAL_ROOT = osp.join(REPO_ROOT, "eval_qwen3vl")
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from qwen3_vl_hdiff_parellel_attn_v3_NGFF_recall import (  # noqa: E402
    Qwen3VLForConditionalGenerationDTD,
)
LOGGER = logging.getLogger("streamingbench_ngff_recall")
PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""
DROP_RATIOS = (0.20, 0.50)
METHOD_ORDER = ("Uniform", "Random", "NGFF")


def _load_frame_selector_types():
    """Load the copied processor under a Transformers-compatible package name."""
    package_name = "transformers.models.qwen3_vl_ngff_recall"
    package_dir = osp.join(
        EVAL_ROOT, "qwen3_vl_hdiff_parellel_attn_v3_NGFF_recall"
    )
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [package_dir]
        sys.modules[package_name] = package
    module_name = package_name + ".video_processing_qwen3_vl"
    if module_name not in sys.modules:
        file_path = osp.join(package_dir, "video_processing_qwen3_vl.py")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[module_name]
    return module.HistogramFrameSelector, module.HistogramFrameSelectorConfig


HistogramFrameSelector, HistogramFrameSelectorConfig = _load_frame_selector_types()


def write_jsonl(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a" if osp.exists(path) else "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_timestamp(value):
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported timestamp: {value!r}")


def resolve_processor_path(ckpt_path):
    adapter_config = osp.join(ckpt_path, "adapter_config.json")
    if not osp.isfile(adapter_config):
        return ckpt_path
    with open(adapter_config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("base_model_name_or_path") or ckpt_path


def build_messages(row, video_dir, min_pixels, max_pixels, min_frames, max_frames, base_fps):
    timestamp = parse_timestamp(row.time_stamp)
    fps = base_fps
    if 300 < timestamp <= 600:
        fps *= 0.5
    elif timestamp > 600:
        fps *= 0.2
    sample_id = str(row.question_id).split("_")[-2]
    video_path = osp.join(video_dir, f"sample_{sample_id}", "video.mp4")
    options = ast.literal_eval(row.options) if isinstance(row.options, str) else list(row.options)
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
                    "video_start": 0,
                    "video_end": timestamp,
                },
                {
                    "type": "text",
                    "text": PROMPT.format(row.question, "\n".join(options)),
                },
            ],
        }
    ]
    return messages, video_path, options


def _find_subsequence(sequence, pattern, start=0):
    if not pattern:
        raise ValueError("Marker tokenization produced an empty token sequence")
    matches = []
    limit = len(sequence) - len(pattern) + 1
    for idx in range(max(0, start), max(0, limit)):
        if sequence[idx : idx + len(pattern)] == pattern:
            matches.append(idx)
    return matches


def _find_unique_marker(tokenizer, input_ids, marker, start):
    for variant in (marker, "\n" + marker, "\n\n" + marker):
        pattern = tokenizer(variant, add_special_tokens=False)["input_ids"]
        matches = _find_subsequence(input_ids, pattern, start=start)
        if len(matches) == 1:
            begin = matches[0]
            return begin, begin + len(pattern)
        if len(matches) > 1:
            raise ValueError(
                f"Expected one {marker!r} marker after the video, found starts={matches}"
            )
    raise ValueError(f"Could not locate {marker!r} after the video tokens")


def build_question_options_mask(tokenizer, input_ids, video_token_id):
    """Select only question content and option content in the final prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected batch-one input_ids, got {tuple(input_ids.shape)}")
    ids = input_ids[0].tolist()
    video_positions = [idx for idx, token in enumerate(ids) if token == video_token_id]
    if not video_positions:
        raise ValueError("No video tokens found while building qer_query_mask")
    search_start = video_positions[-1] + 1
    question_begin, question_label_end = _find_unique_marker(
        tokenizer, ids, "Question", search_start
    )
    options_begin, options_label_end = _find_unique_marker(
        tokenizer, ids, "Options", question_label_end
    )
    answer_begin, _ = _find_unique_marker(
        tokenizer, ids, "The best option is", options_label_end
    )
    # A colon may merge with the following newline in the full prompt even
    # though it is a standalone token when the label is tokenized alone.
    question_end = question_label_end
    if question_end < len(ids) and tokenizer.decode([ids[question_end]]).strip() == ":":
        question_end += 1
    options_end = options_label_end
    if options_end < len(ids) and tokenizer.decode([ids[options_end]]).strip() == ":":
        options_end += 1
    if not question_end < options_begin < options_end < answer_begin:
        raise ValueError("Question/options marker order is invalid")

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[0, question_end:options_begin] = True
    mask[0, options_end:answer_begin] = True
    mask[0, answer_begin:] = False
    video_mask = input_ids == video_token_id
    if bool((mask & video_mask).any()):
        raise ValueError("qer_query_mask unexpectedly includes video tokens")
    if int(mask.sum().item()) == 0:
        raise ValueError("qer_query_mask contains no question/options tokens")
    return mask


def _extract_video_tensor(video_inputs):
    if video_inputs is None or len(video_inputs) != 1:
        count = 0 if video_inputs is None else len(video_inputs)
        raise ValueError(f"StreamingBench expects exactly one video input, got {count}")
    video = video_inputs[0]
    if isinstance(video, tuple):
        video = video[0]
    if isinstance(video, list):
        if not video:
            raise ValueError("Decoded video contains no frames")
        if torch.is_tensor(video[0]):
            video = torch.stack(video)
        else:
            raise TypeError(f"Unsupported decoded frame type: {type(video[0]).__name__}")
    if not torch.is_tensor(video):
        raise TypeError(f"Unsupported decoded video type: {type(video).__name__}")
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected decoded video [T,C,H,W], got {tuple(video.shape)}")
    return video


def compute_ngff_masks(frames, ratios, bins, min_keep_frames, max_interval, size):
    masks = {}
    frame_list = list(frames)
    for ratio in ratios:
        selector = HistogramFrameSelector(
            HistogramFrameSelectorConfig(
                enabled=True,
                method="histogram",
                bins=bins,
                threshold=float(ratio),
                relative=True,
                min_keep_frames=min_keep_frames,
                max_interval=max_interval,
                size=size,
            )
        )
        masks[f"{ratio:.2f}"] = [int(idx) for idx in selector.select_frames(frame_list)]
    return masks


class StreamingBenchNGFFRecallDataset(Dataset):
    def __init__(self, task_df, args, processor_path):
        self.task_df = task_df.reset_index(drop=True)
        self.args = args
        self.processor_path = processor_path
        self._processor = None

    def __len__(self):
        return len(self.task_df)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.processor_path,
                min_pixels=self.args.min_pixels,
                max_pixels=self.args.max_pixels,
                local_files_only=True,
            )
        return self._processor

    def __getitem__(self, idx):
        row = self.task_df.iloc[idx]
        metadata = {
            "idx": int(idx),
            "question_id": str(row.question_id),
            "task_type": str(row.task_type),
            "frames_required": str(row.frames_required),
            "temporal_clue_type": str(row.temporal_clue_type),
            "video_id": str(row.question_id).split("_")[-2],
        }
        started = time.perf_counter()
        try:
            processor = self._get_processor()
            messages, video_path, _ = build_messages(
                row,
                self.args.video_dir,
                self.args.min_pixels,
                self.args.max_pixels,
                self.args.min_frames,
                self.args.max_frames,
                self.args.fps,
            )
            metadata["video_path"] = video_path
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=processor.video_processor.patch_size,
            )
            if video_inputs and isinstance(video_inputs[0], tuple):
                video_kwargs["video_metadata"] = [item[1] for item in video_inputs]
                video_inputs = [item[0] for item in video_inputs]
            # qwen-vl-utils returns one FPS value per video, while the local
            # Transformers 5.6 processor validates a scalar for batch-one use.
            if isinstance(video_kwargs.get("fps"), list):
                if len(video_kwargs["fps"]) != 1:
                    raise ValueError(
                        f"Expected one video FPS value, got {video_kwargs['fps']}"
                    )
                video_kwargs["fps"] = float(video_kwargs["fps"][0])
            video_tensor = _extract_video_tensor(video_inputs)
            candidate_frames = int(video_tensor.shape[0])
            ngff_masks = compute_ngff_masks(
                video_tensor,
                DROP_RATIOS,
                self.args.hist_bins,
                self.args.hist_min_keep_frames,
                self.args.hist_max_interval,
                self.args.hist_size,
            )

            # NGFF is deliberately disabled inside the processor: all frames
            # continue to the visual encoder and QER Attention probe.
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            query_mask = build_question_options_mask(
                processor.tokenizer, inputs["input_ids"], video_token_id
            )
            metadata.update(
                candidate_frames=candidate_frames,
                ngff_masks=ngff_masks,
                qer_query_tokens=int(query_mask.sum().item()),
            )
            return {
                "inputs": inputs,
                "qer_query_mask": query_mask,
                "metadata": metadata,
                "preprocess_seconds": time.perf_counter() - started,
                "error": None,
            }
        except Exception as exc:
            return {
                "inputs": None,
                "qer_query_mask": None,
                "metadata": metadata,
                "preprocess_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_one(batch):
    if len(batch) != 1:
        raise ValueError("NGFF recall analysis currently supports batch_size=1 only")
    return batch[0]


def move_to_device(value, device, non_blocking):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: move_to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    return value


def uniform_keep_indices(total, keep_count):
    if not 0 < keep_count <= total:
        raise ValueError(f"Invalid uniform keep count {keep_count}/{total}")
    if keep_count == total:
        return list(range(total))
    if keep_count == 1:
        return [0]
    selected = np.linspace(0, total - 1, keep_count).round().astype(int).tolist()
    if len(set(selected)) != keep_count:
        raise RuntimeError("Uniform sampling did not produce the requested unique count")
    return sorted(selected)


def random_keep_indices(total, keep_count, base_seed, sample_idx, ratio_key, repeat):
    ratio_int = int(round(float(ratio_key) * 1000))
    seed = np.random.SeedSequence([base_seed, sample_idx, ratio_int, repeat])
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=keep_count, replace=False).astype(int).tolist())


def temporal_unit_members(candidate_frames, temporal_patch_size):
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be positive")
    return [
        list(range(start, min(start + temporal_patch_size, candidate_frames)))
        for start in range(0, candidate_frames, temporal_patch_size)
    ]


def compute_recall_metrics(frame_attention, unit_members, keep_indices, top_ratio=0.20):
    if len(frame_attention) != len(unit_members):
        raise ValueError(
            f"Attention/unit mismatch: {len(frame_attention)} vs {len(unit_members)}"
        )
    total_attention = float(sum(frame_attention))
    if not np.isclose(total_attention, 1.0, atol=1e-5):
        raise ValueError(f"Frame Attention is not normalized: sum={total_attention}")
    keep_set = set(int(idx) for idx in keep_indices)
    preserved = [all(frame_idx in keep_set for frame_idx in members) for members in unit_members]
    attention_preservation = float(
        sum(score for score, is_kept in zip(frame_attention, preserved) if is_kept)
    )
    top_count = max(1, int(math.ceil(len(frame_attention) * top_ratio)))
    ranked = sorted(
        range(len(frame_attention)), key=lambda idx: (-frame_attention[idx], idx)
    )
    top_indices = ranked[:top_count]
    top_recall = float(sum(preserved[idx] for idx in top_indices) / top_count)
    return {
        "attention_preservation": attention_preservation,
        "qer_top_evidence_recall": top_recall,
        "top_count": top_count,
        "top_indices": top_indices,
        "preserved_unit_count": int(sum(preserved)),
        "preserved_units": preserved,
    }


def add_condition_rows(rows, metadata, frame_attention, unit_members, random_seeds):
    total = int(metadata["candidate_frames"])
    for ratio in DROP_RATIOS:
        ratio_key = f"{ratio:.2f}"
        ngff_keep = [int(idx) for idx in metadata["ngff_masks"][ratio_key]]
        keep_count = len(ngff_keep)
        conditions = [("NGFF", None, ngff_keep)]
        conditions.append(("Uniform", None, uniform_keep_indices(total, keep_count)))
        for repeat in range(random_seeds):
            conditions.append(
                (
                    "Random",
                    repeat,
                    random_keep_indices(
                        total,
                        keep_count,
                        base_seed=42,
                        sample_idx=int(metadata["idx"]),
                        ratio_key=ratio_key,
                        repeat=repeat,
                    ),
                )
            )
        for method, random_seed, keep_indices in conditions:
            metrics = compute_recall_metrics(
                frame_attention, unit_members, keep_indices, top_ratio=0.20
            )
            rows.append(
                {
                    "question_id": metadata["question_id"],
                    "video_id": metadata["video_id"],
                    "task_type": metadata["task_type"],
                    "frames_required": metadata["frames_required"],
                    "method": method,
                    "nominal_drop_ratio": ratio,
                    "actual_drop_ratio": 1.0 - keep_count / total,
                    "random_seed": random_seed,
                    "candidate_frames": total,
                    "kept_frames": keep_count,
                    "qer_units": len(frame_attention),
                    "top_count": metrics["top_count"],
                    "attention_preservation": metrics["attention_preservation"],
                    "qer_top_evidence_recall": metrics["qer_top_evidence_recall"],
                }
            )


def _collapse_random_repeats(frame):
    keys = [
        "question_id",
        "video_id",
        "task_type",
        "frames_required",
        "method",
        "nominal_drop_ratio",
    ]
    numeric = [
        "actual_drop_ratio",
        "candidate_frames",
        "kept_frames",
        "qer_units",
        "top_count",
        "attention_preservation",
        "qer_top_evidence_recall",
    ]
    random_rows = frame[frame.method == "Random"].groupby(keys, as_index=False)[numeric].mean()
    other_rows = frame[frame.method != "Random"].drop(columns=["random_seed"])
    return pd.concat([other_rows, random_rows], ignore_index=True)


def _cluster_bootstrap(group, metric, iterations, seed):
    clusters = {
        video_id: values[metric].to_numpy(dtype=float)
        for video_id, values in group.groupby("video_id", sort=True)
    }
    keys = sorted(clusters)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.integers(0, len(keys), size=len(keys))
        values = np.concatenate([clusters[keys[idx]] for idx in selected])
        samples[iteration] = values.mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_main_table(rows, output_csv, output_md, bootstrap_iterations):
    if not rows:
        raise ValueError("No successful metric rows were produced")
    frame = _collapse_random_repeats(pd.DataFrame(rows))
    table_rows = []
    for ratio in DROP_RATIOS:
        for method in METHOD_ORDER:
            group = frame[
                (frame.nominal_drop_ratio == ratio) & (frame.method == method)
            ]
            if group.empty:
                continue
            ap_low, ap_high = _cluster_bootstrap(
                group, "attention_preservation", bootstrap_iterations, seed=42
            )
            top_low, top_high = _cluster_bootstrap(
                group, "qer_top_evidence_recall", bootstrap_iterations, seed=43
            )
            table_rows.append(
                {
                    "nominal_drop_percent": 100 * ratio,
                    "method": method,
                    "questions": int(group.question_id.nunique()),
                    "actual_drop_percent": 100 * group.actual_drop_ratio.mean(),
                    "attention_preservation_percent": 100 * group.attention_preservation.mean(),
                    "attention_preservation_ci_low": 100 * ap_low,
                    "attention_preservation_ci_high": 100 * ap_high,
                    "qer_top_evidence_recall_at_20_percent": 100 * group.qer_top_evidence_recall.mean(),
                    "qer_top_evidence_recall_ci_low": 100 * top_low,
                    "qer_top_evidence_recall_ci_high": 100 * top_high,
                }
            )
    table = pd.DataFrame(table_rows)
    table.to_csv(output_csv, index=False, float_format="%.4f")

    lines = [
        "| Nominal Drop | Method | Questions | Actual Drop | Attention Preservation | QER Top-Evidence Recall@20% |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            "| {nominal_drop_percent:.0f}% | {method} | {questions} | "
            "{actual_drop_percent:.2f}% | {attention_preservation_percent:.2f}% "
            "[{attention_preservation_ci_low:.2f}, {attention_preservation_ci_high:.2f}] | "
            "{qer_top_evidence_recall_at_20_percent:.2f}% "
            "[{qer_top_evidence_recall_ci_low:.2f}, {qer_top_evidence_recall_ci_high:.2f}] |".format(**row)
        )
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return table


def build_dataloader(dataset, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_one,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(**kwargs)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL NGFF recall analysis")
    os.makedirs(args.result_dir, exist_ok=True)
    log_dir = args.log_dir or args.result_dir
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = osp.join(args.result_dir, f"{args.run_name}_{stamp}")
    log_path = osp.join(log_dir, f"{args.run_name}_{stamp}.log")
    probe_path = prefix + "_probe.jsonl"
    metric_path = prefix + "_metrics.jsonl"
    error_path = prefix + "_errors.jsonl"
    table_csv = prefix + "_main_table.csv"
    table_md = prefix + "_main_table.md"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)7s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    task_df = pd.read_csv(args.task_csv)
    if len(task_df) != 250:
        raise ValueError(
            f"This evaluator requires the fixed 10% StreamingBench subset (250 rows), got {len(task_df)}"
        )
    if args.question_id:
        task_df = task_df[task_df.question_id == args.question_id]
    if args.max_samples is not None:
        task_df = task_df.head(args.max_samples)

    processor_path = resolve_processor_path(args.ckpt_path)
    dataset = StreamingBenchNGFFRecallDataset(task_df, args, processor_path)
    dataloader = build_dataloader(dataset, args)
    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        local_files_only=True,
    ).eval()
    temporal_patch_size = int(model.config.vision_config.temporal_patch_size)
    LOGGER.info("10%% subset: %s samples", len(dataset))
    LOGGER.info("NGFF drop ratios: %s", DROP_RATIOS)
    LOGGER.info("QER top-evidence ratio: 20%%")
    LOGGER.info("Temporal patch size: %s (complete-unit preservation)", temporal_patch_size)

    metric_rows = []
    processed = 0
    started = time.perf_counter()
    for batch in tqdm(dataloader, total=len(dataset)):
        metadata = batch["metadata"]
        if batch["error"]:
            write_jsonl(error_path, {"stage": "preprocess", "metadata": metadata, "error": batch["error"]})
            continue
        try:
            inputs = move_to_device(batch["inputs"], model.device, args.pin_memory)
            query_mask = move_to_device(batch["qer_query_mask"], model.device, args.pin_memory)
            model.reset_ngff_recall_probe()
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    drop_method="attention",
                    top_k_ratio=1.0,
                    attn_layer_idx=args.attn_layer_idx,
                    scorer_chunk_size=args.scorer_chunk_size,
                    attn_frame_validation_strict=True,
                    ngff_recall_probe=True,
                    qer_query_mask=query_mask,
                    attention_probe_only=True,
                )
            probe_batch = model.get_last_ngff_recall_probe()
            if not probe_batch or len(probe_batch) != 1:
                raise RuntimeError("Model did not return one NGFF recall Attention probe")
            probe = probe_batch[0]
            frame_attention = [float(value) for value in probe["frame_attention"]]
            members = temporal_unit_members(
                int(metadata["candidate_frames"]), temporal_patch_size
            )
            if len(frame_attention) != len(members):
                raise ValueError(
                    "QER temporal-unit count does not match decoded candidates: "
                    f"attention={len(frame_attention)}, expected={len(members)}"
                )
            sample_rows = []
            add_condition_rows(
                sample_rows,
                metadata,
                frame_attention,
                members,
                random_seeds=args.random_seeds,
            )
            for row in sample_rows:
                write_jsonl(metric_path, row)
            metric_rows.extend(sample_rows)
            write_jsonl(
                probe_path,
                {
                    "question_id": metadata["question_id"],
                    "video_id": metadata["video_id"],
                    "task_type": metadata["task_type"],
                    "candidate_frames": metadata["candidate_frames"],
                    "temporal_patch_size": temporal_patch_size,
                    "unit_members": members,
                    "frame_attention": frame_attention,
                    "ngff_masks": metadata["ngff_masks"],
                    "qer_query_tokens": metadata["qer_query_tokens"],
                    "probe_validation": probe["validation"],
                },
            )
            processed += 1
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            write_jsonl(
                error_path,
                {"stage": "probe", "metadata": metadata, "error": f"{type(exc).__name__}: {exc}"},
            )
            if args.exit_on_oom and "out of memory" in str(exc).lower():
                raise
        except Exception as exc:
            write_jsonl(
                error_path,
                {"stage": "probe", "metadata": metadata, "error": f"{type(exc).__name__}: {exc}"},
            )

    table = build_main_table(
        metric_rows,
        table_csv,
        table_md,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    LOGGER.info("Processed %s/%s samples in %.2fs", processed, len(dataset), time.perf_counter() - started)
    LOGGER.info("Metric rows: %s", metric_path)
    LOGGER.info("Run log: %s", log_path)
    LOGGER.info("Main table CSV: %s", table_csv)
    LOGGER.info("Main table Markdown: %s", table_md)
    LOGGER.info("\n%s", table.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default="Qwen3VL_8B_NGFF_recall")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--task_csv", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--question_id", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--hist_bins", type=int, default=32)
    parser.add_argument("--hist_min_keep_frames", type=int, default=4)
    parser.add_argument("--hist_max_interval", type=int, default=15)
    parser.add_argument("--hist_size", type=int, default=32)
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--scorer_chunk_size", type=int, default=512)
    parser.add_argument("--random_seeds", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--exit_on_oom", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
