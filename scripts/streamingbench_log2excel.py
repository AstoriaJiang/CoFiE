#!/usr/bin/env python3
import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd


CATEGORY_NAMES = [
    "Object Perception",
    "Causal Reasoning",
    "Clips Summarize",
    "Attribute Perception",
    "Event Understanding",
    "Text-Rich Understanding",
    "Prospective Reasoning",
    "Spatial Understanding",
    "Action Perception",
    "Counting",
]

STAGE_NAMES = [
    "Preprocessing Total",
    "Frame Select",
    "Vision Encoder",
    "Token Drop",
    "LLM Prefill",
    "LLM Decode",
    "Text Detokenize (processor.batch_decode)",
    "Other/uncaptured",
]

DEFAULT_LOG_DIRS = [
    "eval_qwen3vl/result_streamingbench/log-NGFF_ablation",
    "eval_qwen3vl/result_streamingbench/log-attn_layer",
]

DEFAULT_OUTPUT = "eval_qwen3vl/result_streamingbench/streamingbench_sweep_logs.xlsx"


def parse_duration_seconds(value):
    value = str(value).strip()
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"([\d.]+)\s*([hms])", value):
        matched = True
        amount = float(amount)
        if unit == "h":
            total += amount * 3600
        elif unit == "m":
            total += amount * 60
        else:
            total += amount
    if matched:
        return total
    try:
        return float(value)
    except ValueError:
        return None


def parse_bool(value):
    if value is None:
        return None
    value = str(value).strip()
    if value in {"True", "true", "1", "yes", "Yes"}:
        return True
    if value in {"False", "false", "0", "no", "No"}:
        return False
    return value


def parse_number(value):
    if value is None:
        return None
    value = str(value).strip()
    if value in {"None", ""}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def clean_col(value):
    value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return value or "unknown"


def message_from_line(line):
    match = re.search(r"\bINFO\s+\|\s*(.*)$", line)
    if match:
        return match.group(1).strip()
    return None


def timestamp_from_line(line):
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", line)
    return match.group(1) if match else None


def parse_run_timestamp(log_path):
    match = re.search(r"_(\d{8}_\d{6})\.log$", os.path.basename(log_path))
    return match.group(1) if match else None


def infer_subset_from_task_csv(task_csv):
    if not task_csv:
        return None, None
    match = re.search(r"streamingbench_(\d+(?:\.\d+)?)pct_seed(\d+)", task_csv)
    if not match:
        return None, None
    return float(match.group(1)), int(match.group(2))


def infer_experiment_type(row):
    drop_method = str(row.get("drop_method") or "").lower()
    hist_enabled = row.get("hist_enabled")
    run_name = str(row.get("run_name") or "").lower()
    if drop_method == "attention" or "attn_layer" in run_name:
        return "attention"
    if hist_enabled is True or "hdiff" in run_name:
        return "hdiff"
    return "baseline"


def parse_key_value(row, msg):
    kv_patterns = {
        "drop_method": r"Drop method:\s*(.+)",
        "drop_threshold": r"Drop threshold:\s*(.+)",
        "attention_layer": r"Attention layer:\s*(.+)",
        "top_k_ratio_keep": r"Top-k ratio \(keep\):\s*(.+)",
        "scorer_chunk_size": r"Scorer chunk size:\s*(.+)",
        "ckpt_path": r"Checkpoint path:\s*(.+)",
        "result_dir": r"Result dir:\s*(.+)",
        "task_csv": r"Task csv:\s*(.+)",
        "video_dir": r"Video dir:\s*(.+)",
        "question_id": r"Question id:\s*(.+)",
        "output_jsonl": r"Output jsonl:\s*(.+)",
        "error_jsonl": r"Error jsonl:\s*(.+)",
        "drop_jsonl": r"Drop ratio info save path:\s*(.+)",
        "drop_pos_jsonl": r"Drop positions save path:\s*(.+)",
        "similarity_jsonl": r"Similarity info save path:\s*(.+)",
        "min_pixels": r"Min pixels:\s*(.+)",
        "max_pixels": r"Max pixels:\s*(.+)",
        "max_frames": r"Max frames:\s*(.+)",
        "min_frames": r"Min frames:\s*(.+)",
        "temperature": r"Temperature \(T\):\s*(.+)",
        "drop_relative_hdiff": r"Drop relative hdiff:\s*(.+)",
        "hist_enabled": r"Histogram frame selection enabled:\s*(.+)",
        "hist_mode": r"Histogram frame selection mode:\s*(.+)",
        "hist_bins": r"Histogram bins:\s*(.+)",
        "hist_threshold": r"Histogram threshold:\s*(.+)",
        "hist_min_keep_frames": r"Histogram min_keep_frames:\s*(.+)",
        "hist_max_interval": r"Histogram max_interval:\s*(.+)",
        "dataloader_batch_size": r"DataLoader batch_size:\s*(.+)",
        "dataloader_num_workers": r"DataLoader num_workers:\s*(.+)",
        "dataloader_prefetch_factor": r"DataLoader prefetch_factor:\s*(.+)",
        "dataloader_pin_memory": r"DataLoader pin_memory:\s*(.+)",
        "dataloader_persistent_workers": r"DataLoader persistent_workers:\s*(.+)",
        "attention_impl": r"Attention implementation:\s*(.+)",
        "load_model_from": r"Load model from\s*(.+)",
        "load_processor_from": r"Load processor from\s*(.+)",
    }
    for key, pattern in kv_patterns.items():
        match = re.match(pattern, msg)
        if not match:
            continue
        value = match.group(1).strip()
        if key in {"drop_relative_hdiff", "hist_enabled", "dataloader_pin_memory", "dataloader_persistent_workers"}:
            row[key] = parse_bool(value)
        elif key in {
            "drop_threshold",
            "attention_layer",
            "top_k_ratio_keep",
            "scorer_chunk_size",
            "min_pixels",
            "max_pixels",
            "max_frames",
            "min_frames",
            "temperature",
            "hist_bins",
            "hist_threshold",
            "hist_min_keep_frames",
            "hist_max_interval",
            "dataloader_batch_size",
            "dataloader_num_workers",
            "dataloader_prefetch_factor",
        }:
            row[key] = parse_number(value)
        else:
            row[key] = value
        return True
    return False


def parse_log(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    row = {
        "log_file": os.path.basename(log_path),
        "log_path": os.path.realpath(log_path),
        "log_dir": os.path.dirname(os.path.realpath(log_path)),
        "run_timestamp": parse_run_timestamp(log_path),
        "completed": False,
    }
    task_rows = []
    stage_rows = []
    stage_metrics = {}
    avg_stage_metrics = {}
    first_ts = None
    last_ts = None

    for line in lines:
        ts = timestamp_from_line(line)
        if ts:
            first_ts = first_ts or ts
            last_ts = ts

        msg = message_from_line(line)
        if not msg:
            continue

        match = re.match(r"Running\s+(.+?)\s+on StreamingBench", msg)
        if match:
            row["run_name"] = match.group(1)
            continue

        if msg == "Drop absolute":
            row["drop_mode"] = "absolute"
            continue
        if msg == "Drop relative":
            row["drop_mode"] = "relative"
            continue

        if parse_key_value(row, msg):
            continue

        task_match = re.match(r"-\s+(.+?):\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", msg)
        if task_match and task_match.group(1) in CATEGORY_NAMES:
            task_type = task_match.group(1)
            correct = int(task_match.group(2))
            total = int(task_match.group(3))
            acc = float(task_match.group(4))
            safe = clean_col(task_type)
            row[f"task_{safe}_correct"] = correct
            row[f"task_{safe}_total"] = total
            row[f"task_{safe}_acc_pct"] = acc
            task_rows.append(
                {
                    "log_file": row["log_file"],
                    "run_name": row.get("run_name"),
                    "task_type": task_type,
                    "correct": correct,
                    "total": total,
                    "acc_pct": acc,
                }
            )
            continue

        total_match = re.match(r"Total:\s+(\d+),\s+Correct:\s+(\d+),\s+Accuracy:\s+([\d.]+)%", msg)
        if total_match:
            row["total_samples"] = int(total_match.group(1))
            row["total_correct"] = int(total_match.group(2))
            row["accuracy_pct"] = float(total_match.group(3))
            row["completed"] = True
            continue

        drop_total_match = re.match(r"Total drop ratio \(weighted drop ratio\):\s+([\d.]+)%", msg)
        if drop_total_match:
            row["token_total_drop_ratio_pct"] = float(drop_total_match.group(1))
            continue

        drop_avg_match = re.match(r"Average drop ratio \(unweighted drop ratio\):\s+([\d.]+)%", msg)
        if drop_avg_match:
            row["token_avg_drop_ratio_pct"] = float(drop_avg_match.group(1))
            continue

        stage_match = re.match(
            r"-\s+(.+?):\s+([\d.]+)s\s+\|\s+([\d.]+)% of end-to-end(?:\s+\|\s+([\d.]+)% inside profiled stages)?",
            msg,
        )
        if stage_match:
            stage_name = stage_match.group(1)
            sec = float(stage_match.group(2))
            pct_end_to_end = float(stage_match.group(3))
            pct_profiled = float(stage_match.group(4)) if stage_match.group(4) is not None else None
            stage_metrics[stage_name] = {
                "sec": sec,
                "pct_end_to_end": pct_end_to_end,
                "pct_profiled": pct_profiled,
            }
            continue

        avg_stage_match = re.match(r"-\s+Avg\s+(.+?)\s+per sample:\s+([\d.]+)s", msg)
        if avg_stage_match:
            avg_stage_metrics[avg_stage_match.group(1)] = float(avg_stage_match.group(2))
            continue

        bottleneck_match = re.match(r"Profiled-stage bottleneck:\s+(.+?)\s+\(([\d.]+)% of profiled stages\)", msg)
        if bottleneck_match:
            row["profiled_stage_bottleneck"] = bottleneck_match.group(1)
            row["profiled_stage_bottleneck_pct"] = float(bottleneck_match.group(2))
            continue

        profiled_match = re.match(r"Profiled samples:\s+(\d+)", msg)
        if profiled_match:
            row["profiled_samples"] = int(profiled_match.group(1))
            continue

        runtime_match = re.match(r"Inference cost time:\s+(.+)", msg)
        if runtime_match:
            row["total_runtime_hms"] = runtime_match.group(1).strip()
            row["total_runtime_sec"] = parse_duration_seconds(row["total_runtime_hms"])
            if row["total_runtime_sec"] is not None:
                row["total_runtime_min"] = row["total_runtime_sec"] / 60
            continue

        hist_patterns = {
            "hist_samples": r"-\s+Samples:\s+(\d+)",
            "hist_raw_frames": r"-\s+Raw frames:\s+(\d+)",
            "hist_kept_frames": r"-\s+Kept frames:\s+(\d+)",
            "hist_video_frame_compression_ratio_pct": r"-\s+Video frame compression ratio:\s+([\d.]+)%",
            "hist_avg_raw_frames_per_sample": r"-\s+Avg raw frames per sample:\s+([\d.]+)",
            "hist_avg_kept_frames_per_sample": r"-\s+Avg kept frames per sample:\s+([\d.]+)",
            "hist_frame_select_time_sec": r"-\s+Frame select time:\s+([\d.]+)s",
        }
        for key, pattern in hist_patterns.items():
            match = re.match(pattern, msg)
            if match:
                row[key] = parse_number(match.group(1))
                break

    row["log_start_time"] = first_ts
    row["log_end_time"] = last_ts
    row["experiment_type"] = infer_experiment_type(row)
    row["model_name"] = os.path.basename(str(row.get("ckpt_path") or "").rstrip("/")) or None
    subset_pct, sample_seed = infer_subset_from_task_csv(row.get("task_csv"))
    row["subset_pct"] = subset_pct
    row["sample_seed"] = sample_seed
    if row.get("top_k_ratio_keep") is not None:
        row["attention_drop_ratio"] = 1 - float(row["top_k_ratio_keep"])

    for stage_name, metric in stage_metrics.items():
        safe = clean_col(stage_name)
        row[f"stage_{safe}_sec"] = metric["sec"]
        row[f"stage_{safe}_pct_end_to_end"] = metric["pct_end_to_end"]
        row[f"stage_{safe}_pct_profiled"] = metric["pct_profiled"]

    for stage_name, avg_sec in avg_stage_metrics.items():
        safe = clean_col(stage_name)
        row[f"avg_{safe}_sec_per_sample"] = avg_sec

    non_overlap_stage_names = [
        "Preprocessing Total",
        "Vision Encoder",
        "Token Drop",
        "LLM Prefill",
        "LLM Decode",
        "Text Detokenize (processor.batch_decode)",
    ]
    row["profiled_stage_total_sec"] = sum(stage_metrics.get(name, {}).get("sec", 0.0) for name in non_overlap_stage_names)
    row["model_compute_sec"] = sum(stage_metrics.get(name, {}).get("sec", 0.0) for name in ["Vision Encoder", "Token Drop", "LLM Prefill", "LLM Decode"])

    for task_row in task_rows:
        task_row["experiment_type"] = row["experiment_type"]
        task_row["run_timestamp"] = row.get("run_timestamp")
        task_row["log_path"] = row["log_path"]
        task_row["accuracy_pct"] = row.get("accuracy_pct")

    for stage_name in STAGE_NAMES:
        metric = stage_metrics.get(stage_name)
        if not metric and stage_name not in avg_stage_metrics:
            continue
        stage_rows.append(
            {
                "log_file": row["log_file"],
                "run_name": row.get("run_name"),
                "experiment_type": row["experiment_type"],
                "run_timestamp": row.get("run_timestamp"),
                "stage_name": stage_name,
                "sec": metric.get("sec") if metric else None,
                "pct_end_to_end": metric.get("pct_end_to_end") if metric else None,
                "pct_profiled": metric.get("pct_profiled") if metric else None,
                "avg_sec_per_sample": avg_stage_metrics.get(stage_name),
                "log_path": row["log_path"],
            }
        )

    return row, task_rows, stage_rows


def collect_log_paths(paths):
    if not paths:
        paths = DEFAULT_LOG_DIRS
    log_paths = []
    for raw_path in paths:
        expanded = glob.glob(raw_path)
        candidates = expanded or [raw_path]
        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                log_paths.extend(str(p) for p in sorted(path.glob("*.log")))
            elif path.is_file() and path.suffix == ".log":
                log_paths.append(str(path))
    return sorted(dict.fromkeys(log_paths))


def latest_per_run(rows):
    latest = {}
    for row in rows:
        run_name = row.get("run_name") or row.get("log_file")
        ts = row.get("run_timestamp") or ""
        prev = latest.get(run_name)
        if prev is None or ts >= (prev.get("run_timestamp") or ""):
            latest[run_name] = row
    return list(latest.values())


def ordered_summary_columns(rows):
    preferred = [
        "experiment_type",
        "run_name",
        "run_timestamp",
        "completed",
        "accuracy_pct",
        "total_correct",
        "total_samples",
        "subset_pct",
        "sample_seed",
        "task_csv",
        "total_runtime_sec",
        "total_runtime_min",
        "total_runtime_hms",
        "profiled_samples",
        "profiled_stage_total_sec",
        "model_compute_sec",
        "profiled_stage_bottleneck",
        "profiled_stage_bottleneck_pct",
        "drop_method",
        "drop_mode",
        "drop_threshold",
        "hist_enabled",
        "hist_mode",
        "hist_bins",
        "hist_threshold",
        "hist_min_keep_frames",
        "hist_max_interval",
        "hist_video_frame_compression_ratio_pct",
        "hist_samples",
        "hist_raw_frames",
        "hist_kept_frames",
        "hist_avg_raw_frames_per_sample",
        "hist_avg_kept_frames_per_sample",
        "hist_frame_select_time_sec",
        "attention_layer",
        "top_k_ratio_keep",
        "attention_drop_ratio",
        "scorer_chunk_size",
        "token_total_drop_ratio_pct",
        "token_avg_drop_ratio_pct",
        "attention_impl",
        "temperature",
        "min_pixels",
        "max_pixels",
        "min_frames",
        "max_frames",
        "dataloader_batch_size",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "model_name",
        "ckpt_path",
        "result_dir",
        "video_dir",
        "output_jsonl",
        "error_jsonl",
        "drop_jsonl",
        "drop_pos_jsonl",
        "similarity_jsonl",
        "log_start_time",
        "log_end_time",
        "log_file",
        "log_dir",
        "log_path",
    ]
    stage_cols = []
    for stage_name in STAGE_NAMES:
        safe = clean_col(stage_name)
        stage_cols.extend(
            [
                f"stage_{safe}_sec",
                f"stage_{safe}_pct_end_to_end",
                f"stage_{safe}_pct_profiled",
                f"avg_{safe}_sec_per_sample",
            ]
        )
    task_cols = []
    for category in CATEGORY_NAMES:
        safe = clean_col(category)
        task_cols.extend([f"task_{safe}_correct", f"task_{safe}_total", f"task_{safe}_acc_pct"])

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    ordered = [col for col in [*preferred, *stage_cols, *task_cols] if col in all_keys]
    ordered.extend(sorted(all_keys - set(ordered)))
    return ordered


def write_excel(rows, task_rows, stage_rows, output_path):
    output_path = os.path.realpath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_columns = ordered_summary_columns(rows)
    summary_df = pd.DataFrame(rows, columns=summary_columns)
    if not summary_df.empty:
        sort_cols = [col for col in ["experiment_type", "hist_threshold", "attention_layer", "run_timestamp"] if col in summary_df.columns]
        summary_df = summary_df.sort_values(sort_cols, kind="stable", na_position="last")

    task_df = pd.DataFrame(task_rows)
    if not task_df.empty:
        task_df = task_df.sort_values(["experiment_type", "run_name", "task_type"], kind="stable")

    stage_df = pd.DataFrame(stage_rows)
    if not stage_df.empty:
        stage_df = stage_df.sort_values(["experiment_type", "run_name", "stage_name"], kind="stable")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_wide", index=False)
        task_df.to_excel(writer, sheet_name="task_type_metrics", index=False)
        stage_df.to_excel(writer, sheet_name="stage_breakdown", index=False)

        workbook = writer.book
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                width = min(max(len(header) + 2, 12), 48)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width

    return output_path, len(summary_df)


def main():
    parser = argparse.ArgumentParser(
        description="Collect StreamingBench sweep logs and export a consolidated Excel workbook."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Log files, log directories, or glob patterns. Defaults to hdiff and attention layer log directories.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output xlsx path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument(
        "--task-csv-contains",
        default=None,
        help="Keep only logs whose parsed Task csv contains this substring, e.g. streamingbench_10pct_seed42.",
    )
    parser.add_argument(
        "--latest-per-run",
        action="store_true",
        help="Keep only the newest log for each run_name after filtering.",
    )
    args = parser.parse_args()

    log_paths = collect_log_paths(args.paths)
    rows = []
    task_rows = []
    stage_rows = []
    for log_path in log_paths:
        row, tasks, stages = parse_log(log_path)
        if args.task_csv_contains and args.task_csv_contains not in str(row.get("task_csv") or ""):
            continue
        rows.append(row)
        task_rows.extend(tasks)
        stage_rows.extend(stages)

    if args.latest_per_run:
        kept = latest_per_run(rows)
        kept_log_paths = {row["log_path"] for row in kept}
        rows = kept
        task_rows = [row for row in task_rows if row["log_path"] in kept_log_paths]
        stage_rows = [row for row in stage_rows if row["log_path"] in kept_log_paths]

    if not rows:
        raise SystemExit("No matching log files found.")

    output_path, row_count = write_excel(rows, task_rows, stage_rows, args.output)
    print(f"Saved Excel: {output_path}")
    print(f"Parsed logs: {row_count}")


if __name__ == "__main__":
    main()
