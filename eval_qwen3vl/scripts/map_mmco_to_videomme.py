import argparse
import os
import os.path as osp
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


MMCO_PATTERN = re.compile(r"\[h264\s+@\s+(0x[0-9a-fA-F]+)\]\s+mmco:\s+unref short failure")
PROGRESS_PATTERN = re.compile(r"\|\s*(\d+)/(\d+)")


@dataclass
class MmcoEvent:
    line_no: int
    decoder_addr: str
    progress_idx: Optional[int]
    total: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map mmco decode warnings from VideoMME eval logs to parquet rows/videoID, "
            "and export a re-encode candidate list."
        )
    )
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="Path to eval log, e.g. eval_qwen3vl/result_videomme/eval_videomme.log",
    )
    parser.add_argument(
        "--task-parquet",
        type=str,
        default=None,
        help="Path to VideoMME parquet. If omitted, try parsing from log line 'Task parquet: ...'.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Video root dir. If omitted, try parsing from log line 'Video dir: ...'.",
    )
    parser.add_argument(
        "--index-offset",
        type=int,
        default=1,
        help=(
            "Offset from tqdm progress index to suspected failing sample index. "
            "Default=1 because warnings usually happen while processing the next sample after current progress update."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for exported files. Default: <log_dir>/mmco_mapping",
    )
    return parser.parse_args()


def _extract_path_from_log(lines: list[str], key: str) -> Optional[str]:
    for line in lines:
        if key in line:
            return line.split(key, 1)[1].strip()
    return None


def _pick_video_id_column(df: pd.DataFrame) -> str:
    candidates = ["videoID", "video_id", "video", "video_path", "url"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find a video id column. Available columns: {list(df.columns)}")


def _build_video_path(video_dir: Optional[str], video_id_value: object) -> Optional[str]:
    if video_dir is None:
        return None
    vid = str(video_id_value)
    if vid.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return osp.join(video_dir, vid)
    return osp.join(video_dir, f"{vid}.mp4")


def parse_mmco_events(log_lines: list[str]) -> list[MmcoEvent]:
    events: list[MmcoEvent] = []
    last_progress_idx: Optional[int] = None
    last_total: Optional[int] = None

    for i, line in enumerate(log_lines, start=1):
        p = PROGRESS_PATTERN.search(line)
        if p:
            last_progress_idx = int(p.group(1))
            last_total = int(p.group(2))

        mm = MMCO_PATTERN.search(line)
        if mm:
            events.append(
                MmcoEvent(
                    line_no=i,
                    decoder_addr=mm.group(1),
                    progress_idx=last_progress_idx,
                    total=last_total,
                )
            )
    return events


def main() -> None:
    args = parse_args()

    with open(args.log_path, "r", encoding="utf-8", errors="replace") as f:
        log_lines = f.readlines()

    task_parquet = args.task_parquet or _extract_path_from_log(log_lines, "Task parquet:")
    video_dir = args.video_dir or _extract_path_from_log(log_lines, "Video dir:")

    if not task_parquet:
        raise ValueError("--task-parquet is required when log does not contain 'Task parquet:'")
    if not osp.exists(task_parquet):
        raise FileNotFoundError(f"Task parquet not found: {task_parquet}")

    events = parse_mmco_events(log_lines)
    if not events:
        print("No mmco warnings found. Nothing to export.")
        return

    task_df = pd.read_parquet(task_parquet).reset_index(drop=True)
    video_col = _pick_video_id_column(task_df)
    total_rows = len(task_df)

    records = []
    for ev in events:
        mapped_idx = None
        if ev.progress_idx is not None:
            mapped_idx = ev.progress_idx + args.index_offset

        is_valid = mapped_idx is not None and 1 <= mapped_idx <= total_rows

        row_data = {}
        if is_valid:
            row = task_df.iloc[mapped_idx - 1]
            row_data["mapped_video_id"] = row[video_col]
            row_data["mapped_video_path"] = _build_video_path(video_dir, row[video_col])
            row_data["mapped_question_id"] = row["question_id"] if "question_id" in task_df.columns else None
            row_data["mapped_duration"] = row["duration"] if "duration" in task_df.columns else None
            row_data["mapped_domain"] = row["domain"] if "domain" in task_df.columns else None
            row_data["mapped_sub_category"] = row["sub_category"] if "sub_category" in task_df.columns else None
        else:
            row_data["mapped_video_id"] = None
            row_data["mapped_video_path"] = None
            row_data["mapped_question_id"] = None
            row_data["mapped_duration"] = None
            row_data["mapped_domain"] = None
            row_data["mapped_sub_category"] = None

        records.append(
            {
                "line_no": ev.line_no,
                "decoder_addr": ev.decoder_addr,
                "progress_idx": ev.progress_idx,
                "progress_total": ev.total,
                "index_offset": args.index_offset,
                "mapped_sample_idx": mapped_idx,
                "mapped_valid": bool(is_valid),
                **row_data,
            }
        )

    events_df = pd.DataFrame(records)
    valid_events_df = events_df[events_df["mapped_valid"]]

    samples_df = (
        valid_events_df.groupby(["mapped_sample_idx", "mapped_video_id", "mapped_video_path"], dropna=False)
        .agg(
            mmco_hits=("line_no", "count"),
            first_log_line=("line_no", "min"),
            last_log_line=("line_no", "max"),
            decoder_count=("decoder_addr", "nunique"),
            question_count=("mapped_question_id", "nunique"),
        )
        .reset_index()
        .sort_values(["mmco_hits", "first_log_line"], ascending=[False, True])
    )

    reencode_df = (
        samples_df.groupby(["mapped_video_id", "mapped_video_path"], dropna=False)
        .agg(
            total_mmco_hits=("mmco_hits", "sum"),
            affected_samples=("mapped_sample_idx", "nunique"),
            earliest_log_line=("first_log_line", "min"),
            latest_log_line=("last_log_line", "max"),
        )
        .reset_index()
        .sort_values(["total_mmco_hits", "affected_samples"], ascending=[False, False])
    )

    output_dir = args.output_dir or osp.join(osp.dirname(args.log_path), "mmco_mapping")
    os.makedirs(output_dir, exist_ok=True)

    log_stem = osp.splitext(osp.basename(args.log_path))[0]
    events_path = osp.join(output_dir, f"{log_stem}_mmco_events.csv")
    samples_path = osp.join(output_dir, f"{log_stem}_mmco_samples.csv")
    reencode_csv_path = osp.join(output_dir, f"{log_stem}_reencode_manifest.csv")
    reencode_txt_path = osp.join(output_dir, f"{log_stem}_reencode_videoids.txt")

    events_df.to_csv(events_path, index=False)
    samples_df.to_csv(samples_path, index=False)
    reencode_df.to_csv(reencode_csv_path, index=False)

    with open(reencode_txt_path, "w", encoding="utf-8") as f:
        for vid in reencode_df["mapped_video_id"].dropna().astype(str).unique().tolist():
            f.write(f"{vid}\n")

    print(f"Log: {args.log_path}")
    print(f"Task parquet: {task_parquet}")
    print(f"Video dir: {video_dir if video_dir else '<not provided>'}")
    print(f"Rows in parquet: {total_rows}")
    print(f"MMCO events: {len(events_df)}")
    print(f"Mapped events (valid index): {len(valid_events_df)}")
    print(f"Affected samples: {len(samples_df)}")
    print(f"Affected unique video IDs: {len(reencode_df)}")
    print(f"Output directory: {output_dir}")
    print(f"- events: {events_path}")
    print(f"- samples: {samples_path}")
    print(f"- reencode manifest: {reencode_csv_path}")
    print(f"- reencode video list: {reencode_txt_path}")


if __name__ == "__main__":
    main()
