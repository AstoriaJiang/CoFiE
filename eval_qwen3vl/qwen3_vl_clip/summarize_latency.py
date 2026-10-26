#!/usr/bin/env python3
"""Summarize JSONL emitted by CLIPPatchFrameSelector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(stats_jsonl: Path) -> list[dict]:
    """Load selector records emitted as one JSON object per video."""
    records = []
    with stats_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_records(records: list[dict]) -> dict:
    """Return paper-friendly aggregate latency and frame-budget statistics."""
    if not records:
        raise ValueError("No CLIP selector records to summarize")

    latencies = np.asarray([row["total_seconds"] for row in records], dtype=np.float64)
    frames_in = sum(int(row["frames_in"]) for row in records)
    frames_out = sum(int(row["frames_out"]) for row in records)
    return {
        "videos": len(records),
        "frames_in": frames_in,
        "frames_out": frames_out,
        "keep_ratio": frames_out / frames_in if frames_in else 1.0,
        "clip_filter_total_seconds": float(latencies.sum()),
        "clip_filter_mean_seconds_per_video": float(latencies.mean()),
        "clip_filter_p50_seconds_per_video": float(np.percentile(latencies, 50)),
        "clip_filter_p95_seconds_per_video": float(np.percentile(latencies, 95)),
        "clip_preprocess_total_seconds": float(sum(row["preprocess_seconds"] for row in records)),
        "clip_forward_total_seconds": float(sum(row["forward_seconds"] for row in records)),
        "selection_total_seconds": float(sum(row["selection_seconds"] for row in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = load_records(args.stats_jsonl)
    if not records:
        raise SystemExit(f"No records found in {args.stats_jsonl}")

    summary = summarize_records(records)
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
