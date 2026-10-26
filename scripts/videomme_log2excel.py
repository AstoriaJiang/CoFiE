import argparse
import os
import re

import pandas as pd


STAGE_COLUMNS = [
    "Frame Extraction",
    "Vision Encoder",
    "Token Drop",
    "LLM Prefill",
    "LLM Decode",
    # "Text Detokenize (processor.batch_decode)",
    "Other/uncaptured",
]

INFERENCE_TIME_STAGES = [
    "Vision Encoder",
    "Token Drop",
    "LLM Prefill",
    "LLM Decode",
]

COLUMNS = [
    "Total Time",
    "Inference Time",
    *STAGE_COLUMNS,
    "Accuracy",
    "long",
    "medium",
    "short",
]


def parse_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    row = {key: None for key in COLUMNS}
    stage_seconds = {}

    match = re.search(r"Inference cost time:\s+(\d+)h\s+(\d+)m\s+(\d+)s", text)
    if match:
        hours, minutes, seconds = (int(v) for v in match.groups())
        if hours:
            row["Total Time"] = f"{hours}h{minutes}m{seconds}s"
        else:
            row["Total Time"] = f"{minutes}m{seconds}s"

    for duration in ("long", "medium", "short"):
        match = re.search(rf"Duration {duration}:\s+\d+/\d+\s+=\s+([\d.]+)%", text)
        if match:
            row[duration] = float(match.group(1))

    match = re.search(r"Accuracy:\s+([\d.]+)%", text)
    if match:
        row["Accuracy"] = float(match.group(1))

    stage_pattern = re.compile(
        r"-\s+(.+?):\s+([\d.]+)s\s+\|\s+([\d.]+)%\s+of end-to-end"
    )
    for stage_name, seconds, percent in stage_pattern.findall(text):
        if stage_name in STAGE_COLUMNS:
            row[stage_name] = f"{seconds}s({percent}%)"
            stage_seconds[stage_name] = float(seconds)

    inf = sum(stage_seconds.get(s, 0.0) for s in INFERENCE_TIME_STAGES)
    if any(s in stage_seconds for s in INFERENCE_TIME_STAGES):
        row["Inference Time"] = round(inf, 4)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", type=str, nargs="+")
    args = parser.parse_args()

    for log_path in args.log_paths:
        base = os.path.splitext(os.path.basename(log_path))[0]
        output_xlsx = os.path.join(os.path.dirname(os.path.realpath(log_path)), f"{base}.xlsx")

        row = parse_log(log_path)
        pd.DataFrame([row], columns=COLUMNS).to_excel(output_xlsx, index=False)
        print(output_xlsx)


if __name__ == "__main__":
    main()
