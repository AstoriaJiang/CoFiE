#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <path_to.log> [path_to_2.log ...]" >&2
  exit 1
fi
for log_path in "$@"; do
  if [[ ! -f "$log_path" ]]; then
    echo "File not found: $log_path" >&2
    exit 1
  fi
done
PY="${REPO_ROOT}/eval_qwen3vl/.venv/bin/python"
[[ ! -x "$PY" ]] && PY="python3"
"$PY" - "$@" <<'PY'
import os, re, sys
import pandas as pd

STAGE_COLUMNS = [
    "Frame Extraction",
    "Vision Encoder",
    "Token Drop",
    "LLM Prefill",
    "LLM Decode",
    "Other/uncaptured",
]

INFERENCE_STAGE_COLUMNS = [
    "Vision Encoder",
    "Token Drop",
    "LLM Prefill",
    "LLM Decode",
]

BASE_COLS = [
    "Total Time",
    "Inference Time",
    *STAGE_COLUMNS,
]

METRIC_COLS = [
    ("OCR", "OCR"),
    ("ACR", "ACR"),
    ("ATR", "ATR"),
    ("STU", "STU"),
    ("FPD", "FPD"),
    ("OJR", "OJR"),
    ("Avg.", "Realtime Avg."),
    ("EPM", "EPM"),
    ("ASI", "ASI"),
    ("HLD", "HLD"),
    ("Avg.", "Backward Avg."),
    ("REC", "REC"),
    ("SSR", "SSR"),
    ("CRR", "CRR"),
    ("Avg.", "Forward Avg."),
    ("ALL avg.", "Total Avg."),
]

def parse_duration_seconds(value):
    value = value.strip()
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
    return float(value)

def parse_log(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    row = {c: None for c in BASE_COLS}
    row.update({key: None for _, key in METRIC_COLS})
    row["Total Avg."] = None
    for line in text.splitlines():
        m = re.search(r"\bINFO \| (.+)$", line)
        if not m:
            continue
        msg = m.group(1).strip()
        tm = re.match(r"Task: (\w+), Acc: ([\d.]+)", msg)
        if tm:
            row[tm.group(1)] = float(tm.group(2))
            continue
        gm = re.match(r"(Realtime|Backward|Forward) Avg\.\s*:\s*([\d.]+)", msg)
        if gm:
            row[f"{gm.group(1)} Avg."] = float(gm.group(2))
            continue
        tm2 = re.match(r"Total Avg\.\s*:\s*([\d.]+)", msg)
        if tm2:
            row["Total Avg."] = float(tm2.group(1))
            continue
        for stage_name in STAGE_COLUMNS:
            sm = re.match(rf"-\s+{re.escape(stage_name)}:\s+([\d.]+)s", msg)
            if sm:
                row[stage_name] = float(sm.group(1))
                break
        else:
            sm = None
        if sm:
            continue
        im = re.match(r"Inference cost time:\s*(.+)$", msg)
        if im:
            row["Total Time"] = parse_duration_seconds(im.group(1))
    inference_values = [row[k] for k in INFERENCE_STAGE_COLUMNS if row[k] is not None]
    if len(inference_values) == len(INFERENCE_STAGE_COLUMNS):
        row["Inference Time"] = sum(inference_values)
    base = os.path.splitext(os.path.basename(log_path))[0]
    out = os.path.join(os.path.dirname(os.path.realpath(log_path)), f"{base}.xlsx")
    columns = [*BASE_COLS, *[label for label, _ in METRIC_COLS]]
    values = [*[row.get(c) for c in BASE_COLS], *[row.get(key) for _, key in METRIC_COLS]]
    pd.DataFrame([values], columns=columns).to_excel(out, index=False, engine="openpyxl")
    print(out)

def main():
    for log_path in sys.argv[1:]:
        parse_log(log_path)

if __name__ == "__main__":
    main()
PY
