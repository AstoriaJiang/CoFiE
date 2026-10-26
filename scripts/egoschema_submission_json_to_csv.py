#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        prog="ego_json2csv",
        description="EgoSchema inference JSON -> CSV",
    )
    p.add_argument("json_path", type=Path)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="默认: 与 JSON 同目录同名 .csv",
    )
    args = p.parse_args()
    out = args.output
    if out is None:
        out = args.json_path.with_suffix(".csv")
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("JSON 根节点必须是对象 {q_uid: answer}")
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_uid", "answer"])
        writer.writerows(data.items())
    print(out)


if __name__ == "__main__":
    main()
