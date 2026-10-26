#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# python scripts/rank_frame_weight.py --input_json <json路径>

def main():
    parser = argparse.ArgumentParser(
        description="Sort frames by frame_weight (ascending) and write frame indices."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path(
            "/root/autodl-tmp/CoFiE/eval_qwen3vl/vis_attention_map/sample_4_2/t2v_attention.json"
        ),
        help="Path to t2v_attention.json",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=None,
        help="Output path for frame rank list (default: same dir as input_json)",
    )
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ranked = sorted(data, key=lambda x: float(x.get("frame_weight", 0.0)))
    frame_indices = [str(int(item["frame_index"])) for item in ranked]

    output_txt = args.output_txt or (args.input_json.parent / "frame_rank.txt")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(frame_indices))
        f.write("\n")

    print(f"Wrote {len(frame_indices)} frame indices to {output_txt}")


if __name__ == "__main__":
    main()
