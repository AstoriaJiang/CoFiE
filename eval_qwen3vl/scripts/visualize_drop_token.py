import argparse
import json
import os
import os.path as osp
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import fetch_video


MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016


def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, "%M:%S")
    else:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


def split_video(video_file, start_time, end_time):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        return output_file
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-ss", str(int(start_time)),
        "-i", video_file,
        "-t", str(int(end_time) - int(start_time)),
        "-vcodec", "libx264", "-acodec", "aac",
        output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file


def resolve_video_path(args):
    if args.video_path:
        return args.video_path
    if not args.task_csv or not args.video_dir or not args.question_id:
        raise ValueError("video_path 或 task_csv + video_dir + question_id 必须提供一组。")
    task_df = pd.read_csv(args.task_csv)
    task_df = task_df[task_df["question_id"] == args.question_id]
    if task_df.empty:
        raise ValueError(f"未找到 question_id={args.question_id}")
    row = task_df.iloc[0]
    video_path = osp.join(args.video_dir, f"sample_{str(row.question_id).split('_')[-2]}", "video.mp4")
    time_stamp_sec = time_to_seconds(row.time_stamp)
    return split_video(video_path, 0, time_stamp_sec)


def infer_fps_from_video_path(video_path):
    file_name = osp.splitext(osp.basename(video_path))[0]
    parts = file_name.split("_")
    if len(parts) >= 3 and parts[0] == "video":
        try:
            end_time = int(parts[-1])
        except ValueError:
            return 1
        if 300 < end_time <= 600:
            return 0.5
        if end_time > 600:
            return 0.2
    return 1


def load_video_frames(video_path, fps, min_pixels, max_pixels, min_frames, max_frames):
    video_tensor = fetch_video(
        {
            "type": "video",
            "video": video_path,
            "fps": fps,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "min_frames": min_frames,
            "max_frames": max_frames,
        }
    )
    if not hasattr(video_tensor, "shape"):
        raise ValueError("当前视频读取结果不是 tensor")
    return video_tensor.clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()


def load_record(jsonl_path, sample_index):
    with open(jsonl_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{jsonl_path} 为空")
    if sample_index >= len(lines):
        raise ValueError(f"sample_index={sample_index} 超出范围，当前只有 {len(lines)} 条记录")
    record = json.loads(lines[sample_index])
    if isinstance(record, list):
        if not record:
            raise ValueError(f"{jsonl_path} 第 {sample_index} 条记录为空列表")
        record = record[0]
    return record


def build_drop_mask_from_dp(drop_pos, grid_h, grid_w):
    mask = np.ones((grid_h, grid_w), dtype=np.float32)
    for h, w in drop_pos:
        if 0 <= h < grid_h and 0 <= w < grid_w:
            mask[h][w] = 0.0
    return mask


def upsample_mask(mask, frame_h, frame_w):
    grid_h, grid_w = mask.shape
    if grid_h == 0 or grid_w == 0:
        return np.ones((frame_h, frame_w), dtype=np.float32)
    out = np.ones((frame_h, frame_w), dtype=np.float32)
    cell_h = frame_h // grid_h
    cell_w = frame_w // grid_w
    for i in range(grid_h):
        for j in range(grid_w):
            h_s = i * cell_h
            h_e = frame_h if i == grid_h - 1 else (i + 1) * cell_h
            w_s = j * cell_w
            w_e = frame_w if j == grid_w - 1 else (j + 1) * cell_w
            out[h_s:h_e, w_s:w_e] = mask[i, j]
    return out


def apply_drop_overlay(frame, mask_full, dim_factor):
    frame_f = frame.astype(np.float32)
    factor = np.where(mask_full > 0.5, 1.0, dim_factor).astype(np.float32)
    result = frame_f * factor[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def make_frame_grid(pil_images, max_cols=4, bg=(0, 0, 0)):
    n = len(pil_images)
    if n == 0:
        return None
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    cell_w = max(im.width for im in pil_images)
    cell_h = max(im.height for im in pil_images)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), bg)
    for i, im in enumerate(pil_images):
        r, c = divmod(i, cols)
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=["feature", "query_feature", "dynamic_feature", "dino_feature"],
        required=True,
    )
    parser.add_argument("--dp_jsonl", type=str, required=True, help="dp_save_path jsonl from inference")
    parser.add_argument("--sim_jsonl", type=str, required=True, help="sim_save_path jsonl (for grid dimensions)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dim_factor", type=float, default=0.3)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--task_csv", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--grid_cols", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_path = resolve_video_path(args)
    fps = infer_fps_from_video_path(video_path)

    sim_record = load_record(args.sim_jsonl, args.sample_index)
    dp_record = load_record(args.dp_jsonl, args.sample_index)

    frames = load_video_frames(
        video_path,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    sim_frames = sim_record.get("frames", [])
    frame_count = min(len(sim_frames), int(frames.shape[0]))
    total_kept, total_all = 0, 0
    raw_list, drop_list = [], []

    for idx in range(frame_count):
        frame = np.transpose(frames[idx], (1, 2, 0))
        frame_info = sim_frames[idx]
        grid_h = frame_info["grid_h"]
        grid_w = frame_info["grid_w"]

        frame_key = str(idx)
        drop_pos = dp_record.get(frame_key, [])
        grid_mask = build_drop_mask_from_dp(drop_pos, grid_h, grid_w)

        kept = int(grid_mask.sum())
        total = grid_mask.size
        total_kept += kept
        total_all += total
        drop_pct = (1 - kept / total) * 100 if total > 0 else 0
        print(f"Frame {idx:3d}: grid {grid_h}x{grid_w}, "
              f"kept {kept}/{total}, dropped {drop_pct:.1f}%")

        mask_full = upsample_mask(grid_mask, frame.shape[0], frame.shape[1])
        overlay = apply_drop_overlay(frame, mask_full, args.dim_factor)

        im_raw = Image.fromarray(frame)
        im_raw.save(osp.join(args.output_dir, f"{idx:03d}_raw.png"))
        im_drop = Image.fromarray(overlay)
        im_drop.save(osp.join(args.output_dir, f"{idx:03d}_drop_mask.png"))
        if args.grid_cols > 0:
            raw_list.append(im_raw)
            drop_list.append(im_drop)

    if total_all > 0:
        print(f"\nTotal: kept {total_kept}/{total_all}, "
              f"dropped {(1 - total_kept / total_all) * 100:.1f}%")
    if args.grid_cols > 0 and frame_count > 0:
        g_raw = make_frame_grid(raw_list, max_cols=args.grid_cols)
        g_drop = make_frame_grid(drop_list, max_cols=args.grid_cols)
        if g_raw is not None:
            g_raw.save(osp.join(args.output_dir, "all_frames_raw.png"))
        if g_drop is not None:
            g_drop.save(osp.join(args.output_dir, "all_frames_drop_mask.png"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
