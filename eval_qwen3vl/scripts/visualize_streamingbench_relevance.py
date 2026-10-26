import argparse
import json
import os
import os.path as osp
from datetime import datetime

import ffmpeg
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
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
    (
        ffmpeg
        .input(video_file, ss=int(start_time))
        .output(output_file, t=(int(end_time) - int(start_time)), vcodec="libx264", acodec="aac")
        .global_args("-nostdin")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
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


def load_similarity_record(sim_jsonl, sample_index):
    with open(sim_jsonl, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{sim_jsonl} 为空")
    if sample_index >= len(lines):
        raise ValueError(f"sample_index={sample_index} 超出范围，当前只有 {len(lines)} 条记录")
    record = json.loads(lines[sample_index])
    if isinstance(record, list):
        if not record:
            raise ValueError(f"{sim_jsonl} 第 {sample_index} 条记录为空列表")
        record = record[0]
    return record


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


def _make_colorbar(height, vmin, vmax, bar_w=20, total_w=80):
    from PIL import ImageDraw
    cmap = cm.get_cmap("RdYlGn_r")
    bar = np.zeros((height, total_w, 3), dtype=np.uint8)
    bar[:] = [40, 40, 40]
    for y in range(height):
        t = 1.0 - y / max(height - 1, 1)
        rgb = np.array(cmap(t)[:3]) * 255.0
        bar[y, 8:8 + bar_w] = rgb.astype(np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    labels = [
        (4,          f"{vmax:.3f}"),
        (height // 2 - 6, f"{(vmin + vmax) / 2:.3f}"),
        (height - 14, f"{vmin:.3f}"),
    ]
    for y, text in labels:
        draw.text((8 + bar_w + 4, y), text, fill=(220, 220, 220))
    return np.array(img)


def overlay_relevance_circles(frame, score_map):
    if score_map is None:
        return frame
    grid_h = len(score_map)
    grid_w = len(score_map[0]) if grid_h > 0 else 0
    if grid_h == 0 or grid_w == 0:
        return frame

    flat = [float(v) for row in score_map for v in row if v is not None]
    if not flat:
        return frame
    vmin, vmax = min(flat), max(flat)
    span = max(vmax - vmin, 1e-6)

    frame_h, frame_w = frame.shape[:2]
    cell_h = frame_h // grid_h
    cell_w = frame_w // grid_w
    radius = max(1, int(min(cell_h, cell_w) * 0.4))
    cmap = cm.get_cmap("RdYlGn_r")

    overlay = frame.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            value = score_map[i][j]
            if value is None:
                continue
            t = float(np.clip((float(value) - vmin) / span, 0.0, 1.0))
            color = (np.array(cmap(t)[:3]) * 255.0).astype(np.uint8)

            h_start = i * cell_h
            h_end = frame_h if i == grid_h - 1 else (i + 1) * cell_h
            w_start = j * cell_w
            w_end = frame_w if j == grid_w - 1 else (j + 1) * cell_w

            cy = h_start + (h_end - h_start) // 2
            cx = w_start + (w_end - w_start) // 2

            yy, xx = np.ogrid[h_start:h_end, w_start:w_end]
            local_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            overlay[h_start:h_end, w_start:w_end][local_mask] = color

    colorbar = _make_colorbar(frame_h, vmin, vmax)
    return np.concatenate([overlay, colorbar], axis=1)


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
    parser.add_argument("--sim_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--task_csv", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--score_key", type=str, default="query_relevance_map")
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--grid_cols", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_path = resolve_video_path(args)
    fps = infer_fps_from_video_path(video_path)
    record = load_similarity_record(args.sim_jsonl, args.sample_index)
    frames = load_video_frames(
        video_path,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    frame_count = min(len(record.get("frames", [])), int(frames.shape[0]))
    raw_list, ovl_list = [], []
    for idx in range(frame_count):
        frame = np.transpose(frames[idx], (1, 2, 0))
        frame_info = record["frames"][idx]
        score_map = frame_info.get(args.score_key)
        overlay = overlay_relevance_circles(frame=frame, score_map=score_map)
        im_raw = Image.fromarray(frame)
        im_raw.save(osp.join(args.output_dir, f"{idx:03d}_raw.png"))
        im_ovl = Image.fromarray(overlay)
        im_ovl.save(osp.join(args.output_dir, f"{idx:03d}_overlay.png"))
        if args.grid_cols > 0:
            raw_list.append(im_raw)
            ovl_list.append(im_ovl)
    if args.grid_cols > 0 and frame_count > 0:
        gr = make_frame_grid(raw_list, max_cols=args.grid_cols)
        go = make_frame_grid(ovl_list, max_cols=args.grid_cols)
        if gr is not None:
            gr.save(osp.join(args.output_dir, "all_frames_raw.png"))
        if go is not None:
            go.save(osp.join(args.output_dir, "all_frames_overlay.png"))


if __name__ == "__main__":
    main()
