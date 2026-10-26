import argparse
import inspect
import json
import os
import os.path as osp
import sys
from datetime import datetime

import ffmpeg
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from matplotlib import cm
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info, fetch_video
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.append(osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl import Qwen3VLForConditionalGenerationDTD

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016

PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""


def time_to_seconds(time_str):
    if len(time_str) == 5:
        t = datetime.strptime(time_str, '%M:%S')
    else:
        t = datetime.strptime(time_str, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


def split_video(video_file, start_time, end_time):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        return output_file
    (
        ffmpeg.input(video_file, ss=int(start_time))
        .output(output_file, t=int(end_time) - int(start_time),
                vcodec='libx264', acodec='aac')
        .global_args('-nostdin')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return output_file


def process_vision_info_with_video_kwargs(messages):
    sig = inspect.signature(process_vision_info)
    if "return_video_kwargs" in sig.parameters:
        return process_vision_info(messages, return_video_kwargs=True)
    img, vid = process_vision_info(messages)
    return img, vid, {"do_sample_frames": False}


class AttentionCollector:
    """
    Hook-based collector for T2V (Text-to-Video) attention:
    - Last LM self-attention layer only (overwrite semantics).
    - Mean over all attention heads.
    - For each visual patch: max attention score across text query tokens.
    """

    def __init__(self):
        self._hooks = []
        self.reset()

    def reset(self):
        self.text_indices = None
        self.frame_info = []
        self.attn_accum = {}

    def _parse_from_hook(self, visual_pos_masks, position_ids):
        vis_mask = visual_pos_masks[0]
        pos = position_ids[:, 0]
        vis_idx = vis_mask.nonzero(as_tuple=True)[0]
        self.text_indices = (~vis_mask).nonzero(as_tuple=True)[0].cpu()
        if vis_idx.numel() == 0:
            return
        t_pos = pos[0, vis_idx]
        for t_val in t_pos.unique(sorted=True):
            sel = (t_pos == t_val).nonzero(as_tuple=True)[0]
            fvi = vis_idx[sel]
            h_ids = pos[1, fvi]
            w_ids = pos[2, fvi]
            h0, w0 = h_ids.min().item(), w_ids.min().item()
            rh = (h_ids - h0).tolist()
            rw = (w_ids - w0).tolist()
            self.frame_info.append({
                'vis_indices': fvi.cpu(),
                'rel_h': rh, 'rel_w': rw,
                'grid_h': max(rh) + 1, 'grid_w': max(rw) + 1,
            })

    def parse_from_input_ids(self, input_ids, video_grid_thw, config):
        ids = input_ids[0]
        vis_s = config.vision_start_token_id
        vis_e = config.vision_end_token_id
        vid_id = config.video_token_id
        starts = (ids == vis_s).nonzero(as_tuple=True)[0]
        ends = (ids == vis_e).nonzero(as_tuple=True)[0]
        segs = []
        for s, e in zip(starts, ends):
            if ids[s + 1] == vid_id:
                segs.append((s.item() + 1, e.item()))
        all_vis = set()
        for s, e in segs:
            all_vis.update(range(s, e))
        self.text_indices = torch.tensor(
            sorted(set(range(len(ids))) - all_vis), dtype=torch.long)
        T, H, W = video_grid_thw[0].tolist()
        tpf = H * W
        for seg_s, seg_e in segs:
            n_f = (seg_e - seg_s) // tpf
            for f in range(n_f):
                base = seg_s + f * tpf
                self.frame_info.append({
                    'vis_indices': torch.arange(base, base + tpf),
                    'rel_h': [i // W for i in range(tpf)],
                    'rel_w': [i % W for i in range(tpf)],
                    'grid_h': H, 'grid_w': W,
                })

    def _lm_pre_hook(self, module, args, kwargs):
        vpm = kwargs.get('visual_pos_masks')
        pid = kwargs.get('position_ids')
        if vpm is not None and pid is not None:
            self.frame_info = []
            self._parse_from_hook(vpm, pid)

    def _attn_post_hook(self, module, input, output):
        if not self.frame_info or self.text_indices is None:
            return output
        aw = output[1]
        if aw is None:
            return output
        attn = aw[0].float().mean(0)  # mean over heads -> (seq, seq)
        dev = attn.device
        ti = self.text_indices.to(dev)
        for k, info in enumerate(self.frame_info):
            vi = info['vis_indices'].to(dev)
            t2v = attn[ti][:, vi]  # (num_text, num_patches)
            patch_weights = t2v.max(dim=0).values.detach().cpu()
            self.attn_accum[k] = {'patch_weights': patch_weights}
        out = list(output)
        out[1] = None
        return tuple(out)

    def register(self, model):
        lm = model.model.language_model
        self._hooks.append(
            lm.register_forward_pre_hook(self._lm_pre_hook, with_kwargs=True))
        for name, mod in model.named_modules():
            if name.endswith('.self_attn'):
                self._hooks.append(mod.register_forward_hook(self._attn_post_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_maps(self):
        results = []
        for k, info in enumerate(self.frame_info):
            gh, gw = info['grid_h'], info['grid_w']
            if k not in self.attn_accum:
                results.append({
                    'grid_h': gh, 'grid_w': gw,
                    'heatmap': None, 'patch_weights': [], 'frame_weight': 0.0,
                })
                continue
            pw = self.attn_accum[k]['patch_weights']
            heatmap = [[None] * gw for _ in range(gh)]
            patch_list = []
            for i, (h, w) in enumerate(zip(info['rel_h'], info['rel_w'])):
                val = pw[i].item()
                heatmap[h][w] = val
                patch_list.append(val)
            frame_weight = float(pw.mean().item())
            results.append({
                'grid_h': gh, 'grid_w': gw,
                'heatmap': heatmap,
                'patch_weights': patch_list,
                'frame_weight': frame_weight,
            })
        return results


def _make_colorbar(height, vmin, vmax, cmap_name='jet', bar_w=20, total_w=80):
    cmap_fn = cm.get_cmap(cmap_name)
    bar = np.full((height, total_w, 3), 40, dtype=np.uint8)
    for y in range(height):
        t = 1.0 - y / max(height - 1, 1)
        bar[y, 8:8 + bar_w] = (np.array(cmap_fn(t)[:3]) * 255).astype(np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    for yy, txt in [(4, f"{vmax:.5f}"),
                     (height // 2 - 6, f"{(vmin + vmax) / 2:.5f}"),
                     (height - 14, f"{vmin:.5f}")]:
        draw.text((8 + bar_w + 4, yy), txt, fill=(220, 220, 220))
    return np.array(img)


def overlay_heatmap(frame, score_map, cmap_name='jet', alpha=0.5):
    if score_map is None:
        return frame
    gh = len(score_map)
    gw = len(score_map[0]) if gh else 0
    if not gh or not gw:
        return frame
    arr = np.zeros((gh, gw), dtype=np.float32)
    valid = np.zeros((gh, gw), dtype=bool)
    for i in range(gh):
        for j in range(gw):
            v = score_map[i][j]
            if v is not None:
                arr[i, j] = v
                valid[i, j] = True
    if not valid.any():
        return frame
    vmin, vmax = float(arr[valid].min()), float(arr[valid].max())
    span = max(vmax - vmin, 1e-8)
    norm = np.clip((arr - vmin) / span, 0, 1)
    fh, fw = frame.shape[:2]
    up = Fnn.interpolate(
        torch.from_numpy(norm)[None, None].float(),
        size=(fh, fw), mode='bilinear', align_corners=False,
    ).squeeze().numpy()
    heat = (np.array(cm.get_cmap(cmap_name)(up))[:, :, :3] * 255).astype(np.uint8)
    blend = (frame.astype(np.float32) * (1 - alpha) +
             heat.astype(np.float32) * alpha).astype(np.uint8)
    cbar = _make_colorbar(fh, vmin, vmax, cmap_name)
    return np.concatenate([blend, cbar], axis=1)


def make_frame_grid(pil_images, max_cols=4, bg=(0, 0, 0)):
    n = len(pil_images)
    if n == 0:
        return None
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    cell_w = max(im.width for im in pil_images)
    cell_h = max(im.height for im in pil_images)
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h), bg)
    for i, im in enumerate(pil_images):
        r, c = divmod(i, cols)
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--question_id', type=str, required=True)
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--task_csv', type=str, required=True)
    p.add_argument('--video_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--drop_method', type=str, default='none')
    p.add_argument('--drop_threshold', type=float, default=0.0)
    p.add_argument('--drop_relative', action='store_true')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--cmap', type=str, default='jet')
    p.add_argument('--min_pixels', type=int, default=MIN_PIXELS)
    p.add_argument('--max_pixels', type=int, default=MAX_PIXELS)
    p.add_argument('--min_frames', type=int, default=MIN_FRAMES)
    p.add_argument('--max_frames', type=int, default=MAX_FRAMES)
    p.add_argument('--fps', type=float, default=1.0,
                    help='每秒采样帧数，传给 qwen_vl_utils 与 processor')
    p.add_argument('--grid_cols', type=int, default=4,
                    help='拼接图每行最多帧数，0 表示不输出拼接图')
    p.add_argument('--output_jsonl', type=str, default=None,
                    help='推理结果 JSONL 输出路径，不指定则不保存')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    task_df = pd.read_csv(args.task_csv)
    row = task_df[task_df['question_id'] == args.question_id].iloc[0]
    sample_id = str(row.question_id).split('_')[-2]
    video_path = osp.join(args.video_dir, f"sample_{sample_id}", "video.mp4")
    ts = time_to_seconds(row.time_stamp)
    video_path = split_video(video_path, 0, ts)
    fps = float(args.fps)

    torch.manual_seed(1234)
    print(f"Loading model from {args.ckpt_path} (eager attention) ...")
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        args.ckpt_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto")
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
             "max_frames": args.max_frames, "min_frames": args.min_frames,
             "fps": fps},
            {"type": "text",
             "text": PROMPT.format(row.question, '\n'.join(eval(row.options)))},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info_with_video_kwargs(
        messages)
    video_kwargs = dict(video_kwargs)
    video_kwargs.setdefault("do_sample_frames", False)
    fv = video_kwargs.get("fps")
    if isinstance(fv, (list, tuple)) and fv:
        video_kwargs["fps"] = fv[0]
    elif fv is None:
        video_kwargs["fps"] = fps
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to('cuda')

    collector = AttentionCollector()
    collector.parse_from_input_ids(
        inputs['input_ids'], inputs.get('video_grid_thw'), model.config)
    collector.register(model)

    dm = args.drop_method if args.drop_method.lower() != 'none' else None
    print("Running forward pass ...")
    with torch.no_grad():
        model(**inputs, output_attentions=True,
              drop_method=dm, drop_threshold=args.drop_threshold,
              drop_absolute=not args.drop_relative)

    collector.remove()
    maps = collector.get_maps()
    print(f"Collected T2V attention (last layer, head-mean, text-max) "
          f"for {len(maps)} frames")

    print("Running generate for inference result ...")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=32)
    generated_text = processor.decode(
        gen_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    frames_np = fetch_video({
        "type": "video", "video": video_path, "fps": fps,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "min_frames": args.min_frames, "max_frames": args.max_frames,
    }).clamp(0, 255).to(torch.uint8).cpu().numpy()

    n = min(len(maps), frames_np.shape[0])
    raw_list, t2v_list = [], []
    json_frames = []
    for idx in range(n):
        frame = np.transpose(frames_np[idx], (1, 2, 0))
        m = maps[idx]
        im_raw = Image.fromarray(frame)
        im_raw.save(osp.join(args.output_dir, f"{idx:03d}_raw.png"))
        im_t2v = Image.fromarray(
            overlay_heatmap(frame, m['heatmap'], args.cmap, args.alpha))
        im_t2v.save(osp.join(args.output_dir, f"{idx:03d}_T2V.png"))
        if args.grid_cols > 0:
            raw_list.append(im_raw)
            t2v_list.append(im_t2v)
        json_frames.append({
            'frame_index': idx,
            'num_patches': len(m['patch_weights']),
            'patch_weights': m['patch_weights'],
            'frame_weight': m['frame_weight'],
        })

    json_path = osp.join(args.output_dir, 't2v_attention.json')
    with open(json_path, 'w') as f:
        json.dump(json_frames, f, indent=2)
    print(f"Saved T2V attention JSON to {json_path}")
    print(f"Saved {n} frames (raw + T2V heatmap) to {args.output_dir}")

    if args.output_jsonl:
        options_list = eval(row.options) if isinstance(row.options, str) else row.options
        result_record = {
            "question_id": args.question_id,
            "question": row.question,
            "answer": str(row.answer),
            "response": generated_text,
            "options": options_list,
            "ckpt_path": args.ckpt_path,
        }
        os.makedirs(osp.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
        print(f"Saved inference result to {args.output_jsonl}")

    if args.grid_cols > 0 and n > 0:
        for name, lst in [('RAW', raw_list), ('T2V', t2v_list)]:
            g = make_frame_grid(lst, max_cols=args.grid_cols)
            if g is not None:
                g.save(osp.join(args.output_dir, f"all_frames_{name}.png"))
        print(f"Saved all_frames_RAW/T2V.png ({args.grid_cols} cols) "
              f"to {args.output_dir}")


if __name__ == '__main__':
    main()
