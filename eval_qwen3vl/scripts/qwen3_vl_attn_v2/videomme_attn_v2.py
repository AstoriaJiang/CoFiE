"""VideoMME evaluation with chunked-scorer attention-based whole-frame dropping (v2)."""

import argparse
import gc
import json
import logging
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_attn_v2 import Qwen3VLForConditionalGenerationDTD

RUN_NAME = "attn_v2_50"
CKPT_PATH = "ckpt/Qwen3-VL-4B-Instruct"
TASK_PARQUET = "./datasets/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "./datasets/Video-MME/data/"
RESULT_DIR = "eval_qwen3vl/result_videomme"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 10000
MIN_FRAMES = 4
FPS = 1

prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def extract_characters_regex(s):
    s = s.strip()
    for prefix in [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    m = re.search(r"[ABCD]", s)
    return m[0] if m else ""


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "CUDA out of memory" in str(exc)
    return False


def run_single_sample(model, processor, video_path, question, options, args, max_frames):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
             "max_frames": max_frames, "min_frames": args.min_frames,
             "fps": args.fps},
            {"type": "text",
             "text": prompt.format(question, '\n'.join(options.tolist()))},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages)
    # Qwen3VLProcessor.__call__
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
        do_sample_frames=False,
    ).to('cuda')

    

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            drop_method='attention',
            attn_layer_idx=args.attn_layer_idx,
            top_k_ratio=args.top_k_ratio,
            scorer_chunk_size=args.scorer_chunk_size,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--top_k_ratio", type=float, default=0.5)
    parser.add_argument("--scorer_chunk_size", type=int, default=512)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir
    run_name = args.run_name
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=run_name, curr_time=curr_time))

    for d in [result_dir, osp.join(result_dir, 'output'), osp.join(result_dir, 'log')]:
        os.makedirs(d, exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Run: {run_name}")
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Attention layer: {args.attn_layer_idx}")
    logger.info(f"Top-k ratio (keep): {args.top_k_ratio}")
    logger.info(f"Scorer chunk size: {args.scorer_chunk_size}")
    logger.info(f"Attn implementation: {args.attn_implementation}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Output JSONL: {output_jsonl}")
    logger.info(f"min_pixels: {args.min_pixels}")
    logger.info(f"max_pixels: {args.max_pixels}")
    logger.info(f"min_frames: {args.min_frames}")
    logger.info(f"max_frames: {args.max_frames}")
    logger.info(f"fps: {args.fps}")
    
    torch.manual_seed(1234)
    selected_attn_impl = args.attn_implementation
    logger.info(f"Loading model with {selected_attn_impl} ...")
    try:
        model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=selected_attn_impl,
            device_map="auto",
        )
    except ImportError as e:
        if selected_attn_impl == "flash_attention_2":
            logger.warning("flash_attention_2 unavailable, fallback to sdpa")
            selected_attn_impl = "sdpa"
            model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=selected_attn_impl,
                device_map="auto",
            )
        else:
            raise
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    task_df = pd.read_parquet(args.task_parquet)
    if args.question_id is not None:
        task_df = task_df[task_df["question_id"] == args.question_id]

    t_start = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            video_id, duration, domain, sub_category, url, videoID = \
                row.video_id, row.duration, row.domain, row.sub_category, row.url, row.videoID
            question_id, task_type, question, options, answer = \
                row.question_id, row.task_type, row.question, row.options, row.answer
            video_path = osp.join(args.video_dir, videoID + '.mp4')

            curr_max_frames = max(args.min_frames, args.max_frames)
            response = None
            while True:
                try:
                    response = run_single_sample(
                        model=model,
                        processor=processor,
                        video_path=video_path,
                        question=question,
                        options=options,
                        args=args,
                        max_frames=curr_max_frames,
                    )
                    break
                except Exception as e:
                    if not is_cuda_oom(e):
                        raise
                    next_max_frames = max(args.min_frames, curr_max_frames // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if next_max_frames >= curr_max_frames:
                        raise
                    logger.warning(
                        f"OOM on {question_id} with max_frames={curr_max_frames}, "
                        f"retry with max_frames={next_max_frames}"
                    )
                    curr_max_frames = next_max_frames

            cnt_total['overall'] += 1
            cnt_total[duration] += 1
            if extract_characters_regex(response) == answer:
                cnt_correct['overall'] += 1
                cnt_correct[duration] += 1

            output_dict = {
                'video_id': video_id,
                'duration': duration,
                'domain': domain,
                'sub_category': sub_category,
                'url': url,
                'videoID': videoID,
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'options': options.tolist(),
                'answer': answer,
                'response': response,
            }
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')

        except Exception as e:
            logger.error(f"Error processing {row.question_id}: {e}")

    t_cost = int(time.time() - t_start)

    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, "
            f"Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.1f}%")
    for dur in task_df.duration.unique():
        if cnt_total[dur] == 0:
            continue
        logger.info(
            f"Duration {dur}: {cnt_correct[dur]}/{cnt_total[dur]} = "
            f"{100 * cnt_correct[dur] / cnt_total[dur]:.1f}%")

    logger.info(
        f"Inference cost time: {t_cost // 3600}h {(t_cost % 3600) // 60}m {t_cost % 60}s")


if __name__ == '__main__':
    main()
