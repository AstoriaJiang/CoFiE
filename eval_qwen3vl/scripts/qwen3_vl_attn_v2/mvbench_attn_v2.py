"""MVBench evaluation with chunked-scorer attention-based whole-frame dropping (v2)."""

import argparse
import gc
import json
import logging
import os
import os.path as osp
import re
import string
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_attn_v2 import Qwen3VLForConditionalGenerationDTD

RUN_NAME = "attn_v2_50"
CKPT_PATH = "ckpt/Qwen3-VL-8B-Instruct"
RESULT_DIR = "eval_qwen3vl/result_mvbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 720
MIN_FRAMES = 4
FPS = 1

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

SUB_TASKS = list(DATA_LIST.keys())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def build_prompt(question, candidates):
    option_prompt = ""
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(candidates):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"
    full_text = "Question:" + question + "\nOption:\n" + option_prompt + "Only give the best option.\n"
    return full_text


def extract_answer(response):
    response = response.strip()
    for prefix in [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]:
        response = response.replace(prefix, "")
    m = re.search(r"[A-E]", response)
    return m[0] if m else ""


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "CUDA out of memory" in str(exc)
    return False


def run_single_sample(model, processor, video_path, question, candidates, args, max_frames):
    prompt = build_prompt(question, candidates)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
             "max_frames": max_frames, "min_frames": args.min_frames,
             "fps": args.fps},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
        do_sample_frames=False,
    ).to('cuda')

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=16,
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


def get_gt_option_letter(candidates, answer):
    option_letters = string.ascii_uppercase
    for i, candidate in enumerate(candidates):
        if candidate == answer:
            return option_letters[i]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--sub_tasks", type=str, nargs="+", default=SUB_TASKS)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--top_k_ratio", type=float, default=0.5)
    parser.add_argument("--scorer_chunk_size", type=int, default=512)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
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
    logger.info(f"Sub tasks: {args.sub_tasks}")

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
    except ImportError:
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

    t_start = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    for sub_task in args.sub_tasks:
        logger.info(f"Loading dataset for sub_task: {sub_task}")
        dataset = load_dataset("OpenGVLab/MVBench", sub_task, split="train")
        dataset_folder = DATA_LIST[sub_task]

        for row in tqdm(dataset, desc=sub_task):
            try:
                question = row["question"]
                candidates = row["candidates"]
                answer = row["answer"]
                video_name = row["video"]

                video_path = osp.join(args.video_dir, dataset_folder, video_name)
                if not osp.exists(video_path):
                    alt_video_path = osp.join(args.video_dir, "data0613", dataset_folder, video_name)
                    if osp.exists(alt_video_path):
                        video_path = alt_video_path
                    else:
                        logger.warning(f"Video not found: {video_path}")
                        continue

                curr_max_frames = max(args.min_frames, args.max_frames)
                response = None
                while True:
                    try:
                        response = run_single_sample(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            question=question,
                            candidates=candidates,
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
                            f"OOM with max_frames={curr_max_frames}, "
                            f"retry with max_frames={next_max_frames}"
                        )
                        curr_max_frames = next_max_frames

                gt_letter = get_gt_option_letter(candidates, answer)
                pred_letter = extract_answer(response)

                cnt_total['overall'] += 1
                cnt_total[sub_task] += 1
                if pred_letter == gt_letter:
                    cnt_correct['overall'] += 1
                    cnt_correct[sub_task] += 1

                output_dict = {
                    'sub_task': sub_task,
                    'video': video_name,
                    'question': question,
                    'candidates': candidates,
                    'answer': answer,
                    'gt_letter': gt_letter,
                    'response': response,
                    'pred_letter': pred_letter,
                }
                with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')

            except Exception as e:
                logger.error(f"Error processing {video_name}: {e}")

    t_cost = int(time.time() - t_start)

    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, "
            f"Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.1f}%")
    for sub_task in args.sub_tasks:
        if cnt_total[sub_task] == 0:
            continue
        logger.info(
            f"Sub-task {sub_task}: {cnt_correct[sub_task]}/{cnt_total[sub_task]} = "
            f"{100 * cnt_correct[sub_task] / cnt_total[sub_task]:.1f}%")

    logger.info(
        f"Inference cost time: {t_cost // 3600}h {(t_cost % 3600) // 60}m {t_cost % 60}s")


if __name__ == '__main__':
    main()
