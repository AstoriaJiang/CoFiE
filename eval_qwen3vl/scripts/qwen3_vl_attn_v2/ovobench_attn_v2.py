"""OvOBench evaluation with chunked-scorer attention-based whole-frame dropping (v2)."""

import argparse
import gc
import json
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from moviepy.editor import VideoFileClip
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl_attn_v2 import Qwen3VLForConditionalGenerationDTD

RUN_NAME = "attn_v2_50"
CKPT_PATH = "ckpt/Qwen3-VL-4B-Instruct"
TASK_JSON = "./datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR = "./datasets/OVO-Bench/src_videos"
RESULT_DIR = "eval_qwen3vl/result_ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 720
MIN_FRAMES = 4
FPS = 1

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    elif task == "CRR":
        question = _anno_["question"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return prompt


def chunk_video(video_path, end_time, video_dir, result_dir, start_time=0):
    end_time = math.ceil(end_time)
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_dir = osp.join(video_dir, "chunked")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    output_file = osp.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if osp.exists(output_file):
        return output_file
    video = VideoFileClip(video_path)
    if end_time > video.duration:
        end_time = video.duration
    clip = video.subclip(start_time, end_time)
    tmp_dir = osp.join(result_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_audiofile = osp.join(tmp_dir, f"temp_audio_{video_name}_{start_time}_{end_time}.mp3")
    clip.write_videofile(output_file, logger=None, temp_audiofile=temp_audiofile)
    return output_file


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "CUDA out of memory" in str(exc)
    return False


def get_response(prompt, video_path, model, processor, args, max_frames):
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


def get_response_with_retry(prompt, video_path, model, processor, args):
    curr_max_frames = max(args.min_frames, args.max_frames)
    while True:
        try:
            return get_response(prompt, video_path, model, processor, args, curr_max_frames)
        except Exception as e:
            if not is_cuda_oom(e):
                raise
            next_max_frames = max(args.min_frames, curr_max_frames // 2)
            gc.collect()
            torch.cuda.empty_cache()
            if next_max_frames >= curr_max_frames:
                raise
            logger.warning(
                f"OOM with max_frames={curr_max_frames}, retry with max_frames={next_max_frames}"
            )
            curr_max_frames = next_max_frames


def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        scores = {}
        for i in range(len(results)):
            if results[i]["task"] not in scores:
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        import re
        def get_score_REC(response, gt):
            if response is None:
                return 0
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)

        def get_score_SSR_CRR(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for result in results:
            if result["task"] == "REC":
                cnt_correct = 0
                for test_info_ in result["test_info"]:
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "SSR":
                cnt_correct = 0
                for test_info_ in result["test_info"]:
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or \
                       (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "CRR":
                cnt_correct = 0
                for test_info_ in result["test_info"]:
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or \
                       (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores

    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {"backward": [], "realtime": [], "forward": []}

    if len(backward_results) > 0:
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
            avg_scores["backward"].append(sum(v) / len(v))
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward']) / len(avg_scores['backward']):.2f}\n")

    if len(realtime_results) > 0:
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
            avg_scores["realtime"].append(sum(v) / len(v))
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime']) / len(avg_scores['realtime']):.2f}\n")

    if len(forward_results) > 0:
        forward_results, forward_scores = calculate_score_forward(forward_results)
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
            avg_scores["forward"].append(sum(v) / len(v))
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward']) / len(avg_scores['forward']):.2f}\n")

    all_avgs = []
    for cat in ["backward", "realtime", "forward"]:
        if len(avg_scores[cat]) > 0:
            all_avgs.append(sum(avg_scores[cat]) / len(avg_scores[cat]))
    if len(all_avgs) > 0:
        logger.info(f"Total Avg.: {100 * sum(all_avgs) / len(all_avgs):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
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

    for d in [result_dir, osp.join(result_dir, 'output'), osp.join(result_dir, 'log'), osp.join(result_dir, 'tmp')]:
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

    if not osp.exists(args.task_json):
        logger.error(f"Task JSON file not found: {args.task_json}")
        sys.exit(1)
    with open(args.task_json, 'r') as f:
        content = f.read().strip()
        if not content:
            logger.error(f"Task JSON file is empty: {args.task_json}")
            sys.exit(1)
        try:
            task_list = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {args.task_json}: {e}")
            sys.exit(1)

    t_start = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id_, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], \
                    item['options'], item['realtime'], item['gt']
                prompt = build_prompt(task=task, question=question, options=options,
                                      _anno_=None, index=None)
                video_path = osp.join(args.video_dir, video)
                chunk_video_path = chunk_video(video_path, realtime, args.video_dir, result_dir)
                response = get_response_with_retry(prompt, chunk_video_path, model, processor, args)

                output_dict = {
                    'id': id_,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                }

            elif item['task'] in forward_tasks:
                id_, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                for i in range(len(test_info)):
                    prompt = build_prompt(task=task, question=None, options=None,
                                          _anno_=item, index=i)
                    realtime = test_info[i]['realtime']
                    video_path = osp.join(args.video_dir, video)
                    chunk_video_path = chunk_video(video_path, realtime, args.video_dir, result_dir)
                    response = get_response_with_retry(prompt, chunk_video_path, model, processor, args)
                    item['test_info'][i]['response'] = response
                output_dict = item

            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")

    t_cost = int(time.time() - t_start)

    results = defaultdict(list)
    if osp.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            if item['task'] in backward_tasks:
                results['backward'].append(item)
            elif item['task'] in realtime_tasks:
                results['realtime'].append(item)
            elif item['task'] in forward_tasks:
                results['forward'].append(item)
        score(results)

    logger.info(f"Inference cost time: {t_cost // 3600}h {(t_cost % 3600) // 60}m {t_cost % 60}s")


if __name__ == '__main__':
    main()
