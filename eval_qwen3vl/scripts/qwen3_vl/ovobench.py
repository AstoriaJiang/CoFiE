from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
import importlib.util
from collections import defaultdict
import argparse
import sys
_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.append(osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl import Qwen3VLForConditionalGenerationDTD

RUN_NAME = "feature_0d5"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "ckpt/Qwen3-VL-8B-Instruct"

TASK_JSON = "./datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR = "./datasets/OVO-Bench/src_videos"
RESULT_DIR = "eval_qwen3vl/result_ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
SIM_SAVE_PATH = "similarity/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 1000
FPS = 1

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

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
        answer = _anno_["answer"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return prompt

def get_response(prompt, video_path, model, processor, temperature=None, video_start=None, video_end=None):
    video_content = {
        "type": "video",
        "video": video_path,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "min_frames": MIN_FRAMES,
        "max_frames": MAX_FRAMES,
        "fps": FPS,
    }
    if video_start is not None:
        video_content["video_start"] = video_start
    if video_end is not None:
        video_content["video_end"] = video_end
    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]

    video_decode_t0 = time.perf_counter()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(
        messages,
        image_patch_size=processor.video_processor.patch_size,
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
        do_resize=False,
    )
    inputs = inputs.to(torch.device('cuda'))
    vd_time = time.perf_counter() - video_decode_t0

    model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
    _gen_kwargs = {}
    if temperature is None or temperature == 0:
        _gen_kwargs["do_sample"] = False
    else:
        _gen_kwargs["do_sample"] = True
        _gen_kwargs["temperature"] = temperature
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        **_gen_kwargs,
        drop_method=DROP_METHOD,
        drop_threshold=DROP_THRESHOLD,
        drop_absolute=DROP_ABSOLUTE,
        dr_save_path=DR_SAVE_PATH,
        dp_save_path=DP_SAVE_PATH,
        sim_save_path=SIM_SAVE_PATH,
    )
    model_timing_after = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    td_t0 = time.perf_counter()
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    td_time = time.perf_counter() - td_t0

    response = output_text[0]

    timing = {"video_decode": vd_time, "text_decode": td_time}
    if model_timing_before and model_timing_after:
        for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
            timing[key] = max(0.0, model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0))
    return response, timing

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
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
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
        for i, result in enumerate(results):
            if result["task"] == "REC":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "SSR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "CRR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores

    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }

    if len(backward_results) > 0:
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["backward"].append(sum(v)/len(v))
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")

    if len(realtime_results) > 0:
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["realtime"].append(sum(v)/len(v))
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")

    if len(forward_results) > 0:
        forward_results, forward_scores = calculate_score_forward(forward_results)
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["forward"].append(sum(v)/len(v))
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")

    valid_avgs = []
    if len(avg_scores['backward']) > 0:
        valid_avgs.append(sum(avg_scores['backward'])/len(avg_scores['backward']))
    if len(avg_scores['realtime']) > 0:
        valid_avgs.append(sum(avg_scores['realtime'])/len(avg_scores['realtime']))
    if len(avg_scores['forward']) > 0:
        valid_avgs.append(sum(avg_scores['forward'])/len(avg_scores['forward']))
    if len(valid_avgs) > 0:
        logger.info(f"Total Avg.: {100 * sum(valid_avgs)/len(valid_avgs):.2f}")
    else:
        logger.info("Total Avg.: N/A (no valid results)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sim_save_path", type=str, default=None)
    parser.add_argument("--dp_save_path", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    SIM_SAVE_PATH = args.sim_save_path or osp.join(RESULT_DIR, SIM_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    DP_SAVE_PATH = args.dp_save_path or osp.join(RESULT_DIR, "drop_pos/{run_name}_{curr_time}.jsonl".format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop_pos'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'similarity'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f"Running {RUN_NAME} on OvOBench")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Drop positions save path: {DP_SAVE_PATH}")
    logger.info(f"Similarity info save path: {SIM_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS: {FPS}")
    logger.info(f"Temperature (T): {args.temperature}")

    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    logger.info(f"Attention implementation: {attn_impl}")
    model = Qwen3VLForConditionalGenerationDTD.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    with open(TASK_JSON, 'r') as f:
        task_list = json.load(f)

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    profiled_samples = 0

    start_time = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
                prompt = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )
                video_path = osp.join(VIDEO_DIR, video)
                response, timing = get_response(
                    prompt=prompt,
                    video_path=video_path,
                    model=model,
                    processor=processor,
                    temperature=args.temperature,
                    video_start=0,
                    video_end=int(realtime),
                )

                output_dict = {
                    'id': id,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                }

            elif item['task'] in forward_tasks:
                id, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                timing = None
                for i in range(len(test_info)):
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=i,
                    )
                    realtime = test_info[i]['realtime']
                    video_path = osp.join(VIDEO_DIR, video)
                    resp, timing = get_response(
                        prompt=prompt,
                        video_path=video_path,
                        model=model,
                        processor=processor,
                        temperature=args.temperature,
                        video_start=0,
                        video_end=int(realtime),
                    )
                    item['test_info'][i]['response'] = resp

                output_dict = item

            if timing:
                for k, v in timing.items():
                    if k in stage_totals:
                        stage_totals[k] += v
            profiled_samples += 1

            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing {item}: {e}")
    end_time = time.time()
    cost_time_sec = end_time - start_time
    cost_time = int(cost_time_sec)

    try:
        results = defaultdict(list)
        with open(OUTPUT_JSONL, 'r') as f:
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

        drop_enabled = DROP_METHOD is not None and str(DROP_METHOD).lower() != "none"
        if drop_enabled and DR_SAVE_PATH is not None and osp.exists(DR_SAVE_PATH):
            drop_list, total_list, ratio_list = [], [], []
            with open(DR_SAVE_PATH, 'r') as f:
                lines = f.readlines()
            for line in lines:
                drop_ratio_info = json.loads(line)
                drop_list.append(drop_ratio_info['drop'])
                total_list.append(drop_ratio_info['total'])
                ratio_list.append(drop_ratio_info['ratio'])
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
            logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")
        elif drop_enabled and DR_SAVE_PATH is not None:
            logger.warning(f"Drop ratio file not found, skip statistics: {DR_SAVE_PATH}")

        stage_keys = ["video_decode", "visual_encode", "token_drop", "llm_prefill", "llm_decode", "text_decode"]
        stage_name_map = {
            "video_decode": "Frame Extraction",
            "visual_encode": "Vision Encoder",
            "token_drop": "Token Drop",
            "llm_prefill": "LLM Prefill",
            "llm_decode": "LLM Decode",
            "text_decode": "Text Detokenize (processor.batch_decode)",
        }
        measured_stage_total = sum(stage_totals.values())
        logger.info("Stage time breakdown:")
        for key in stage_keys:
            sec = stage_totals[key]
            pct_total = (100.0 * sec / cost_time_sec) if cost_time_sec > 0 else 0.0
            pct_profiled = (100.0 * sec / measured_stage_total) if measured_stage_total > 0 else 0.0
            logger.info(
                f"- {stage_name_map[key]}: {sec:.2f}s | {pct_total:.2f}% of end-to-end | {pct_profiled:.2f}% inside profiled stages"
            )
        other_time = max(0.0, cost_time_sec - measured_stage_total)
        other_pct = (100.0 * other_time / cost_time_sec) if cost_time_sec > 0 else 0.0
        logger.info(f"- Other/uncaptured: {other_time:.2f}s | {other_pct:.2f}% of end-to-end")

        if measured_stage_total > 0:
            bottleneck_key = max(stage_totals, key=stage_totals.get)
            logger.info(
                f"Profiled-stage bottleneck: {stage_name_map[bottleneck_key]} ({100.0 * stage_totals[bottleneck_key] / measured_stage_total:.2f}% of profiled stages)"
            )

        if profiled_samples > 0:
            logger.info(f"Profiled samples: {profiled_samples}")
            for key in stage_keys:
                logger.info(f"- Avg {stage_name_map[key]} per sample: {stage_totals[key] / profiled_samples:.4f}s")
    finally:
        logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
