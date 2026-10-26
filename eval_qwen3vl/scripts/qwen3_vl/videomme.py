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
from collections.abc import Mapping
import argparse
import sys
from torch.utils.data import DataLoader, Dataset
_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.append(osp.join(_repo, 'eval_qwen3vl'))
from qwen3_vl import Qwen3VLForConditionalGenerationDTD

# Parameters
RUN_NAME = "feature_0d5"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"

TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval_qwen3vl/result_videomme"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
SIM_SAVE_PATH = "similarity/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MAX_FRAMES = 1000
MIN_FRAMES = 4
FPS = 1

# Prompt template
prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

## helper functions
def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

def resolve_processor_path(ckpt_path):
    adapter_config_path = osp.join(ckpt_path, "adapter_config.json")
    if osp.isfile(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        return adapter_config.get("base_model_name_or_path") or ckpt_path
    return ckpt_path

def options_to_list(options):
    return options.tolist() if hasattr(options, "tolist") else list(options)

def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)

def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("Only batch_size=1 is supported for variable-length VideoMME samples")
    return batch[0]

def move_to_cuda(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {key: move_to_cuda(value, device, non_blocking) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(value, device, non_blocking) for value in obj)
    if isinstance(obj, list):
        return [move_to_cuda(value, device, non_blocking) for value in obj]
    return obj

def build_dataloader(dataset, batch_size=1, num_workers=0, prefetch_factor=2, pin_memory=False, persistent_workers=False):
    if batch_size != 1:
        raise ValueError("Only batch_size=1 is supported for variable-length VideoMME samples")
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)

class VideoMMEDataset(Dataset):
    def __init__(self, task_df, processor_path, video_dir, min_pixels, max_pixels, max_frames, min_frames, fps):
        self.task_df = task_df.reset_index(drop=True)
        self.processor_path = processor_path
        self.video_dir = video_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.fps = fps
        self._processor = None

    def __len__(self):
        return len(self.task_df)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.processor_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        return self._processor

    def __getitem__(self, idx):
        row = self.task_df.iloc[idx]
        options = options_to_list(row.options)
        metadata = {
            "idx": int(idx),
            "video_id": row.video_id,
            "duration": row.duration,
            "domain": row.domain,
            "sub_category": row.sub_category,
            "url": row.url,
            "videoID": row.videoID,
            "question_id": row.question_id,
            "task_type": row.task_type,
            "question": row.question,
            "options": options,
            "answer": row.answer,
        }
        try:
            processor = self._get_processor()
            video_path = osp.join(self.video_dir, row.videoID + '.mp4')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                            "max_frames": self.max_frames,
                            "min_frames": self.min_frames,
                            "fps": self.fps,
                        },
                        {
                            "type": "text",
                            "text": prompt.format(row.question, '\n'.join(options)),
                        },
                    ],
                }
            ]
            video_decode_t0 = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            return {
                "inputs": inputs,
                "metadata": metadata,
                "preprocess_time": time.perf_counter() - video_decode_t0,
                "error": None,
            }
        except Exception as e:
            return {
                "inputs": None,
                "metadata": metadata,
                "preprocess_time": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true") # Default is absolute
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sim_save_path", type=str, default=None)
    parser.add_argument("--dp_save_path", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    args = parser.parse_args()

    # Update global variables
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_PARQUET = args.task_parquet
    VIDEO_DIR = args.video_dir
    if args.log_dir:
        LOG_PATH = osp.join(args.log_dir, f"{RUN_NAME}_{curr_time}.log")
    else:
        LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    SIM_SAVE_PATH = args.sim_save_path or osp.join(RESULT_DIR, SIM_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    DP_SAVE_PATH = args.dp_save_path or osp.join(RESULT_DIR, "drop_pos/{run_name}_{curr_time}.jsonl".format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MAX_FRAMES = args.max_frames
    MIN_FRAMES = args.min_frames
    FPS = args.fps

    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop_pos'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'similarity'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Print run info
    logger.info(f"Running {RUN_NAME} on Video-MME")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task parquet: {TASK_PARQUET}")
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
    logger.info(f"DataLoader batch_size: {args.batch_size}")
    logger.info(f"DataLoader num_workers: {args.num_workers}")
    logger.info(f"DataLoader prefetch_factor: {args.prefetch_factor}")
    logger.info(f"DataLoader pin_memory: {args.pin_memory}")
    logger.info(f"DataLoader persistent_workers: {args.persistent_workers}")
    logger.info(
        "Generation mode: greedy"
        if args.temperature is None or args.temperature == 0
        else "Generation mode: sample"
    )

    # Load model and processor
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
    processor_path = resolve_processor_path(CKPT_PATH)
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model from {CKPT_PATH}")
    logger.info(f"Load processor from {processor_path}")

    if hasattr(model, "reset_stage_timing"):
        model.reset_stage_timing()

    # Load task info
    task_df = pd.read_parquet(TASK_PARQUET)

    stage_totals = {
        "video_decode": 0.0,
        "visual_encode": 0.0,
        "token_drop": 0.0,
        "llm_prefill": 0.0,
        "llm_decode": 0.0,
        "text_decode": 0.0,
    }
    profiled_samples = 0

    # Inference
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    dataset = VideoMMEDataset(
        task_df,
        processor_path,
        VIDEO_DIR,
        MIN_PIXELS,
        MAX_PIXELS,
        MAX_FRAMES,
        MIN_FRAMES,
        FPS,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    for batch in tqdm(dataloader, total=len(task_df)):
        metadata = batch["metadata"]
        stage_totals["video_decode"] += float(batch.get("preprocess_time", 0.0))
        if batch.get("error"):
            logger.error(f"Error in preprocessing {metadata}: {batch['error']}")
            continue
        try:
            inputs = move_to_cuda(batch["inputs"], torch.device('cuda'), non_blocking=args.pin_memory)
            model_timing_before = model.get_stage_timing() if hasattr(model, "get_stage_timing") else {}
            _gen_kwargs = {}
            if args.temperature is None or args.temperature == 0:
                _gen_kwargs["do_sample"] = False
            else:
                _gen_kwargs["do_sample"] = True
                _gen_kwargs["temperature"] = args.temperature
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
            if model_timing_before and model_timing_after:
                for key in ("visual_encode", "token_drop", "llm_prefill", "llm_decode"):
                    stage_totals[key] += max(
                        0.0,
                        model_timing_after.get(key, 0.0) - model_timing_before.get(key, 0.0),
                    )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            text_decode_t0 = time.perf_counter()
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            stage_totals["text_decode"] += time.perf_counter() - text_decode_t0

            response = output_text[0]
            profiled_samples += 1
            duration = metadata["duration"]
            answer = metadata["answer"]
            cnt_total['overall'] += 1
            cnt_total[duration] += 1
            if extract_characters_regex(response) == answer:
                cnt_correct['overall'] += 1
                cnt_correct[duration] += 1

            output_dict = {
                'video_id': metadata["video_id"],
                'duration': duration,
                'domain': metadata["domain"],
                'sub_category': metadata["sub_category"],
                'url': metadata["url"],
                'videoID': metadata["videoID"],
                'question_id': metadata["question_id"],
                'task_type': metadata["task_type"],
                'question': metadata["question"],
                'options': metadata["options"],
                'answer': answer,
                'response': response
            }
            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict, default=json_default) + '\n')
        except Exception as e:
            logger.error(f"Error in processing {metadata}: {e}")
        # break
    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.1f}%")
    for duration in task_df.duration.unique():
        if cnt_total[duration] == 0:
            continue
        logger.info(f"Duration {duration}: {cnt_correct[duration]}/{cnt_total[duration]} = {100*cnt_correct[duration]/cnt_total[duration]:.1f}%")

    # Collect drop ratio info
    if (
        DROP_METHOD is not None
        and str(DROP_METHOD).lower() != "none"
        and DR_SAVE_PATH is not None
        and osp.exists(DR_SAVE_PATH)
    ):
        drop_list, total_list, ratio_list = [], [], []
        with open(DR_SAVE_PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info['drop'])
            total_list.append(drop_ratio_info['total'])
            ratio_list.append(drop_ratio_info['ratio'])
        if drop_list and total_list and sum(total_list) > 0:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
            logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")

    # Print stage time breakdown
    cost_time_sec = end_time - start_time
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

    # Print time
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
