"""
Test script to compare original vs custom processor outputs.
"""
import sys
import os.path as osp
_repo = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.append(osp.join(_repo, 'eval_qwen3vl'))

import torch
import numpy as np
from transformers import AutoProcessor
from qwen3_vl_fps import Qwen3VLVideoProcessor
from qwen_vl_utils import process_vision_info

CKPT_PATH = "ckpt/Qwen3-VL-4B-Instruct"
VIDEO_PATH = "./datasets/Video-MME/data/mAwgdX5VxGc.mp4"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MAX_FRAMES = 1000
MIN_FRAMES = 4
FPS = 1

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "max_frames": MAX_FRAMES,
                "min_frames": MIN_FRAMES,
                "fps": FPS
            },
            {
                "type": "text",
                "text": "Describe this video."
            },
        ],
    }
]

def test_original_processor():
    print("=" * 60)
    print("Testing ORIGINAL processor (transformers)")
    print("=" * 60)
    
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    
    print(f"video_processor type: {type(processor.video_processor)}")
    print(f"video_processor.size: {processor.video_processor.size}")
    print(f"video_processor.fps: {processor.video_processor.fps}")
    print(f"video_processor.min_frames: {processor.video_processor.min_frames}")
    print(f"video_processor.max_frames: {processor.video_processor.max_frames}")
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    print(f"pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    print(f"video_grid_thw: {inputs['video_grid_thw']}")
    print(f"pixel_values_videos mean: {inputs['pixel_values_videos'].mean().item():.6f}")
    print(f"pixel_values_videos std: {inputs['pixel_values_videos'].std().item():.6f}")
    print(f"pixel_values_videos first 10 values: {inputs['pixel_values_videos'].flatten()[:10].tolist()}")
    
    return inputs

def test_custom_processor():
    print("=" * 60)
    print("Testing CUSTOM processor (with fallback fix)")
    print("=" * 60)
    
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    
    original_size = processor.video_processor.size
    processor.video_processor = Qwen3VLVideoProcessor.from_pretrained(CKPT_PATH)
    processor.video_processor.size = original_size
    
    print(f"video_processor type: {type(processor.video_processor)}")
    print(f"video_processor.size: {processor.video_processor.size}")
    print(f"video_processor.fps: {processor.video_processor.fps}")
    print(f"video_processor.min_frames: {processor.video_processor.min_frames}")
    print(f"video_processor.max_frames: {processor.video_processor.max_frames}")
    print(f"frame_selector.enabled: {processor.video_processor.frame_selector.config.enabled}")
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    print(f"pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    print(f"video_grid_thw: {inputs['video_grid_thw']}")
    print(f"pixel_values_videos mean: {inputs['pixel_values_videos'].mean().item():.6f}")
    print(f"pixel_values_videos std: {inputs['pixel_values_videos'].std().item():.6f}")
    print(f"pixel_values_videos first 10 values: {inputs['pixel_values_videos'].flatten()[:10].tolist()}")
    
    return inputs

if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    inputs_orig = test_original_processor()
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    inputs_custom = test_custom_processor()
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    shape_match = inputs_orig['pixel_values_videos'].shape == inputs_custom['pixel_values_videos'].shape
    print(f"Shape match: {shape_match}")
    
    if shape_match:
        diff = (inputs_orig['pixel_values_videos'] - inputs_custom['pixel_values_videos']).abs()
        print(f"Max absolute difference: {diff.max().item():.10f}")
        print(f"Mean absolute difference: {diff.mean().item():.10f}")
        
        if diff.max().item() == 0:
            print("RESULT: Outputs are IDENTICAL!")
        else:
            print("RESULT: Outputs are DIFFERENT!")
    else:
        print(f"Original shape: {inputs_orig['pixel_values_videos'].shape}")
        print(f"Custom shape: {inputs_custom['pixel_values_videos'].shape}")
        print("RESULT: Outputs have DIFFERENT SHAPES!")
