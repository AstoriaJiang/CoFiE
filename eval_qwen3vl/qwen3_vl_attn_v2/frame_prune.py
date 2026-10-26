"""
Frame-level pruning utilities based on per-frame attention weights.

Provides:
- ``compute_frame_rank``: sort frames by weight (ascending) and return indices.
- ``select_topk_frames``: keep only the top-k% highest-weight frames.
- ``drop_low_attention_frames``: physically remove low-weight frames from
  the raw video tensor, compatible with ``streamingbench_frame_drop.py``.
- ``save_results``: persist ``t2v_attention.json``, ``frame_rank.txt``,
  and ``keep_frames.json`` into an output directory.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch


def compute_frame_rank(frame_weights: List[Dict]) -> List[int]:
    """Return frame indices sorted ascending by ``frame_weight``.

    Compatible with the ``frame_rank.txt`` format used by
    ``rank_frame_weight.py`` and ``streamingbench_frame_drop.py``.
    """
    ranked = sorted(frame_weights, key=lambda x: x['frame_weight'])
    return [int(item['frame_index']) for item in ranked]


def select_topk_frames(
    frame_weights: List[Dict],
    top_k_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Select frames with the highest attention weights.

    Parameters
    ----------
    frame_weights : list[dict]
        Output of ``LayerAttentionCollector.get_frame_weights()``.
    top_k_ratio : float
        Fraction of frames to *keep* (e.g. 0.5 = keep top 50%).

    Returns
    -------
    keep_indices : list[int]
        Frame indices to keep, sorted ascending.
    drop_indices : list[int]
        Frame indices to drop, sorted ascending.
    """
    n_frames = len(frame_weights)
    if n_frames == 0:
        return [], []

    ratio = max(0.0, min(1.0, float(top_k_ratio)))
    n_keep = max(1, int(round(n_frames * ratio)))
    n_keep = min(n_keep, n_frames)
    ranked_desc = sorted(
        frame_weights, key=lambda x: x['frame_weight'], reverse=True)
    keep_set = {int(item['frame_index']) for item in ranked_desc[:n_keep]}
    keep_indices = sorted(keep_set)
    drop_indices = sorted(set(range(n_frames)) - keep_set)
    return keep_indices, drop_indices


def drop_low_attention_frames(
    video_tensor: torch.Tensor,
    keep_indices: List[int],
    tps: int = 2,
) -> torch.Tensor:
    """Remove frames not in *keep_indices* from a raw video tensor.

    Parameters
    ----------
    video_tensor : Tensor
        Shape ``(T_raw, C, H, W)`` where ``T_raw = n_frames * tps``.
    keep_indices : list[int]
        Frame indices (after ViT temporal merge) to keep.
    tps : int
        Temporal patch size (raw pixel frames per merged frame).

    Returns
    -------
    Tensor
        Filtered video tensor containing only kept frames.
    """
    if video_tensor.ndim != 4:
        raise ValueError(
            f"Expected video tensor shape (T, C, H, W), got {tuple(video_tensor.shape)}")
    if video_tensor.shape[0] == 0:
        raise ValueError("video_tensor has zero frames")
    if tps <= 0:
        raise ValueError(f"tps must be positive, got {tps}")

    max_frame_idx = (int(video_tensor.shape[0]) - 1) // int(tps)
    keep_sorted = sorted({
        int(f) for f in keep_indices if 0 <= int(f) <= max_frame_idx
    })
    if not keep_sorted:
        keep_sorted = [0]

    raw_indices = []
    for f in keep_sorted:
        for j in range(tps):
            ridx = f * tps + j
            if ridx < int(video_tensor.shape[0]):
                raw_indices.append(ridx)

    if not raw_indices:
        raw_indices = [0]
    return video_tensor[raw_indices]


def save_results(
    output_dir: str,
    frame_weights: List[Dict],
    keep_indices: List[int],
    drop_indices: List[int],
    *,
    layer_idx: int,
    top_k_ratio: float,
    question_id: str = "",
):
    """Persist attention analysis results to *output_dir*.

    Creates three files:
    - ``t2v_attention.json`` – full per-frame / per-patch weights.
    - ``frame_rank.txt`` – frame indices sorted ascending by weight.
    - ``keep_frames.json`` – metadata about this pruning run.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 't2v_attention.json'), 'w') as f:
        json.dump(frame_weights, f, indent=2)

    ranked = compute_frame_rank(frame_weights)
    with open(os.path.join(output_dir, 'frame_rank.txt'), 'w') as f:
        f.write('\n'.join(str(i) for i in ranked) + '\n')

    meta = {
        'question_id': question_id,
        'layer_idx': layer_idx,
        'top_k_ratio': top_k_ratio,
        'n_frames': len(frame_weights),
        'n_keep': len(keep_indices),
        'n_drop': len(drop_indices),
        'keep_indices': keep_indices,
        'drop_indices': drop_indices,
    }
    with open(os.path.join(output_dir, 'keep_frames.json'), 'w') as f:
        json.dump(meta, f, indent=2)
