"""Model-based video-frame filtering with CLIP patch embeddings.

The selector intentionally exposes the same ``config``/``select_frames`` surface
as the histogram selector used by ``qwen3_vl_hdiff_parellel``.  It can therefore
replace only the first (pre-encoder) filtering stage while leaving the optional
Qwen attention-based token filtering stage unchanged.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CLIPPatchFrameSelectorConfig:
    enabled: bool = True
    model_path: str = "ckpt/clip-vit-base-patch32"
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 32
    strategy: str = "relative"
    drop_ratio: float = 0.2
    similarity_threshold: float = 0.95
    min_keep_frames: int = 4
    max_interval: int = 30
    stats_path: Optional[str] = None
    local_files_only: bool = True


class CLIPPatchFrameSelector:
    """Select frames from mean cosine similarity of aligned CLIP patches.

    For adjacent frames ``t-1`` and ``t``, the score is::

        sim(t-1, t) = mean_p cosine(z[t-1, p], z[t, p])

    where ``z[..., p]`` is the final CLIP vision hidden state for spatial patch
    ``p`` (the CLS token is excluded).  ``relative`` selection keeps frame 0 and
    the frames with the lowest adjacent similarity, giving an exact drop ratio.
    ``threshold`` selection keeps frames whose similarity to the previous frame
    or most recently kept frame is below ``similarity_threshold``.
    """

    def __init__(self, config: Optional[CLIPPatchFrameSelectorConfig] = None):
        self.config = config or CLIPPatchFrameSelectorConfig()
        self._model = None
        self._image_processor = None
        self._owner_pid: Optional[int] = None
        self._model_load_seconds = 0.0
        self._last_stats: dict = {}
        self.reset_stats()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.strategy not in {"relative", "threshold"}:
            raise ValueError("CLIP frame selection strategy must be 'relative' or 'threshold'")
        if not 0.0 <= self.config.drop_ratio < 1.0:
            raise ValueError("CLIP drop_ratio must be in [0, 1)")
        if not -1.0 <= self.config.similarity_threshold <= 1.0:
            raise ValueError("CLIP similarity_threshold must be in [-1, 1]")
        if self.config.batch_size < 1:
            raise ValueError("CLIP batch_size must be positive")

    @property
    def last_stats(self) -> dict:
        return dict(self._last_stats)

    def reset_stats(self) -> None:
        self._stats = {
            "calls": 0,
            "frames_in": 0,
            "frames_out": 0,
            "preprocess_seconds": 0.0,
            "forward_seconds": 0.0,
            "selection_seconds": 0.0,
            "total_seconds": 0.0,
        }

    def get_stats(self) -> dict:
        result = dict(self._stats)
        result["model_load_seconds"] = self._model_load_seconds
        calls = max(int(result["calls"]), 1)
        result["mean_seconds"] = result["total_seconds"] / calls
        result["keep_ratio"] = (
            result["frames_out"] / result["frames_in"] if result["frames_in"] else 1.0
        )
        return result

    def _resolve_device(self) -> torch.device:
        requested = str(self.config.device).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CLIP selector requested CUDA, but CUDA is not available")
        return torch.device(requested)

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        name = str(self.config.dtype).lower()
        if name == "auto":
            return torch.float16 if device.type == "cuda" else torch.float32
        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if name not in aliases:
            raise ValueError(f"Unsupported CLIP dtype: {self.config.dtype}")
        dtype = aliases[name]
        if device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype

    def _ensure_model(self) -> None:
        current_pid = os.getpid()
        if self._model is not None and self._owner_pid == current_pid:
            return

        load_t0 = time.perf_counter()
        # Do not retain a model inherited across a fork.
        self._model = None
        self._image_processor = None
        from transformers import CLIPImageProcessor, CLIPVisionModel

        model_path = str(Path(self.config.model_path).expanduser())
        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        self._image_processor = CLIPImageProcessor.from_pretrained(
            model_path,
            local_files_only=self.config.local_files_only,
        )
        self._model = CLIPVisionModel.from_pretrained(
            model_path,
            local_files_only=self.config.local_files_only,
            torch_dtype=dtype,
        ).eval().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._model_load_seconds += time.perf_counter() - load_t0
        self._owner_pid = current_pid

    @staticmethod
    def _frame_to_hwc_uint8(frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D video frame, got shape {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = array[..., :3]
        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            if array.max(initial=0.0) <= 1.0:
                array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    @torch.inference_mode()
    def _extract_patch_embeddings(self, frames: Sequence) -> tuple[torch.Tensor, float, float]:
        self._ensure_model()
        assert self._model is not None and self._image_processor is not None
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        all_embeddings = []
        preprocess_seconds = 0.0
        forward_seconds = 0.0

        for start in range(0, len(frames), self.config.batch_size):
            batch_frames = [self._frame_to_hwc_uint8(f) for f in frames[start : start + self.config.batch_size]]
            t0 = time.perf_counter()
            inputs = self._image_processor(images=batch_frames, return_tensors="pt")
            preprocess_seconds += time.perf_counter() - t0
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype, non_blocking=False)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            hidden = self._model(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - t0
            all_embeddings.append(F.normalize(hidden.float(), dim=-1).cpu())

        return torch.cat(all_embeddings, dim=0), preprocess_seconds, forward_seconds

    @staticmethod
    def adjacent_patch_similarities(patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Return mean aligned-patch cosine similarities for adjacent frames."""
        if patch_embeddings.ndim != 3:
            raise ValueError("patch_embeddings must have shape [frames, patches, hidden]")
        if patch_embeddings.shape[0] < 2:
            return patch_embeddings.new_empty((0,))
        normalized = F.normalize(patch_embeddings.float(), dim=-1)
        return (normalized[:-1] * normalized[1:]).sum(dim=-1).mean(dim=-1)

    def _relative_keep(self, similarities: torch.Tensor, n: int) -> list[int]:
        keep_count = int(round(n * (1.0 - self.config.drop_ratio)))
        keep_count = min(max(keep_count, self.config.min_keep_frames, 1), n)
        # Frame 0 has no predecessor. Other frames are ranked by ascending
        # similarity, i.e. largest semantic/appearance change first.
        ranked = (torch.argsort(similarities, descending=False) + 1).tolist()
        return sorted([0] + ranked[: max(0, keep_count - 1)])

    def _threshold_keep(self, patch_embeddings: torch.Tensor) -> list[int]:
        n = patch_embeddings.shape[0]
        normalized = F.normalize(patch_embeddings.float(), dim=-1)
        keep = [0]
        last_keep = 0
        for idx in range(1, n - 1):
            adjacent_sim = (normalized[idx - 1] * normalized[idx]).sum(-1).mean().item()
            reference_sim = (normalized[last_keep] * normalized[idx]).sum(-1).mean().item()
            should_keep = min(adjacent_sim, reference_sim) < self.config.similarity_threshold
            if self.config.max_interval > 0 and idx - last_keep >= self.config.max_interval:
                should_keep = True
            if should_keep:
                keep.append(idx)
                last_keep = idx
        if n > 1:
            keep.append(n - 1)
        min_keep = min(max(self.config.min_keep_frames, 1), n)
        if len(keep) < min_keep:
            keep.extend(np.linspace(0, n - 1, min_keep).round().astype(int).tolist())
        return sorted(set(keep))

    def _append_stats(self, record: dict) -> None:
        if not self.config.stats_path:
            return
        path = Path(self.config.stats_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        # One O_APPEND write keeps records intact when distributed ranks share a file.
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    def select_frames(self, frames: Sequence) -> list[int]:
        n = len(frames)
        if not self.config.enabled or n <= 2 or n <= self.config.min_keep_frames:
            total_t0 = time.perf_counter()
            keep = list(range(n))
            record = {
                "pid": os.getpid(),
                "frames_in": n,
                "frames_out": len(keep),
                "keep_indices": keep,
                "preprocess_seconds": 0.0,
                "forward_seconds": 0.0,
                "selection_seconds": 0.0,
                "total_seconds": time.perf_counter() - total_t0,
            }
        else:
            # Warm up the model outside the per-video timer.  Model loading is
            # an initialization cost, not frame-filter latency reported in the
            # paper comparison.  _extract_patch_embeddings calls this again as
            # a fork-safety guard, but it becomes a no-op in this process.
            self._ensure_model()
            total_t0 = time.perf_counter()
            embeddings, preprocess_seconds, forward_seconds = self._extract_patch_embeddings(frames)
            selection_t0 = time.perf_counter()
            adjacent = self.adjacent_patch_similarities(embeddings)
            if self.config.strategy == "relative":
                keep = self._relative_keep(adjacent, n)
            else:
                keep = self._threshold_keep(embeddings)
            selection_seconds = time.perf_counter() - selection_t0
            record = {
                "pid": os.getpid(),
                "frames_in": n,
                "frames_out": len(keep),
                "keep_indices": keep,
                "adjacent_similarities": adjacent.tolist(),
                "preprocess_seconds": preprocess_seconds,
                "forward_seconds": forward_seconds,
                "selection_seconds": selection_seconds,
                "total_seconds": time.perf_counter() - total_t0,
            }

        self._last_stats = record
        self._stats["calls"] += 1
        self._stats["frames_in"] += record["frames_in"]
        self._stats["frames_out"] += record["frames_out"]
        for key in ("preprocess_seconds", "forward_seconds", "selection_seconds", "total_seconds"):
            self._stats[key] += record[key]
        self._append_stats(record)
        return keep

    def to_dict(self) -> dict:
        return asdict(self.config)
