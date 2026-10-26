"""lmms-eval adapter for CLIP first-stage filtering + optional attention drop."""

from __future__ import annotations

from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl_hdiff_parellel import (
    Qwen3_VL_HDiff_Parellel,
    _load_hdiff_video_processor_class,
)

from .frame_selector import CLIPPatchFrameSelector, CLIPPatchFrameSelectorConfig


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@register_model("qwen3_vl_clip")
class Qwen3_VL_CLIP(Qwen3_VL_HDiff_Parellel):
    """Qwen3-VL with CLIP patch filtering before its visual encoder."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        clip_enabled: bool = True,
        clip_model_path: str = "ckpt/clip-vit-base-patch32",
        clip_device: str = "cuda",
        clip_dtype: str = "float16",
        clip_batch_size: int = 32,
        clip_strategy: str = "relative",
        clip_drop_ratio: float = 0.2,
        clip_similarity_threshold: float = 0.95,
        clip_min_keep_frames: int = 4,
        clip_max_interval: int = 30,
        clip_stats_path: Optional[str] = None,
        clip_local_files_only: bool = True,
        **kwargs,
    ) -> None:
        # Histogram filtering is forced off. The inherited model code is retained
        # solely for the optional second-stage attention/token drop implementation.
        kwargs.pop("hist_enabled", None)
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            hist_enabled=False,
            **kwargs,
        )
        self.clip_enabled = _as_bool(clip_enabled)
        if not self.clip_enabled:
            return

        video_processor_cls = _load_hdiff_video_processor_class()
        original_size = getattr(self.processor.video_processor, "size", None)
        video_processor = video_processor_cls.from_pretrained(pretrained)
        if original_size is not None:
            video_processor.size = original_size
        video_processor.frame_selector = CLIPPatchFrameSelector(
            CLIPPatchFrameSelectorConfig(
                enabled=True,
                model_path=clip_model_path,
                device=clip_device,
                dtype=clip_dtype,
                batch_size=int(clip_batch_size),
                strategy=str(clip_strategy),
                drop_ratio=float(clip_drop_ratio),
                similarity_threshold=float(clip_similarity_threshold),
                min_keep_frames=int(clip_min_keep_frames),
                max_interval=int(clip_max_interval),
                stats_path=clip_stats_path,
                local_files_only=_as_bool(clip_local_files_only),
            )
        )
        self.processor.video_processor = video_processor

    def get_clip_filter_stats(self) -> dict:
        selector = getattr(self.processor.video_processor, "frame_selector", None)
        return selector.get_stats() if selector is not None else {}
