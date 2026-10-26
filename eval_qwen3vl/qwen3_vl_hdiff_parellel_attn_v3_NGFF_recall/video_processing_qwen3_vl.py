# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""video processor class for Qwen3-VL."""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torchvision_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


@dataclass
class HistogramFrameSelectorConfig:
    enabled: bool = False
    method: str = "histogram"
    bins: int = 32
    threshold: float = 0.15
    drop_ratio: Optional[float] = None
    relative: bool = False
    min_keep_frames: int = 4
    max_interval: int = 30
    size: int = 32


class HistogramFrameSelector:
    def __init__(self, config: Optional[HistogramFrameSelectorConfig] = None):
        self.config = config or HistogramFrameSelectorConfig()
        self._dct_cache = {}

    def _to_gray(self, frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                frame = np.moveaxis(frame, 0, -1)
            if frame.shape[-1] >= 3:
                frame = frame[..., :3].astype(np.float32)
                return frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114
            return frame[..., 0].astype(np.float32)
        return frame.astype(np.float32)

    def _histogram(self, frame) -> np.ndarray:
        gray = self._to_gray(frame)
        gray = gray[np.isfinite(gray)]
        if gray.size == 0:
            return np.zeros(self.config.bins, dtype=np.float32)

        hist_range = (0.0, 255.0) if gray.max() > 1.0 else (0.0, 1.0)
        hist, _ = np.histogram(np.clip(gray, *hist_range), bins=self.config.bins, range=hist_range)
        hist = hist.astype(np.float32)
        total = hist.sum()
        return hist / total if total > 0 else hist

    @staticmethod
    def _diff(hist1: np.ndarray, hist2: np.ndarray) -> float:
        return float(np.abs(hist1 - hist2).sum() * 0.5)

    def _small_gray(self, frame, height=None, width=None) -> np.ndarray:
        gray = self._to_gray(frame)
        gray = np.nan_to_num(gray, copy=False)
        if gray.max(initial=0) > 1.0:
            gray = gray / 255.0
        size = max(8, int(self.config.size))
        height = height or size
        width = width or size
        if gray.shape[0] == height and gray.shape[1] == width:
            return gray.astype(np.float32)
        ys = np.linspace(0, gray.shape[0] - 1, height).round().astype(int)
        xs = np.linspace(0, gray.shape[1] - 1, width).round().astype(int)
        return gray[np.ix_(ys, xs)].astype(np.float32)

    def _dct_matrix(self, n: int) -> np.ndarray:
        if n not in self._dct_cache:
            x = np.arange(n)
            u = np.arange(n)[:, None]
            mat = np.cos(np.pi * (2 * x + 1) * u / (2 * n)).astype(np.float32)
            mat[0] *= np.sqrt(1.0 / n)
            mat[1:] *= np.sqrt(2.0 / n)
            self._dct_cache[n] = mat
        return self._dct_cache[n]

    def _hash(self, frame, method: str) -> np.ndarray:
        if method == "ahash":
            gray = self._small_gray(frame, 8, 8)
            return gray > gray.mean()
        if method == "dhash":
            gray = self._small_gray(frame, 8, 9)
            return gray[:, 1:] > gray[:, :-1]
        gray = self._small_gray(frame, 32, 32)
        dct = self._dct_matrix(32) @ gray @ self._dct_matrix(32).T
        vals = dct[:8, :8].reshape(-1)[1:]
        return vals > np.median(vals)

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.count_nonzero(a != b)) / float(a.size)

    @staticmethod
    def _ssim_distance(a: np.ndarray, b: np.ndarray) -> float:
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_a, mu_b = float(a.mean()), float(b.mean())
        va, vb = float(a.var()), float(b.var())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        denom = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
        if denom <= 0:
            return 0.0
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom
        return float(max(0.0, min(2.0, 1.0 - ssim)))

    def _lite_flow_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        block, search = 4, 2
        h, w = curr.shape
        scores = []
        for y in range(0, h - block + 1, block):
            for x in range(0, w - block + 1, block):
                patch = curr[y : y + block, x : x + block]
                best, best_disp = float("inf"), 0.0
                for dy in range(-search, search + 1):
                    yy = y + dy
                    if yy < 0 or yy + block > h:
                        continue
                    for dx in range(-search, search + 1):
                        xx = x + dx
                        if xx < 0 or xx + block > w:
                            continue
                        sad = float(np.abs(patch - prev[yy : yy + block, xx : xx + block]).mean())
                        if sad < best:
                            best = sad
                            best_disp = math.sqrt(dx * dx + dy * dy)
                scores.append(best_disp * best)
        return float(np.mean(scores)) if scores else 0.0

    def _relative_keep(self, scores: list[tuple[int, float]], n: int) -> list[int]:
        drop_ratio = self.config.drop_ratio if self.config.drop_ratio is not None else self.config.threshold
        if not 0 <= drop_ratio < 1:
            raise ValueError("Frame selection drop ratio requires value in [0, 1)")
        keep_count = int(round(n * (1 - drop_ratio)))
        keep_count = min(max(keep_count, self.config.min_keep_frames, 1), n)
        ranked = [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]
        keep_indices = []
        for idx in [0] + ranked:
            if idx not in keep_indices:
                keep_indices.append(idx)
            if len(keep_indices) == keep_count:
                break
        return sorted(keep_indices)

    def _select_by_method(self, frames: list, method: str) -> list[int]:
        n = len(frames)
        if method in ("ahash", "dhash", "phash"):
            hashes = [self._hash(frame, method) for frame in frames]
            scores = [(0, float("inf"))]
            scores.extend((idx, self._hamming(hashes[idx], hashes[idx - 1])) for idx in range(1, n))
            return self._relative_keep(scores, n)

        grays = [self._small_gray(frame) for frame in frames]
        scores = [(0, float("inf"))]
        diffs = []
        for idx in range(1, n):
            if method == "ssim":
                score = self._ssim_distance(grays[idx - 1], grays[idx])
            elif method == "lite_flow":
                score = self._lite_flow_score(grays[idx - 1], grays[idx])
            else:
                score = float(np.abs(grays[idx] - grays[idx - 1]).mean())
            diffs.append(score)
            scores.append((idx, score))

        if method == "scene_change" and diffs:
            threshold = float(np.mean(diffs) + np.std(diffs))
            boost = max(diffs)
            scores = [(idx, score + (boost if score >= threshold else 0.0)) for idx, score in scores]
        return self._relative_keep(scores, n)

    def select_frames(self, frames: list) -> list[int]:
        n = len(frames)
        if n <= 2 or n <= self.config.min_keep_frames:
            return list(range(n))

        method = str(getattr(self.config, "method", "histogram")).lower()
        if method in {"ahash", "dhash", "phash", "ssim", "frame_diff", "lite_flow", "scene_change"}:
            return self._select_by_method(frames, method)

        histograms = [self._histogram(frame) for frame in frames]
        if self.config.relative:
            if not 0 <= self.config.threshold < 1:
                raise ValueError("Relative histogram frame selection requires threshold in [0, 1)")
            scores = [(0, float("inf"))]
            for idx in range(1, n):
                scores.append((idx, self._diff(histograms[idx], histograms[idx - 1])))

            keep_count = int(round(n * (1 - self.config.threshold)))
            keep_count = min(max(keep_count, self.config.min_keep_frames), n)
            keep_indices = [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:keep_count]]
            return sorted(keep_indices)

        keep_indices = [0]
        last_keep = 0

        for idx in range(1, n - 1):
            d_adj = self._diff(histograms[idx], histograms[idx - 1])
            d_ref = self._diff(histograms[idx], histograms[last_keep])
            keep = max(d_adj, d_ref) >= self.config.threshold
            if self.config.max_interval > 0 and idx - last_keep >= self.config.max_interval:
                keep = True
            if keep:
                keep_indices.append(idx)
                last_keep = idx

        keep_indices.append(n - 1)
        min_keep = min(max(self.config.min_keep_frames, 1), n)
        if len(keep_indices) < min_keep:
            keep_indices.extend(np.linspace(0, n - 1, min_keep).round().astype(int).tolist())
            keep_indices = sorted(set(keep_indices))
        return keep_indices


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3VLVideoProcessorInitKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    frame_selector_config: dict


@add_start_docstrings(
    "Constructs a fast Qwen3-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
class Qwen3VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = Qwen3VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, frame_selector_config: Optional[dict] = None, **kwargs: Unpack[Qwen3VLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        self.frame_selector = HistogramFrameSelector(self._build_frame_selector_config(frame_selector_config))

    @staticmethod
    def _build_frame_selector_config(frame_selector_config: Optional[dict]) -> HistogramFrameSelectorConfig:
        if frame_selector_config is None:
            return HistogramFrameSelectorConfig()
        return HistogramFrameSelectorConfig(**dict(frame_selector_config))

    def _frame_selector_config_to_dict(self) -> dict:
        if hasattr(self, "frame_selector") and self.frame_selector is not None:
            return asdict(self.frame_selector.config)
        return asdict(HistogramFrameSelectorConfig())

    def to_dict(self):
        output = super().to_dict()
        output.pop("frame_selector", None)
        output["frame_selector_config"] = self._frame_selector_config_to_dict()
        return output

    def _decode_and_sample_videos(
        self,
        videos,
        video_metadata,
        do_sample_frames=None,
        sample_indices_fn=None,
    ):
        if not self.frame_selector.config.enabled:
            return super()._decode_and_sample_videos(videos, video_metadata, do_sample_frames, sample_indices_fn)

        from ...video_utils import is_valid_video, make_batched_metadata, make_batched_videos

        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, video_metadata=video_metadata)
        if sample_indices_fn is None:
            def sample_indices_fn(*sample_args, **sample_kwargs):
                return self.sample_frames(*sample_args, use_frame_selector=False, **sample_kwargs)
        else:
            original_sample_indices_fn = sample_indices_fn

            def sample_indices_fn(*sample_args, **sample_kwargs):
                sample_kwargs["use_frame_selector"] = False
                return original_sample_indices_fn(*sample_args, **sample_kwargs)

        if is_valid_video(videos[0]) and do_sample_frames:
            sampled_videos = []
            sampled_metadata = []
            for video, metadata in zip(videos, video_metadata):
                indices = sample_indices_fn(metadata=metadata, video_frames=list(video))
                metadata.frames_indices = indices
                sampled_videos.append(video[indices])
                sampled_metadata.append(metadata)
            videos = sampled_videos
            video_metadata = sampled_metadata
        elif not is_valid_video(videos[0]):
            if isinstance(videos[0], list):
                videos = [
                    torch.stack([tvF.pil_to_tensor(image) for image in images], dim=0)
                    for images in self.fetch_images(videos)
                ]
                if do_sample_frames:
                    raise ValueError("Sampling frames from a list of images is not supported! Set `do_sample_frames=False`.")
            else:
                videos, video_metadata = self.fetch_videos(videos, sample_indices_fn=sample_indices_fn)

        return videos, video_metadata

    def _standardize_kwargs(self, **kwargs) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        video_frames: list | None = None,
        use_frame_selector: bool = True,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        fps = fps if fps is not None else self.fps

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)
        if use_frame_selector and self.frame_selector.config.enabled and video_frames is not None:
            sampled_frames = [video_frames[i] for i in indices]
            keep_indices = self.frame_selector.select_frames(sampled_frames)
            indices = indices[keep_indices]

        return indices

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        do_sample_frames = kwargs.get("do_sample_frames", self.do_sample_frames)
        if do_sample_frames is None:
            do_sample_frames = self.do_sample_frames
        if self.frame_selector.config.enabled and do_sample_frames:
            resized_videos = [video[self.frame_selector.select_frames(list(video))] for video in resized_videos]

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            T = patches.shape[1]
            if pad := -T % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen3VLVideoProcessor"]
