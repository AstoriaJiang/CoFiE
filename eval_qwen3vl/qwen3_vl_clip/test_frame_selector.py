import tempfile
import unittest

import numpy as np
import torch

from qwen3_vl_clip.frame_selector import CLIPPatchFrameSelector, CLIPPatchFrameSelectorConfig
from qwen3_vl_clip.summarize_latency import summarize_records
from qwen3_vl_clip.streamingbench import _filter_videos, _sample_subset


class CLIPPatchFrameSelectorTest(unittest.TestCase):
    def test_aligned_patch_mean_cosine(self):
        embeddings = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [0.0, -1.0]],
            ]
        )
        actual = CLIPPatchFrameSelector.adjacent_patch_similarities(embeddings)
        torch.testing.assert_close(actual, torch.tensor([0.0, 0.5]))

    def test_relative_selection_keeps_largest_changes(self):
        selector = CLIPPatchFrameSelector(
            CLIPPatchFrameSelectorConfig(drop_ratio=0.5, min_keep_frames=1)
        )
        keep = selector._relative_keep(torch.tensor([0.9, 0.1, 0.5]), n=4)
        self.assertEqual(keep, [0, 2])

    def test_threshold_selection_obeys_interval_and_endpoints(self):
        selector = CLIPPatchFrameSelector(
            CLIPPatchFrameSelectorConfig(
                strategy="threshold", similarity_threshold=0.8, min_keep_frames=1, max_interval=2
            )
        )
        embeddings = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        )
        self.assertEqual(selector._threshold_keep(embeddings), [0, 2, 3, 4])

    def test_frame_conversion_accepts_chw_float(self):
        frame = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        converted = CLIPPatchFrameSelector._frame_to_hwc_uint8(frame)
        self.assertEqual(converted.shape, (2, 2, 3))
        np.testing.assert_array_equal(converted[0, 0], np.array([0, 127, 255], dtype=np.uint8))

    def test_stats_jsonl_for_short_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/stats.jsonl"
            selector = CLIPPatchFrameSelector(
                CLIPPatchFrameSelectorConfig(min_keep_frames=4, stats_path=path)
            )
            self.assertEqual(selector.select_frames([1, 2, 3]), [0, 1, 2])
            self.assertEqual(selector.get_stats()["frames_in"], 3)

    def test_latency_summary(self):
        summary = summarize_records(
            [
                {"frames_in": 10, "frames_out": 8, "total_seconds": 1.0, "preprocess_seconds": 0.2, "forward_seconds": 0.7, "selection_seconds": 0.1},
                {"frames_in": 20, "frames_out": 10, "total_seconds": 3.0, "preprocess_seconds": 0.5, "forward_seconds": 2.2, "selection_seconds": 0.3},
            ]
        )
        self.assertEqual(summary["videos"], 2)
        self.assertEqual(summary["keep_ratio"], 0.6)
        self.assertEqual(summary["clip_filter_mean_seconds_per_video"], 2.0)

    def test_streamingbench_filters_predecoded_frames_once(self):
        selector = CLIPPatchFrameSelector(CLIPPatchFrameSelectorConfig(min_keep_frames=1))
        selector.select_frames = lambda frames: [0, len(frames) - 1]
        video = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
        filtered = _filter_videos([video], selector)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].shape[0], 2)
        np.testing.assert_array_equal(filtered[0][0], video[0])
        np.testing.assert_array_equal(filtered[0][1], video[-1])

    def test_streamingbench_subset_is_stratified_and_deterministic(self):
        rows = [
            {"question_id": f"a-{index}", "task_type": "A"}
            for index in range(10)
        ] + [
            {"question_id": f"b-{index}", "task_type": "B"}
            for index in range(20)
        ]
        task_df = __import__("pandas").DataFrame(rows)
        first = _sample_subset(task_df, 0.1, 42)
        second = _sample_subset(task_df, 0.1, 42)
        self.assertEqual(len(first), 3)
        self.assertEqual(first["task_type"].value_counts().to_dict(), {"B": 2, "A": 1})
        self.assertEqual(first["question_id"].tolist(), second["question_id"].tolist())


if __name__ == "__main__":
    unittest.main()
