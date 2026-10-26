import unittest
from types import SimpleNamespace

import torch

from modeling_qwen3_vl_DTD import Qwen3VLForConditionalGenerationDTD


class AttentionFrameGroupingTest(unittest.TestCase):
    def setUp(self):
        self.video_token_id = 99
        self.model_stub = SimpleNamespace(
            config=SimpleNamespace(
                video_token_id=self.video_token_id,
                vision_start_token_id=97,
                vision_end_token_id=98,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )
        )

    def parse(self, input_ids, temporal_ids, grids, strict=True):
        position_ids = torch.stack(
            [
                torch.tensor(temporal_ids),
                torch.zeros(len(temporal_ids), dtype=torch.long),
                torch.zeros(len(temporal_ids), dtype=torch.long),
            ]
        )
        return Qwen3VLForConditionalGenerationDTD._parse_text_and_video_frame_indices(
            self.model_stub,
            torch.tensor(input_ids),
            position_ids,
            torch.device("cpu"),
            expected_video_grids=torch.tensor(grids) if grids is not None else None,
            strict=strict,
        )

    def test_separate_frame_segments(self):
        # Boundary/timestamp tokens between frames must remain non-video queries.
        ids = [10, 11, 99, 99, 12, 13, 99, 99, 14]
        text, groups, validation = self.parse(
            ids,
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [[2, 2, 4]],
        )
        self.assertEqual([group.tolist() for group in groups], [[2, 3], [6, 7]])
        self.assertEqual(text.tolist(), [0, 1, 4, 5, 8])
        self.assertTrue(validation["passed"])

    def test_continuous_video_segment_split_by_temporal_id(self):
        ids = [10, 99, 99, 99, 99, 14]
        _, groups, validation = self.parse(
            ids,
            [0, 0, 0, 1, 1, 1],
            [[2, 2, 4]],
        )
        self.assertEqual([group.tolist() for group in groups], [[1, 2], [3, 4]])
        self.assertTrue(validation["passed"])

    def test_temporal_change_and_discontinuity_both_split(self):
        ids = [99, 99, 7, 99, 99, 99, 99]
        _, groups, validation = self.parse(
            ids,
            [0, 0, 0, 0, 0, 1, 1],
            [[3, 2, 4]],
        )
        self.assertEqual([group.tolist() for group in groups], [[0, 1], [3, 4], [5, 6]])
        self.assertTrue(validation["passed"])

    def test_strict_grid_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            self.parse(
                [10, 99, 99, 14],
                [0, 0, 0, 0],
                [[2, 2, 4]],
                strict=True,
            )

    def test_grid_assignment_supports_multiple_videos_and_samples(self):
        input_ids = torch.tensor(
            [
                [99, 99, 99, 99, 0, 0],
                [99, 99, 0, 0, 0, 0],
            ]
        )
        grids = torch.tensor([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
        assignments = Qwen3VLForConditionalGenerationDTD._assign_video_grids_to_samples(
            self.model_stub, input_ids, grids
        )
        self.assertEqual([len(rows) for rows in assignments], [2, 1])
        self.assertEqual(assignments[0][0].tolist(), [1, 2, 4])
        self.assertEqual(assignments[1][0].tolist(), [1, 2, 4])

    def test_attention_drop_removes_only_selected_video_frame(self):
        input_ids = torch.tensor([[10, 99, 99, 12, 99, 99, 14]])
        hidden = torch.zeros(1, 7, 4)
        cos = torch.zeros(1, 7, 4)
        sin = torch.zeros(1, 7, 4)
        position_ids = torch.zeros(3, 1, 7, dtype=torch.long)
        groups = [[torch.tensor([1, 2]), torch.tensor([4, 5])]]
        validations = [{"passed": True, "issues": []}]

        result = Qwen3VLForConditionalGenerationDTD.token_drop(
            self.model_stub,
            method="attention",
            threshold=0.0,
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            position_ids=position_ids,
            input_ids=input_ids,
            frame_weights=[[0.1, 0.9]],
            frame_token_groups=groups,
            frame_validations=validations,
            top_k_ratio=0.5,
            validation_strict=True,
        )
        keep_mask = result[-1][0]
        self.assertEqual((~keep_mask).nonzero(as_tuple=True)[0].tolist(), [1, 2])
        self.assertTrue(validations[0]["selection"]["all_dropped_positions_are_video_tokens"])
        self.assertEqual(validations[0]["selection"]["actual_video_token_drop"], 2)


if __name__ == "__main__":
    unittest.main()
