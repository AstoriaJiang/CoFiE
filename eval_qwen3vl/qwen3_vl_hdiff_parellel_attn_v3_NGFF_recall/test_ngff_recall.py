import importlib.util
import os
import sys
import unittest

import torch


HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT_PATH = os.path.join(
    REPO_ROOT,
    "eval_qwen3vl",
    "scripts",
    "qwen3_vl_hdiff_parellel_attn_v3_NGFF_recall",
    "streamingbench_ngff_recall.py",
)
spec = importlib.util.spec_from_file_location("streamingbench_ngff_recall_tested", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) + 100 for char in text]}

    def decode(self, token_ids):
        return "".join(chr(token - 100) for token in token_ids)


class NGFFRecallMetricTest(unittest.TestCase):
    def test_question_options_mask_excludes_markers_and_answer_cue(self):
        tokenizer = CharacterTokenizer()
        video_token_id = 9
        suffix = (
            "Question: Which animal appears?\n\n"
            "Options:\nA. Cat\nB. Dog\nC. Bird\nD. Fish\n\n"
            "The best option is:"
        )
        ids = [1, video_token_id, video_token_id, 2] + tokenizer(suffix)["input_ids"]
        input_ids = torch.tensor([ids])
        mask = module.build_question_options_mask(tokenizer, input_ids, video_token_id)
        selected = "".join(chr(token - 100) for token in input_ids[0][mask[0]].tolist())
        self.assertIn("Which animal appears?", selected)
        self.assertIn("A. Cat", selected)
        self.assertIn("D. Fish", selected)
        self.assertNotIn("Question:", selected)
        self.assertNotIn("Options:", selected)
        self.assertNotIn("The best option is:", selected)
        self.assertFalse(bool((mask & (input_ids == video_token_id)).any()))

    def test_complete_temporal_unit_preservation(self):
        attention = [0.5, 0.3, 0.2]
        members = module.temporal_unit_members(6, 2)
        metrics = module.compute_recall_metrics(
            attention,
            members,
            keep_indices=[0, 1, 4],
            top_ratio=0.20,
        )
        self.assertEqual(members, [[0, 1], [2, 3], [4, 5]])
        self.assertAlmostEqual(metrics["attention_preservation"], 0.5)
        self.assertAlmostEqual(metrics["qer_top_evidence_recall"], 1.0)
        self.assertEqual(metrics["top_count"], 1)

    def test_partial_top_unit_is_not_recalled(self):
        metrics = module.compute_recall_metrics(
            [0.1, 0.7, 0.2],
            module.temporal_unit_members(6, 2),
            keep_indices=[0, 1, 2, 4, 5],
            top_ratio=0.20,
        )
        self.assertAlmostEqual(metrics["attention_preservation"], 0.3)
        self.assertAlmostEqual(metrics["qer_top_evidence_recall"], 0.0)

    def test_odd_last_unit_uses_only_real_frame(self):
        members = module.temporal_unit_members(5, 2)
        metrics = module.compute_recall_metrics(
            [0.2, 0.3, 0.5], members, keep_indices=[0, 1, 4], top_ratio=0.20
        )
        self.assertEqual(members, [[0, 1], [2, 3], [4]])
        self.assertAlmostEqual(metrics["attention_preservation"], 0.7)
        self.assertAlmostEqual(metrics["qer_top_evidence_recall"], 1.0)

    def test_top_twenty_percent_rounds_up_and_breaks_ties_by_time(self):
        attention = [0.2] * 5
        metrics = module.compute_recall_metrics(
            attention,
            module.temporal_unit_members(10, 2),
            keep_indices=[0, 1],
            top_ratio=0.20,
        )
        self.assertEqual(metrics["top_count"], 1)
        self.assertEqual(metrics["top_indices"], [0])
        self.assertEqual(metrics["qer_top_evidence_recall"], 1.0)

    def test_no_drop_is_perfect(self):
        attention = [0.4, 0.35, 0.25]
        metrics = module.compute_recall_metrics(
            attention,
            module.temporal_unit_members(6, 2),
            keep_indices=list(range(6)),
        )
        self.assertAlmostEqual(metrics["attention_preservation"], 1.0)
        self.assertAlmostEqual(metrics["qer_top_evidence_recall"], 1.0)

    def test_uniform_and_random_keep_exact_budget(self):
        uniform = module.uniform_keep_indices(11, 6)
        random_a = module.random_keep_indices(11, 6, 42, 7, "0.50", 0)
        random_b = module.random_keep_indices(11, 6, 42, 7, "0.50", 0)
        self.assertEqual(len(uniform), 6)
        self.assertEqual(len(set(uniform)), 6)
        self.assertEqual(len(random_a), 6)
        self.assertEqual(len(set(random_a)), 6)
        self.assertEqual(random_a, random_b)


if __name__ == "__main__":
    unittest.main()
