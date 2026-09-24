"""Checks shared feature semantics for the trainable NNUE value model."""

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / "train" / "nnue_value.py"
spec = importlib.util.spec_from_file_location("nnue_value", MODULE)
nnue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nnue)

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


class NnueValueTests(unittest.TestCase):
    def test_both_perspectives_have_one_feature_per_piece(self):
        self.assertEqual(len(nnue.feature_indices(START, "red")), 32)
        self.assertEqual(len(nnue.feature_indices(START, "black")), 32)
        self.assertEqual(set(nnue.feature_indices(START, "red")), set(nnue.feature_indices(START, "black")))

    def test_teacher_score_uses_final_best_candidate(self):
        row = {"best": "b2e2", "candidates": [
            {"move": "h2e2", "score": 500, "scoreType": "cp"},
            {"move": "b2e2", "score": 25, "scoreType": "cp"}]}
        self.assertEqual(nnue.teacher_score(row), 25)

    def test_network_output_is_bounded_in_centipawns(self):
        dataset = nnue.ValueDataset([{"fen": START, "score": 30}])
        indices, mask, side, _ = nnue.collate([dataset[0]])
        value = nnue.NnueValue(hidden=8, head=4, output_scale=2000)(indices, mask, side)
        self.assertEqual(tuple(value.shape), (1,))
        self.assertLessEqual(abs(float(value[0].detach())), 2000)


if __name__ == "__main__":
    unittest.main()
