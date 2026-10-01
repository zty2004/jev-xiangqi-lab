import json
import tempfile
import unittest
from pathlib import Path

from jev_engine.xiangqi import START_FEN
from train.sample_tree_distillation import load_positions, sample_tree_rows


class SampleTreeDistillationTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "kind": "position", "game": "fixture", "ply": 0, "fen": START_FEN,
            "candidates": [
                {"rank": 2, "move": "h2e2", "pv": ["h2e2", "h7e7"]},
                {"rank": 1, "move": "b2e2", "pv": ["b2e2", "b7e7"]},
            ],
        }

    def test_samples_best_and_nonbest_internal_branches(self):
        rows = sample_tree_rows([self.row], roots=1, candidates=2, plies=2, seed=7)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["treeCandidateRank"] for row in rows], [1, 1, 2, 2])
        self.assertEqual(rows[0]["treePrefix"], ["b2e2"])
        self.assertEqual(rows[1]["treePrefix"], ["b2e2", "b7e7"])
        self.assertTrue(all(row["best"] in row["legal"] for row in rows))

    def test_deduplicates_shared_positions(self):
        duplicate = {**self.row, "candidates": [self.row["candidates"][1]]}
        rows = sample_tree_rows([self.row, duplicate], roots=2, candidates=1, plies=2, seed=3)
        self.assertEqual(len(rows), 2)

    def test_load_positions_filters_phase_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            path.write_text("\n".join((json.dumps({"kind": "meta"}), json.dumps(self.row))) + "\n")
            self.assertEqual(load_positions(str(path), "opening"), [self.row])
            with self.assertRaises(ValueError):
                load_positions(str(path), "endgame")

    def test_load_positions_rejects_rows_without_variations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            row = {**self.row, "candidates": [{"rank": 1, "move": "b2e2"}]}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_positions(str(path), None)


if __name__ == "__main__":
    unittest.main()
