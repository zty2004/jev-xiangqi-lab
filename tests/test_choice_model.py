"""Regression checks for data isolation and choice probability reporting."""

import importlib.util
import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / "train" / "choice_model.py"
spec = importlib.util.spec_from_file_location("choice_model", MODULE)
choice_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choice_model)


class ChoiceModelTests(unittest.TestCase):
    def test_multiple_teacher_files_keep_game_ids_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for source in range(2):
                filename = Path(directory) / f"source-{source}.jsonl"
                with filename.open("w", encoding="utf-8") as handle:
                    for game in range(4):
                        handle.write(json.dumps({"kind": "position", "game": game,
                                                 "fen": f"source-{source}-game-{game} w",
                                                 "legal": ["a0a1"], "best": "a0a1"}) + "\n")
                files.append(filename)
            rows = choice_model.load_teacher_sources(files)
            self.assertEqual(len({row["game"] for row in rows}), 8)
            self.assertEqual(len(rows), 8)

    def test_splits_keep_games_and_positions_disjoint(self):
        rows = [{"game": game, "fen": f"unique-{game} w", "legal": ["a0a1"], "best": "a0a1"}
                for game in range(20)]
        rows.extend({"game": game, "fen": "repeated w", "legal": ["a0a1"], "best": "a0a1"}
                    for game in range(20))
        splits = choice_model.split_rows(rows, 7)
        self.assertEqual(len(splits), 4)
        self.assertTrue(all(splits))
        self.assertEqual(sum(row["fen"] == "repeated w" for split in splits for row in split), 1)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse({row["game"] for row in splits[left]} & {row["game"] for row in splits[right]})
                self.assertFalse({row["fen"] for row in splits[left]} & {row["fen"] for row in splits[right]})

    def test_multiclass_scores_for_uniform_binary_choice(self):
        predictions = [(torch.tensor([0.0, 0.0]), 0), (torch.tensor([0.0, 0.0]), 1)]
        metrics = choice_model.policy_metrics(predictions, 1.0)
        self.assertAlmostEqual(metrics["nll"], math.log(2))
        self.assertAlmostEqual(metrics["brier"], 0.5)
        self.assertAlmostEqual(metrics["mean_top_probability"], 0.5)
        self.assertAlmostEqual(metrics["ece10"], 0.0)

    def test_temperature_reduces_overconfidence(self):
        predictions = [(torch.tensor([4.0, 0.0]), best) for best in [0, 1, 0, 1]]
        temperature = choice_model.fit_temperature(predictions)
        self.assertGreater(temperature, 1.0)
        self.assertLess(choice_model.policy_metrics(predictions, temperature)["nll"],
                        choice_model.policy_metrics(predictions, 1.0)["nll"])


if __name__ == "__main__":
    unittest.main()
