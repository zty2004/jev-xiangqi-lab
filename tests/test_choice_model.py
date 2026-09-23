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
    def test_history_planes_and_outcome_labels_use_side_to_move(self):
        fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 30 1"
        previous = ["rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"]
        planes = choice_model.encode_position(fen, previous, 2, 46)
        self.assertEqual(tuple(planes.shape), (46, 10, 9))
        self.assertTrue(torch.equal(planes[:14], planes[16:30]))
        self.assertAlmostEqual(float(planes[44, 0, 0]), 0.25)
        self.assertAlmostEqual(float(planes[45, 0, 0]), 2 / 3)
        row = {"fen": fen, "legal": ["a9a8", "b7b8"], "best": "a9a8",
               "candidates": [], "previous": previous, "repetitionCount": 2, "winner": "red"}
        loss_sample = choice_model.TeacherDataset([row], 46, "wdl", "best")[0]
        self.assertEqual(int(loss_sample[3]), 0)
        self.assertEqual(float(loss_sample[5]), 1)
        self.assertEqual(float(loss_sample[2][0]), 1)
        row["winner"] = "black"
        self.assertEqual(int(choice_model.TeacherDataset([row], 46, "wdl")[0][3]), 2)
        row["winner"] = "draw"
        self.assertEqual(int(choice_model.TeacherDataset([row], 46, "wdl")[0][3]), 1)
        row["winner"] = None
        self.assertEqual(float(choice_model.TeacherDataset([row], 46, "wdl")[0][5]), 0)

    def test_history_labels_require_exact_teacher_hash_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "teacher.jsonl"
            labels = Path(directory) / "labels.jsonl"
            row = {"kind": "position", "game": 0, "ply": 8, "fen": "board w",
                   "legal": ["a0a1"], "best": "a0a1"}
            teacher.write_text(json.dumps(row) + "\n", encoding="utf-8")
            meta = {"kind": "meta", "schema": "history-labels-v1",
                    "teacherSha256": choice_model.sha256_file(teacher)}
            label = {"kind": "history-label", "game": 0, "ply": 8,
                     "previous": ["prior b"], "repetitionCount": 1, "winner": "draw"}
            labels.write_text(json.dumps(meta) + "\n" + json.dumps(label) + "\n", encoding="utf-8")
            enriched = choice_model.attach_history_labels([row], labels, teacher)
            self.assertEqual(enriched[0]["winner"], "draw")
            teacher.write_text(json.dumps({**row, "best": "a0a2"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                choice_model.attach_history_labels([row], labels, teacher)

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
