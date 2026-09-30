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
    def test_game_phase_matches_engine_thresholds(self):
        self.assertEqual(choice_model.game_phase("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"), "opening")
        self.assertEqual(choice_model.game_phase("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 20"), "middlegame")
        self.assertEqual(choice_model.game_phase("4k4/9/9/9/4p4/9/9/9/9/R3K4 w - - 0 40"), "endgame")

    def test_source_index_recovers_prefixed_multi_source_games(self):
        self.assertEqual(choice_model.source_index({"game": "3:game-7"}), 3)
        self.assertEqual(choice_model.source_index({"game": 12}), 0)

    def test_v1_checkpoint_migrates_to_v2_without_changing_logits(self):
        torch.manual_seed(7)
        original = choice_model.ChoiceNet(16, 1, policy_features="v1")
        checkpoint = {"state_dict": original.state_dict(), "policy_features": "v1"}
        migrated = choice_model.ChoiceNet(16, 1, policy_features="v2")
        choice_model.initialize_from_checkpoint(migrated, checkpoint)
        boards = torch.randn(2, 16, 10, 9)
        moves = torch.tensor([[[0, 1], [20, 29]], [[89, 80], [44, 36]]])
        mask = torch.ones((2, 2), dtype=torch.bool)
        original.eval()
        migrated.eval()
        with torch.no_grad():
            old_logits, old_value = original(boards, moves, mask)
            new_logits, new_value = migrated(boards, moves, mask)
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertTrue(torch.equal(old_value, new_value))

    def test_attention_policy_scores_only_requested_legal_moves(self):
        model = choice_model.ChoiceNet(32, 2, policy_features="attention")
        boards = torch.randn(2, 16, 10, 9)
        moves = torch.tensor([[[0, 1], [20, 29], [0, 0]], [[89, 80], [44, 36], [10, 19]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        logits, values = model(boards, moves, mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(float(logits[0, 2].detach()), -1e9)
        self.assertTrue(torch.isfinite(logits[mask]).all())

    def test_policy_planes_cover_legal_xiangqi_displacements(self):
        self.assertEqual(len(choice_model.MOVE_DELTAS), 50)
        model = choice_model.ChoiceNet(32, 2, policy_features="planes")
        boards = torch.randn(2, 16, 10, 9)
        moves = torch.tensor([[[0, 8], [20, 39], [0, 0]], [[89, 80], [44, 25], [10, 21]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        logits, values = model(boards, moves, mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(float(logits[0, 2].detach()), -1e9)
        for source, target in moves[mask].tolist():
            self.assertGreaterEqual(choice_model.move_plane_index(source, target), 0)

    def test_horizontal_mirror_is_an_involution_for_fen_and_moves(self):
        fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        self.assertEqual(choice_model.mirror_fen(choice_model.mirror_fen(fen)), fen)
        self.assertEqual(choice_model.mirror_move("b2e2"), "h2e2")
        self.assertEqual(choice_model.mirror_move(choice_model.mirror_move("b2e2")), "b2e2")

    def test_mirror_augmentation_doubles_training_examples_without_changing_targets(self):
        row = {"fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/4C2C1/9/RNBAKABNR w - - 0 1",
               "legal": ["b2e2", "h2e2"], "best": "b2e2", "source": "opening-book",
               "policy": {"b2e2": 0.75, "h2e2": 0.25}}
        dataset = choice_model.TeacherDataset([row], mirror_augmentation=True)
        self.assertEqual(len(dataset), 2)
        original, mirrored = dataset[0], dataset[1]
        self.assertTrue(torch.equal(original[2], mirrored[2]))
        self.assertFalse(torch.equal(original[0], mirrored[0]))

    def test_dominant_teacher_target_keeps_final_best_above_stale_multipv(self):
        row = {"fen": "board w", "legal": ["a0a1", "a0a2"], "best": "a0a2",
               "candidates": [{"move": "a0a1", "score": 100, "scoreType": "cp"}]}
        soft, _ = choice_model.target_distribution(row)
        dominant, _ = choice_model.target_distribution(row, 0.7)
        self.assertAlmostEqual(float(soft[1]), 0.3)
        self.assertAlmostEqual(float(dominant[1]), 0.7)
        self.assertAlmostEqual(float(dominant.sum()), 1)

    def test_selfplay_target_uses_recorded_search_distribution(self):
        row = {"fen": "board w", "legal": ["a0a1", "a0a2", "b0c2"], "best": "a0a1", "value": 350,
               "searchPolicy": [{"move": "a0a1", "probability": 0.6},
                                {"move": "b0c2", "probability": 0.4}],
               "candidates": [{"move": "a0a1", "score": 999}]}
        target, value = choice_model.target_distribution(row, 0.7)
        self.assertTrue(torch.allclose(target, torch.tensor([0.6, 0.0, 0.4])))
        self.assertAlmostEqual(value, math.tanh(0.5))

    def test_selfplay_target_rejects_duplicate_moves(self):
        row = {"legal": ["a0a1"], "best": "a0a1", "searchPolicy": [
            {"move": "a0a1", "probability": 0.5}, {"move": "a0a1", "probability": 0.5}]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            choice_model.target_distribution(row)

    def test_untrained_wdl_head_is_not_exposed_as_a_probability(self):
        model = choice_model.ChoiceNet(16, 1, 46, "wdl", 0.0)
        fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        response = choice_model.rank(model, fen, ["b2e2", "h2e2"], torch.device("cpu"))
        self.assertNotIn("wdl", response)
        self.assertNotIn("value", response)
        self.assertAlmostEqual(sum(item["probability"] for item in response["choices"]), 1, delta=1e-6)

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

    def test_source_sampling_weights_leave_opening_book_and_other_sources_at_one(self):
        rows = [{"game": "0:1"}, {"game": "1:2"}, {"game": "2:3"},
                {"game": "opening", "source": "opening-book"}]
        self.assertEqual(choice_model.source_sample_weights(rows, {2: 4}), [1.0, 1.0, 4, 1.0])

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
        self.assertAlmostEqual(metrics["top1"], 0.5)
        self.assertAlmostEqual(metrics["top3"], 1.0)
        self.assertAlmostEqual(metrics["top5"], 1.0)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 0.75)

    def test_temperature_reduces_overconfidence(self):
        predictions = [(torch.tensor([4.0, 0.0]), best) for best in [0, 1, 0, 1]]
        temperature = choice_model.fit_temperature(predictions)
        self.assertGreater(temperature, 1.0)
        self.assertLess(choice_model.policy_metrics(predictions, temperature)["nll"],
                        choice_model.policy_metrics(predictions, 1.0)["nll"])


if __name__ == "__main__":
    unittest.main()
