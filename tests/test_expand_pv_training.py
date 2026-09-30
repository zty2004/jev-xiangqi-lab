import unittest

from train.expand_pv_training import PV_SOURCE, continuation_rows, principal_variation


class FakeHelper:
    def set_position(self, command):
        self.command = command

    def fen(self):
        return "4k4/9/9/9/9/9/9/9/9/4K4 b - - 1 1"

    def legal_moves(self):
        return ["h9g7", "b9c7"]


class ExpandPvTrainingTests(unittest.TestCase):
    def test_prefers_pv_attached_to_final_best(self):
        row = {"best": "b2e2", "candidates": [
            {"rank": 1, "move": "h2e2", "pv": ["h2e2", "h9g7"]},
            {"rank": 2, "move": "b2e2", "pv": ["b2e2", "b9c7"]},
        ]}
        self.assertEqual(principal_variation(row), ["b2e2", "b9c7"])

    def test_continuation_keeps_game_split_and_has_lower_training_weight(self):
        row = {"game": 7, "ply": 20, "fen": "root w - - 0 1", "best": "b2e2",
               "sourceIndex": 9, "candidates": [
                   {"rank": 1, "move": "b2e2", "pv": ["b2e2", "h9g7"]}]}
        rows = continuation_rows(row, FakeHelper(), 4, 0.15)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game"], 7)
        self.assertEqual(rows[0]["best"], "h9g7")
        self.assertEqual(rows[0]["source"], PV_SOURCE)
        self.assertEqual(rows[0]["trainingWeight"], 0.15)


if __name__ == "__main__":
    unittest.main()
