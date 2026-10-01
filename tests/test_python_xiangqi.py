import unittest

from jev_engine.xiangqi import (START_FEN, legal_moves, move_name, parse_fen,
                                perft, play_move, to_fen)


class PythonXiangqiTests(unittest.TestCase):
    def test_fen_round_trip_and_opening_move(self):
        position = parse_fen()
        self.assertEqual(to_fen(position), START_FEN)
        self.assertEqual(len(legal_moves(position)), 44)
        next_position = play_move(position, "b2e2")
        self.assertEqual(next_position.side, "black")
        self.assertIn("b7e7", {move_name(move) for move in legal_moves(next_position)})

    def test_perft_matches_independent_engine_fixtures(self):
        cases = [
            (START_FEN, (44, 1920, 79666)),
            ("r1ea1a3/4kh3/2h1e4/pHp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2EAKAE2 w - - 0 1",
             (38, 1128, 43929)),
            ("1ceak4/9/h2a5/2p1p3p/5cp2/2h2H3/6PCP/3AE4/2C6/3A1K1H1 w - - 0 1",
             (7, 281, 8620)),
        ]
        for fen, expected in cases:
            position = parse_fen(fen)
            for depth, nodes in enumerate(expected, 1):
                self.assertEqual(perft(position, depth), nodes, f"{fen} depth {depth}")

    def test_facing_generals_make_sideways_rook_move_illegal(self):
        position = parse_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 w - - 0 1")
        moves = {move_name(move) for move in legal_moves(position)}
        self.assertNotIn("e4d4", moves)
        self.assertNotIn("e4f4", moves)


if __name__ == "__main__":
    unittest.main()
