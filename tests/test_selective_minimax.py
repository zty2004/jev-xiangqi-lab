import unittest

from jev_engine.selective_minimax import SelectiveMinimaxSearch
from jev_engine.xiangqi import legal_moves, move_name, parse_fen, play_move, position_key


class FakeEvaluator:
    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def evaluate_batch(self, positions, move_batches):
        self.calls.extend(position_key(position) for position in positions)
        output = []
        for position, moves in zip(positions, move_batches):
            names = [move_name(move) for move in moves]
            weights = [0.001] * len(moves)
            for preferred, weight in (("b2e2", 0.9), ("h2e2", 0.1),
                                      ("b7e7", 0.9), ("h7e7", 0.1)):
                if preferred in names:
                    weights[names.index(preferred)] = weight
            output.append((weights, self.values.get(position_key(position), 0.0)))
        return output


class SelectiveMinimaxTests(unittest.TestCase):
    def test_leaf_value_can_overrule_root_policy(self):
        root = parse_fen()
        bad = play_move(root, "b2e2")
        good = play_move(root, "h2e2")
        evaluator = FakeEvaluator({position_key(bad): 0.8, position_key(good): -0.2})
        result = SelectiveMinimaxSearch(evaluator, internal_width=1).run(root, 1)
        self.assertEqual(result.move, "h2e2")
        self.assertEqual(result.maximum_depth, 1)
        self.assertEqual(result.model_evaluations, 1 + len(legal_moves(root)))

    def test_root_prior_can_guard_against_small_value_reordering(self):
        root = parse_fen()
        bad = play_move(root, "b2e2")
        good = play_move(root, "h2e2")
        evaluator = FakeEvaluator({position_key(bad): 0.1, position_key(good): -0.1})
        result = SelectiveMinimaxSearch(
            evaluator, internal_width=1, root_prior_weight=0.5).run(root, 1)
        self.assertEqual(result.move, "b2e2")

    def test_policy_is_reapplied_at_every_internal_node(self):
        root = parse_fen()
        evaluator = FakeEvaluator()
        result = SelectiveMinimaxSearch(evaluator, internal_width=1).run(root, 2)
        root_moves = len(legal_moves(root))
        self.assertEqual(result.maximum_depth, 2)
        self.assertEqual(result.model_evaluations, 1 + root_moves + root_moves)
        self.assertEqual(len(evaluator.calls), result.model_evaluations)


if __name__ == "__main__":
    unittest.main()
