import unittest

from jev_engine.policy_search import PuctSearch
from jev_engine.xiangqi import move_name, parse_fen


class FakeEvaluator:
    def __init__(self):
        self.positions = 0

    def evaluate_batch(self, positions, move_batches):
        self.positions += len(positions)
        results = []
        for position, moves in zip(positions, move_batches):
            weights = [0.001] * len(moves)
            if position.fullmove == 1 and position.side == "red":
                preferred = next(index for index, move in enumerate(moves)
                                 if move_name(move) == "b2e2")
                weights[preferred] = 1.0
            results.append((weights, 0.0))
        return results


class PolicySearchTests(unittest.TestCase):
    def test_policy_guides_root_and_is_reapplied_below_root(self):
        evaluator = FakeEvaluator()
        result = PuctSearch(evaluator, batch_size=8, maximum_depth=8).run(parse_fen(), 96)
        self.assertEqual(result.move, "b2e2")
        self.assertGreater(result.maximum_depth, 1)
        self.assertGreater(result.model_evaluations, len(result.policy))
        self.assertEqual(result.model_evaluations, evaluator.positions)
        self.assertEqual(result.visits, 96)
        self.assertEqual(len(result.policy), 44)
        self.assertAlmostEqual(sum(item[2] for item in result.policy), 1.0)

    def test_terminal_position_returns_without_model_call(self):
        evaluator = FakeEvaluator()
        position = parse_fen("4k4/9/9/9/9/9/9/9/3rrr3/4K4 w - - 0 1")
        result = PuctSearch(evaluator).run(position, 8)
        self.assertIsNone(result.move)
        self.assertEqual(result.value, -1.0)
        self.assertEqual(result.model_evaluations, 0)

    def test_depth_limit_evaluates_but_does_not_expand_frontier(self):
        evaluator = FakeEvaluator()
        result = PuctSearch(evaluator, batch_size=8, maximum_depth=1).run(parse_fen(), 32)
        self.assertEqual(result.maximum_depth, 1)
        self.assertEqual(result.expanded_nodes, 1)
        self.assertGreater(result.model_evaluations, 1)


if __name__ == "__main__":
    unittest.main()
