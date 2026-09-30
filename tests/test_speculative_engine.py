import unittest

from hybrid.speculative_engine import SearchResult, SpeculativeEngine, parse_info


class FakeTarget:
    def __init__(self, final="c0e2"):
        self.final = final
        self.searches = []

    def set_position(self, position):
        self.position = position

    def fen(self):
        return "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

    def legal_moves(self):
        return ["b2e2", "c0e2", "h0g2"]

    def search(self, movetime_ms, searchmoves=None, multipv=1):
        self.searches.append((movetime_ms, searchmoves, multipv))
        return SearchResult(searchmoves[0] if searchmoves else self.final, depth=8, nodes=100)


class FakeDraft:
    def choices(self, fen, legal, history=None):
        return [{"move": move, "probability": probability} for move, probability in
                zip(["b2e2", "h0g2", "c0e2"], [0.7, 0.2, 0.1])]


class SpeculativeEngineTests(unittest.TestCase):
    def test_info_parser_keeps_target_search_evidence(self):
        info = parse_info("info depth 15 seldepth 28 multipv 1 score cp 37 nodes 12345 pv b2e2 h9g7")
        self.assertEqual((info.depth, info.selective_depth, info.score, info.nodes), (15, 28, 37, 12345))
        self.assertEqual(info.pv, ("b2e2", "h9g7"))

    def test_target_can_reject_draft_after_restricted_verification(self):
        target = FakeTarget(final="c0e2")
        result = SpeculativeEngine(target, FakeDraft(), top_k=2).choose("position startpos", 1000)
        self.assertEqual(target.searches[0][1], ["b2e2", "h0g2"])
        self.assertIsNone(target.searches[1][1])
        self.assertEqual(result.move, "c0e2")
        self.assertEqual(result.disposition, "rejected-fallback")

    def test_target_acceptance_is_reported_without_bypassing_full_search(self):
        target = FakeTarget(final="b2e2")
        result = SpeculativeEngine(target, FakeDraft(), top_k=2).choose("position startpos", 1000)
        self.assertEqual(len(target.searches), 2)
        self.assertEqual(result.disposition, "accepted-top1")


if __name__ == "__main__":
    unittest.main()
