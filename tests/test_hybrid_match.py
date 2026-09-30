import json
import tempfile
import unittest

from hybrid.match import load_openings, position_command, summarize


class HybridMatchTests(unittest.TestCase):
    def test_openings_are_distinct_and_reproducible(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            for game, moves in enumerate((["a0a1", "a9a8", "b0c2"],
                                          ["b2e2", "b7e7", "b0c2"],
                                          ["a0a1", "a9a8", "b0c2"])):
                handle.write(json.dumps({"kind": "game", "game": game, "opening": moves}) + "\n")
            handle.flush()
            first = load_openings(handle.name, 2, 2, 7)
            second = load_openings(handle.name, 1, 2, 7, offset=1)
        self.assertEqual(first[1:], second)
        self.assertEqual(len({tuple(item) for item in first}), 2)

    def test_summary_scores_hybrid_and_speculation(self):
        games = [
            {"hybridResult": "win", "speculation": ["accepted-top1", "corrected-within-draft"],
             "hybridMoveTimesMs": [990], "nativeMoveTimesMs": [1001]},
            {"hybridResult": "draw", "speculation": ["rejected-fallback"],
             "hybridMoveTimesMs": [995], "nativeMoveTimesMs": [1000]},
        ]
        report = summarize(games)
        self.assertEqual(report["hybridScore"], 0.75)
        self.assertAlmostEqual(report["draftTop1Acceptance"], 1 / 3)
        self.assertAlmostEqual(report["draftTopKSurvival"], 2 / 3)

    def test_position_command_preserves_full_history(self):
        self.assertEqual(position_command(["b2e2", "h9g7"]),
                         "position startpos moves b2e2 h9g7")


if __name__ == "__main__":
    unittest.main()
