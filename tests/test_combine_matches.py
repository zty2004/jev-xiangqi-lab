import json
import tempfile
import unittest
from pathlib import Path

from hybrid.combine_matches import combine


class CombineMatchesTests(unittest.TestCase):
    def test_combines_disjoint_complete_pairs(self):
        common = {"kind": "jev-pikafish-fixed-time-match", "pikafishSha256": "p",
                  "modelSha256": "m", "movetimeMs": 5000, "threads": 1, "hashMb": 256,
                  "topK": 8, "verificationFraction": 0.15, "openingPlies": 8, "seed": 7}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for pair in (0, 1):
                games = [{"pair": pair, "hybridColor": color, "hybridResult": "draw",
                          "speculation": [], "hybridMoveTimesMs": [], "nativeMoveTimesMs": []}
                         for color in ("red", "black")]
                path = Path(directory) / f"{pair}.json"
                path.write_text(json.dumps({**common, "games": games}), encoding="utf-8")
                paths.append(str(path))
            result = combine(paths)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["summary"]["games"], 4)

    def test_rejects_duplicate_games(self):
        common = {"kind": "x", "pikafishSha256": "p", "modelSha256": "m", "movetimeMs": 1,
                  "threads": 1, "hashMb": 1, "topK": 1, "verificationFraction": 0.1,
                  "openingPlies": 1, "seed": 1}
        game = {"pair": 0, "hybridColor": "red"}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(2):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps({**common, "games": [game]}), encoding="utf-8")
                paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                combine(paths)


if __name__ == "__main__":
    unittest.main()
