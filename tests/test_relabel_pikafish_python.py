import unittest

from train.relabel_pikafish_python import parse_pikafish_info, select_multipv


class RelabelPikafishPythonTests(unittest.TestCase):
    def test_parses_complete_pv(self):
        line = "info depth 12 seldepth 27 multipv 2 score cp -31 nodes 250001 pv h9g7 b0c2"
        self.assertEqual(parse_pikafish_info(line), {
            "depth": 12, "selectiveDepth": 27, "nodes": 250001, "rank": 2,
            "move": "h9g7", "pv": ["h9g7", "b0c2"], "score": -31, "scoreType": "cp"})

    def test_selects_depth_with_most_unique_candidates_then_deepest(self):
        lines = [
            "info depth 12 seldepth 20 multipv 1 score cp 30 nodes 90 pv b2e2 h9g7",
            "info depth 12 seldepth 21 multipv 2 score cp 10 nodes 90 pv h2e2 h9g7",
            "info depth 13 seldepth 24 multipv 1 score cp 35 nodes 100 pv b2e2 h9g7",
        ]
        result = select_multipv(lines, "b2e2")
        self.assertEqual(result["depth"], 12)
        self.assertEqual([item["move"] for item in result["candidates"]], ["b2e2", "h2e2"])
        self.assertEqual(result["candidates"][0]["pv"], ["b2e2", "h9g7"])


if __name__ == "__main__":
    unittest.main()
