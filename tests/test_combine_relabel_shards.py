import json
import tempfile
import unittest
from pathlib import Path

from train.combine_relabel_shards import combine


class CombineRelabelShardsTests(unittest.TestCase):
    def test_combines_complete_shards_in_source_order(self):
        common = {"kind": "meta", "schema": "jev-pikafish-pv-relabel-v1",
                  "sourceSha256": "source", "teacherBinarySha256": "teacher", "nodes": 10,
                  "multipv": 2, "hashMb": 16, "shardCount": 2, "sourcePositions": 2}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for shard, source_index in ((0, 0), (1, 1)):
                meta = {**common, "shardIndex": shard, "shardPositions": 1}
                row = {"kind": "position", "sourceIndex": source_index, "best": "a0a1",
                       "legal": ["a0a1"], "candidates": [
                           {"rank": 1, "move": "a0a1", "pv": ["a0a1"],
                            "score": 0, "scoreType": "cp"}]}
                path = Path(directory) / f"{shard}.jsonl"
                path.write_text(json.dumps(meta) + "\n" + json.dumps(row) + "\n")
                paths.append(str(path))
            metadata, positions = combine(paths[::-1])
        self.assertEqual(metadata["positions"], 2)
        self.assertEqual([row["sourceIndex"] for row in positions], [0, 1])

    def test_rejects_incomplete_shard_set(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "meta", "schema": "jev-pikafish-pv-relabel-v1",
                                     "shardIndex": 0, "shardCount": 2}) + "\n")
            handle.flush()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                combine([handle.name])


if __name__ == "__main__":
    unittest.main()
