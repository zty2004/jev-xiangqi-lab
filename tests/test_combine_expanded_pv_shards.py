import json
import tempfile
import unittest
from pathlib import Path

from train.combine_expanded_pv_shards import combine
from train.expand_pv_training import PV_SOURCE


class CombineExpandedPvShardsTests(unittest.TestCase):
    def test_combines_roots_and_attached_continuations(self):
        common = {"kind": "meta", "schema": "jev-pv-expanded-training-v1", "source": "root",
                  "maxPvPlies": 4, "continuationWeight": 0.15,
                  "continuationSource": PV_SOURCE, "shardCount": 2}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for shard in (0, 1):
                root = {"kind": "position", "game": shard, "sourceIndex": shard, "ply": 1}
                pv = {"kind": "position", "game": shard, "source": PV_SOURCE,
                      "pvRootSourceIndex": shard, "ply": "1:pv:1"}
                path = Path(directory) / f"{shard}.jsonl"
                path.write_text("\n".join(json.dumps(item) for item in
                                           ({**common, "shardIndex": shard}, root, pv)) + "\n")
                paths.append(str(path))
            metadata, rows = combine(paths[::-1])
        self.assertEqual((metadata["roots"], metadata["continuations"]), (2, 2))
        self.assertEqual([row["game"] for row in rows], [0, 0, 1, 1])

    def test_rejects_detached_continuation(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            meta = {"kind": "meta", "schema": "jev-pv-expanded-training-v1", "source": "root",
                    "maxPvPlies": 4, "continuationWeight": 0.15,
                    "continuationSource": PV_SOURCE, "shardCount": 1, "shardIndex": 0}
            root = {"kind": "position", "game": 0, "sourceIndex": 0}
            pv = {"kind": "position", "game": 0, "source": PV_SOURCE, "pvRootSourceIndex": 7}
            handle.write("\n".join(json.dumps(item) for item in (meta, root, pv)) + "\n")
            handle.flush()
            with self.assertRaisesRegex(ValueError, "detached"):
                combine([handle.name])


if __name__ == "__main__":
    unittest.main()
