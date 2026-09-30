"""Validate and combine PV-expanded training shards without changing game IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.expand_pv_training import PV_SOURCE


def combine(filenames: list[str]) -> tuple[dict, list[dict]]:
    metadata, rows = [], []
    for filename in filenames:
        with open(filename, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if not items or items[0].get("schema") != "jev-pv-expanded-training-v1":
            raise ValueError(f"invalid expanded PV shard: {filename}")
        metadata.append(items[0])
        rows.extend(items[1:])
    stable_keys = ("schema", "source", "maxPvPlies", "continuationWeight",
                   "continuationSource", "shardCount")
    expected = {key: metadata[0].get(key) for key in stable_keys}
    if any({key: item.get(key) for key in stable_keys} != expected for item in metadata[1:]):
        raise ValueError("expanded PV shard metadata mismatch")
    if {item.get("shardIndex") for item in metadata} != set(range(expected["shardCount"])):
        raise ValueError("expanded PV shards are incomplete")
    roots = [row for row in rows if row.get("source") != PV_SOURCE]
    continuations = [row for row in rows if row.get("source") == PV_SOURCE]
    root_indices = [row.get("sourceIndex") for row in roots]
    if len(root_indices) != len(set(root_indices)) or set(root_indices) != set(range(len(roots))):
        raise ValueError("expanded PV roots are duplicated or incomplete")
    root_by_game = {str(row["game"]): row["sourceIndex"] for row in roots}
    if len(root_by_game) != len(roots):
        raise ValueError("expanded PV roots do not have unique games")
    for row in continuations:
        if row.get("pvRootSourceIndex") != root_by_game.get(str(row.get("game"))):
            raise ValueError("continuation is detached from its root game")
    rows.sort(key=lambda row: (root_by_game[str(row["game"])], row.get("source") == PV_SOURCE,
                               str(row.get("ply"))))
    combined = {"kind": "meta", **expected, "roots": len(roots),
                "continuations": len(continuations), "positions": len(rows)}
    return combined, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output exists: {output}")
    metadata, rows = combine(args.input)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
