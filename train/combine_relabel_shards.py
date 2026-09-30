"""Validate and combine disjoint Pikafish PV relabel shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def combine(filenames: list[str]) -> tuple[dict, list[dict]]:
    metadata, positions = [], []
    for filename in filenames:
        with open(filename, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or rows[0].get("schema") != "jev-pikafish-pv-relabel-v1":
            raise ValueError(f"invalid relabel shard: {filename}")
        metadata.append(rows[0])
        positions.extend(rows[1:])
    stable_keys = ("schema", "sourceSha256", "teacherBinarySha256", "nodes", "multipv",
                   "hashMb", "shardCount", "sourcePositions")
    expected = {key: metadata[0].get(key) for key in stable_keys}
    if any({key: item.get(key) for key in stable_keys} != expected for item in metadata[1:]):
        raise ValueError("relabel shard metadata mismatch")
    shard_count = expected["shardCount"]
    shard_indices = {item.get("shardIndex") for item in metadata}
    if shard_indices != set(range(shard_count)):
        raise ValueError("relabel shards are incomplete")
    indices = [row.get("sourceIndex") for row in positions]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate relabel source index")
    if set(indices) != set(range(expected["sourcePositions"])):
        raise ValueError("relabel positions are incomplete")
    for row in positions:
        if row.get("best") not in row.get("legal", []) or not row.get("candidates"):
            raise ValueError(f"invalid relabel row {row.get('sourceIndex')}")
        if any(not item.get("pv") or item.get("move") != item["pv"][0]
               for item in row["candidates"]):
            raise ValueError(f"invalid candidate PV at row {row.get('sourceIndex')}")
    positions.sort(key=lambda row: row["sourceIndex"])
    combined = {"kind": "meta", **expected, "positions": len(positions),
                "source": metadata[0].get("source"), "sourceMetadata": metadata[0].get("sourceMetadata")}
    return combined, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output exists: {output}")
    metadata, positions = combine(args.input)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in positions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "positions": len(positions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
