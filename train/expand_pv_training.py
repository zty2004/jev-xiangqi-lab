"""Expand Pikafish principal variations into weighted continuation examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.speculative_engine import PikafishProcess

PV_SOURCE = "pikafish-pv-continuation"


def principal_variation(row: dict) -> list[str]:
    candidates = row.get("candidates", [])
    selected = next((item for item in candidates if item.get("move") == row.get("best") and item.get("pv")), None)
    selected = selected or next((item for item in candidates if item.get("rank") == 1 and item.get("pv")), None)
    return list(selected["pv"]) if selected else []


def continuation_rows(row: dict, helper, max_plies: int, weight: float) -> list[dict]:
    pv = principal_variation(row)
    if len(pv) < 2:
        return []
    derived = []
    for index in range(1, min(len(pv), max_plies + 1)):
        prefix, best = pv[:index], pv[index]
        helper.set_position(f"position fen {row['fen']} moves {' '.join(prefix)}")
        fen = helper.fen()
        legal = helper.legal_moves()
        if best not in legal:
            raise ValueError(f"PV move {best} is illegal after prefix {' '.join(prefix)}")
        derived.append({
            "kind": "position", "game": row["game"], "ply": f"{row.get('ply', 0)}:pv:{index}",
            "fen": fen, "legal": legal, "best": best, "played": best,
            "candidates": [], "source": PV_SOURCE, "trainingWeight": weight,
            "pvRootSourceIndex": row.get("sourceIndex"), "pvPrefix": prefix,
        })
    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pikafish", required=True)
    parser.add_argument("--max-pv-plies", type=int, default=4)
    parser.add_argument("--continuation-weight", type=float, default=0.15)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if (args.max_pv_plies < 1 or not 0 < args.continuation_weight <= 1 or args.shard_count < 1 or
            not 0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid PV expansion configuration")
    with open(args.input, encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    roots = [row for row in source if row.get("kind") == "position"]
    roots = [row for index, row in enumerate(roots) if index % args.shard_count == args.shard_index]
    metadata = {
        "kind": "meta", "schema": "jev-pv-expanded-training-v1", "source": str(Path(args.input)),
        "roots": len(roots), "maxPvPlies": args.max_pv_plies,
        "continuationWeight": args.continuation_weight, "continuationSource": PV_SOURCE,
        "shardIndex": args.shard_index, "shardCount": args.shard_count,
    }
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output exists: {output}")
    helper = PikafishProcess(args.pikafish, threads=1, hash_mb=16)
    continuations = 0
    try:
        with output.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for count, row in enumerate(roots, 1):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                extra = continuation_rows(row, helper, args.max_pv_plies, args.continuation_weight)
                for item in extra:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                continuations += len(extra)
                if count % 500 == 0 or count == len(roots):
                    print(f"roots={count}/{len(roots)} continuations={continuations}", flush=True)
    finally:
        helper.close()


if __name__ == "__main__":
    main()
