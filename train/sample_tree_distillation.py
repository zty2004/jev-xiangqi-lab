"""Sample internal search-tree positions from Pikafish MultiPV lines.

The existing PV expansion data teaches the policy which move followed the
principal variation, but deliberately carries no value target.  This sampler
also walks the non-best MultiPV branches and emits their internal positions so
they can be re-analysed by Pikafish.  The resulting labels supervise both the
policy and value heads away from the root and away from the best line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from jev_engine.xiangqi import legal_moves, move_name, parse_fen, play_move, position_key, to_fen
from train.choice_model import game_phase


TREE_SEED_SOURCE = "pikafish-tree-seed"


def sha256_file(filename: str) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_positions(filename: str, phase: str | None) -> list[dict]:
    with open(filename, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    positions = [row for row in rows if row.get("kind") == "position" and
                 any(item.get("pv") for item in row.get("candidates", []))]
    if phase:
        positions = [row for row in positions if game_phase(row["fen"]) == phase]
    if not positions:
        raise ValueError("source contains no eligible teacher positions")
    return positions


def ordered_candidates(row: dict, maximum: int) -> list[dict]:
    candidates = [item for item in row.get("candidates", [])
                  if item.get("pv") and item.get("pv", [None])[0] == item.get("move")]
    candidates.sort(key=lambda item: (item.get("rank", 10**9), item.get("move", "")))
    return candidates[:maximum]


def sample_tree_rows(source: list[dict], roots: int, candidates: int, plies: int,
                     seed: int, parent_source_index: int = 0) -> list[dict]:
    if roots < 1 or candidates < 1 or plies < 1:
        raise ValueError("tree sampling limits must be positive")
    selected = list(enumerate(source))
    random.Random(seed).shuffle(selected)
    selected = selected[:roots]
    seen: set[str] = set()
    result: list[dict] = []
    for source_index, row in selected:
        for candidate in ordered_candidates(row, candidates):
            position = parse_fen(row["fen"])
            prefix: list[str] = []
            for ply, notation in enumerate(candidate["pv"][:plies], 1):
                position = play_move(position, notation)
                prefix.append(notation)
                key = position_key(position)
                if key in seen:
                    continue
                seen.add(key)
                moves = [move_name(move) for move in legal_moves(position)]
                if not moves:
                    break
                result.append({
                    "kind": "position",
                    "game": row.get("game", f"source-{source_index}"),
                    "ply": f"{row.get('ply', 0)}:tree:{candidate.get('rank', 0)}:{ply}",
                    "fen": to_fen(position),
                    "legal": moves,
                    # A legal placeholder is required by generic JSONL readers;
                    # relabel_pikafish_python replaces it before training.
                    "best": moves[0],
                    "played": moves[0],
                    "candidates": [],
                    "source": TREE_SEED_SOURCE,
                    # Keep the derived line in the same train/held-out split
                    # as its root when this file is combined after the parent
                    # teacher source.
                    "splitGroup": f"{parent_source_index}:{row.get('game', f'source-{source_index}')}",
                    "treeRootSourceIndex": row.get("sourceIndex", source_index),
                    "treeCandidateRank": candidate.get("rank"),
                    "treeCandidateMove": candidate.get("move"),
                    "treePrefix": list(prefix),
                })
    if not result:
        raise ValueError("teacher candidates contain no usable internal positions")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roots", type=int, default=2000)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--plies", type=int, default=4)
    parser.add_argument("--phase", choices=("opening", "middlegame", "endgame"))
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--parent-source-index", type=int, default=0,
                        help="index of the root teacher in the eventual combined --data list")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output exists: {output}")
    source = load_positions(args.input, args.phase)
    if args.parent_source_index < 0:
        raise ValueError("parent source index must be nonnegative")
    rows = sample_tree_rows(source, args.roots, args.candidates, args.plies, args.seed,
                            args.parent_source_index)
    metadata = {
        "kind": "meta",
        "schema": "jev-pikafish-tree-seed-v2",
        "source": str(Path(args.input)),
        "sourceSha256": sha256_file(args.input),
        "sourcePositions": len(source),
        "sampledRoots": min(args.roots, len(source)),
        "maximumCandidatesPerRoot": args.candidates,
        "maximumPliesPerCandidate": args.plies,
        "phase": args.phase,
        "seed": args.seed,
        "parentSourceIndex": args.parent_source_index,
        "derivedPositions": len(rows),
        "requiresPikafishRelabel": True,
    }
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
