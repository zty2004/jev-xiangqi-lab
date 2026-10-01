"""Compare direct Jev choices with all-node PUCT on held-out teacher positions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

from jev_engine.policy_search import PuctSearch, TorchChoiceEvaluator
from jev_engine.xiangqi import legal_moves, move_name, parse_fen
from train.choice_model import (game_phase, load_teacher_sources, sha256_file,
                                split_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-depth", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--phase", choices=("opening", "middlegame", "endgame"),
                        default="middlegame")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--sample-seed", type=int, default=20261004)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("--positions must be positive")

    _, _, _, test = split_rows(load_teacher_sources(args.data), args.seed)
    rows = [row for row in test if game_phase(row["fen"]) == args.phase]
    random.Random(args.sample_seed).shuffle(rows)
    rows = rows[:args.positions]
    if not rows:
        raise ValueError("no held-out positions match the requested phase")
    evaluator = TorchChoiceEvaluator(args.model, args.device)
    search = PuctSearch(evaluator, args.c_puct, args.batch_size, args.maximum_depth)
    direct_hits = search_hits = direct_top8 = search_top8 = corrections = spoils = 0
    elapsed, depths, model_evaluations = [], [], []
    for index, row in enumerate(rows, 1):
        position = parse_fen(row["fen"])
        moves = legal_moves(position)
        names = [move_name(move) for move in moves]
        if set(names) != set(row["legal"]):
            missing = sorted(set(row["legal"]) - set(names))
            extra = sorted(set(names) - set(row["legal"]))
            raise ValueError(f"Python rules disagree with teacher row: missing={missing} extra={extra}")
        priors, _ = evaluator.evaluate_batch([position], [moves])[0]
        direct = sorted(zip(names, priors), key=lambda item: item[1], reverse=True)
        result = search.run(position, args.simulations)
        direct_hit = direct[0][0] == row["best"]
        search_hit = result.move == row["best"]
        direct_hits += direct_hit
        search_hits += search_hit
        direct_top8 += row["best"] in {move for move, _ in direct[:8]}
        search_top8 += row["best"] in {item[0] for item in result.policy[:8]}
        corrections += search_hit and not direct_hit
        spoils += direct_hit and not search_hit
        elapsed.append(result.elapsed_ms)
        depths.append(result.maximum_depth)
        model_evaluations.append(result.model_evaluations)
        if index % 10 == 0 or index == len(rows):
            print(f"positions={index}/{len(rows)} direct={direct_hits/index:.3f} "
                  f"search={search_hits/index:.3f}", file=sys.stderr, flush=True)

    count = len(rows)
    print(json.dumps({
        "schema": "jev-all-node-policy-search-benchmark-v1",
        "model": args.model,
        "modelSha256": sha256_file(args.model),
        "data": [{"path": filename, "sha256": sha256_file(filename)}
                 for filename in args.data],
        "seed": args.seed,
        "sampleSeed": args.sample_seed,
        "phase": args.phase,
        "positions": count,
        "simulations": args.simulations,
        "batchSize": args.batch_size,
        "maximumDepthLimit": args.maximum_depth,
        "cPuct": args.c_puct,
        "directTop1": direct_hits / count,
        "searchTop1": search_hits / count,
        "top1Gain": (search_hits - direct_hits) / count,
        "directTop8": direct_top8 / count,
        "searchTop8": search_top8 / count,
        "top8Gain": (search_top8 - direct_top8) / count,
        "corrections": corrections,
        "spoils": spoils,
        "meanElapsedMs": statistics.fmean(elapsed),
        "medianElapsedMs": statistics.median(elapsed),
        "meanMaximumDepth": statistics.fmean(depths),
        "maximumReachedDepth": max(depths),
        "meanModelEvaluations": statistics.fmean(model_evaluations),
    }, indent=2))


if __name__ == "__main__":
    main()
