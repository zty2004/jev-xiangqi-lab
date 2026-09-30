"""Run reproducible paired matches between native Pikafish and Jev+Pikafish."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from hybrid.speculative_engine import JevDraftModel, PikafishProcess, SpeculativeEngine


def sha256_file(filename: str) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_openings(filename: str, count: int, plies: int, seed: int) -> list[list[str]]:
    rows = []
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("kind") == "game" and len(item.get("opening", [])) >= plies:
                rows.append(item["opening"][:plies])
    unique = list(dict.fromkeys(tuple(row) for row in rows))
    random.Random(seed).shuffle(unique)
    if len(unique) < count:
        raise ValueError(f"only {len(unique)} distinct openings, need {count}")
    return [list(row) for row in unique[:count]]


def position_command(moves: list[str]) -> str:
    return "position startpos" + (" moves " + " ".join(moves) if moves else "")


def summarize(games: list[dict]) -> dict:
    outcomes = Counter(game["hybridResult"] for game in games)
    dispositions = Counter(item for game in games for item in game["speculation"])
    hybrid_times = [item for game in games for item in game["hybridMoveTimesMs"]]
    native_times = [item for game in games for item in game["nativeMoveTimesMs"]]
    decisive = outcomes["win"] + outcomes["loss"]
    return {
        "games": len(games),
        "hybridWins": outcomes["win"],
        "draws": outcomes["draw"],
        "hybridLosses": outcomes["loss"],
        "hybridScore": ((outcomes["win"] + outcomes["draw"] / 2) / len(games)) if games else 0,
        "decisiveScore": (outcomes["win"] / decisive) if decisive else None,
        "speculation": dict(dispositions),
        "draftTop1Acceptance": (dispositions["accepted-top1"] / sum(dispositions.values())) if dispositions else 0,
        "draftTopKSurvival": ((dispositions["accepted-top1"] + dispositions["corrected-within-draft"]) /
                              sum(dispositions.values())) if dispositions else 0,
        "meanHybridMoveMs": statistics.fmean(hybrid_times) if hybrid_times else 0,
        "meanNativeMoveMs": statistics.fmean(native_times) if native_times else 0,
    }


def play_game(native: PikafishProcess, hybrid: SpeculativeEngine, opening: list[str],
              hybrid_color: str, movetime_ms: int, max_plies: int) -> dict:
    moves = list(opening)
    fen_counts = Counter()
    speculation = []
    hybrid_times, native_times = [], []
    draft_ranks = []
    reason, winner = "max-plies", None

    native.clear_hash()
    hybrid.target.clear_hash()
    for _ in range(len(moves), max_plies):
        side = "red" if len(moves) % 2 == 0 else "black"
        command = position_command(moves)
        if side == hybrid_color:
            result = hybrid.choose(command, movetime_ms)
            move = result.move
            speculation.append(result.disposition)
            hybrid_times.append(result.elapsed_ms)
            draft_ranks.append(next((index + 1 for index, (candidate, _) in enumerate(result.draft)
                                     if candidate == move), None))
        else:
            native.set_position(command)
            result = native.search(movetime_ms)
            move = result.move
            native_times.append(result.elapsed_ms)
        if not move:
            winner = "black" if side == "red" else "red"
            reason = "no-legal-move"
            break
        moves.append(move)

        native.set_position(position_command(moves))
        fen = native.fen()
        key = " ".join(fen.split()[:2])
        fen_counts[key] += 1
        if fen_counts[key] >= 3:
            reason = "threefold-repetition"
            break
        if int(fen.split()[4]) >= 120:
            reason = "120-ply-rule"
            break
        if not native.legal_moves():
            winner = side
            reason = "no-legal-move"
            break

    hybrid_result = "draw" if winner is None else "win" if winner == hybrid_color else "loss"
    return {
        "hybridColor": hybrid_color,
        "hybridResult": hybrid_result,
        "winner": winner,
        "reason": reason,
        "opening": opening,
        "moves": moves,
        "plies": len(moves),
        "speculation": speculation,
        "hybridFinalDraftRanks": draft_ranks,
        "hybridMoveTimesMs": hybrid_times,
        "nativeMoveTimesMs": native_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pikafish", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--movetime-ms", type=int, default=5000)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--verification-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.pairs < 1 or args.movetime_ms < 100 or args.max_plies <= args.opening_plies:
        raise ValueError("invalid match configuration")

    openings = load_openings(args.openings, args.pairs, args.opening_plies, args.seed)
    draft = JevDraftModel(args.model, args.device)
    native = PikafishProcess(args.pikafish, args.threads, args.hash)
    target = PikafishProcess(args.pikafish, args.threads, args.hash)
    hybrid = SpeculativeEngine(target, draft, args.top_k, args.verification_fraction)
    games = []
    metadata = {
        "kind": "jev-pikafish-fixed-time-match",
        "createdUnix": int(time.time()),
        "pikafish": str(Path(args.pikafish).resolve()),
        "pikafishSha256": sha256_file(args.pikafish),
        "pikafishNnueSha256": (sha256_file(str(Path(args.pikafish).resolve().parent / "pikafish.nnue"))
                                if (Path(args.pikafish).resolve().parent / "pikafish.nnue").is_file() else None),
        "model": str(Path(args.model).resolve()),
        "modelSha256": sha256_file(args.model),
        "pairs": args.pairs,
        "movetimeMs": args.movetime_ms,
        "threads": args.threads,
        "hashMb": args.hash,
        "topK": args.top_k,
        "verificationFraction": args.verification_fraction,
        "openingPlies": args.opening_plies,
        "seed": args.seed,
    }
    try:
        for pair, opening in enumerate(openings):
            for hybrid_color in ("red", "black"):
                game = play_game(native, hybrid, opening, hybrid_color, args.movetime_ms, args.max_plies)
                game.update({"game": len(games), "pair": pair})
                games.append(game)
                print(json.dumps({"game": game["game"], "pair": pair, "hybridColor": hybrid_color,
                                  "result": game["hybridResult"], "reason": game["reason"],
                                  "plies": game["plies"]}), flush=True)
                Path(args.output).write_text(json.dumps({**metadata, "summary": summarize(games),
                                                        "games": games}, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    finally:
        native.close()
        target.close()
    print(json.dumps(summarize(games)), flush=True)


if __name__ == "__main__":
    main()
