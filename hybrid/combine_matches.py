"""Combine disjoint paired-match shards and recompute the aggregate summary."""

import argparse
import json
from pathlib import Path

from hybrid.match import summarize


def combine(filenames: list[str]) -> dict:
    reports = [json.loads(Path(filename).read_text(encoding="utf-8")) for filename in filenames]
    if not reports:
        raise ValueError("no match reports")
    keys = ("kind", "pikafishSha256", "modelSha256", "movetimeMs", "threads", "hashMb",
            "topK", "verificationFraction", "openingPlies", "seed")
    expected = {key: reports[0].get(key) for key in keys}
    if any({key: report.get(key) for key in keys} != expected for report in reports[1:]):
        raise ValueError("match shards have incompatible configurations")
    games = [game for report in reports for game in report["games"]]
    identities = [(game["pair"], game["hybridColor"]) for game in games]
    if len(identities) != len(set(identities)):
        raise ValueError("match shards contain duplicate pair/color games")
    pair_colors = {}
    for pair, color in identities:
        pair_colors.setdefault(pair, set()).add(color)
    if any(colors != {"red", "black"} for colors in pair_colors.values()):
        raise ValueError("every opening must contain a complete red/black pair")
    games.sort(key=lambda game: (game["pair"], 0 if game["hybridColor"] == "red" else 1))
    return {**expected, "pairs": len(pair_colors), "summary": summarize(games), "games": games,
            "sourceReports": [str(Path(filename)) for filename in filenames]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = combine(args.input)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
