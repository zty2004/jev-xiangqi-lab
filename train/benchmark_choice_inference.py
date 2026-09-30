"""Benchmark single-position Jev policy inference with realistic legal moves."""

import argparse
import json
import statistics
import time

import choice_model as choice


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.positions < 1 or args.warmup < 0:
        parser.error("--positions must be positive and --warmup nonnegative")

    rows = choice.load_rows(args.data)[:args.positions]
    device = choice.select_device(args.device)
    reports = []
    for filename in args.model:
        model, temperature = choice.load_model(filename, device)
        for row in rows[:min(args.warmup, len(rows))]:
            choice.rank(model, row["fen"], row["legal"], device, temperature,
                        row.get("history"))
        elapsed = []
        for row in rows:
            started = time.perf_counter()
            choice.rank(model, row["fen"], row["legal"], device, temperature,
                        row.get("history"))
            elapsed.append((time.perf_counter() - started) * 1000)
        reports.append({
            "model": filename,
            "sha256": choice.sha256_file(filename),
            "positions": len(rows),
            "device": str(device),
            "meanMs": statistics.fmean(elapsed),
            "medianMs": statistics.median(elapsed),
            "p95Ms": percentile(elapsed, 0.95),
            "maxMs": max(elapsed),
        })
    print(json.dumps({"kind": "jev-choice-inference-benchmark-v1", "models": reports},
                     indent=2))


if __name__ == "__main__":
    main()
