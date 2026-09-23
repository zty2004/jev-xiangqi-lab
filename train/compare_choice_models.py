"""Paired policy comparison on the same held-out games and positions."""

import argparse
import json
import math
import random
from collections import defaultdict

import torch

import choice_model as choice


def interval(values):
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="existing model")
    parser.add_argument("--model-b", required=True, help="candidate model")
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--history-data")
    parser.add_argument("--exclude-data", action="append", default=[])
    parser.add_argument("--test-source-index", type=int, help="limit the held-out comparison to one --data source (zero based)")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.test_source_index is not None and not (0 <= args.test_source_index < len(args.data)):
        parser.error("--test-source-index is outside the --data sources")
    _, _, _, test = choice.split_rows(choice.load_teacher_sources(args.data, args.history_data), args.seed)
    if args.test_source_index is not None:
        test = [row for row in test if (str(row["game"]).split(":", 1)[0] if len(args.data) > 1 else "0") == str(args.test_source_index)]
    excluded = {" ".join(row["fen"].split()[:2])
                for filename in args.exclude_data for row in choice.load_rows(filename)}
    test = [row for row in test if " ".join(row["fen"].split()[:2]) not in excluded]
    if not test:
        raise ValueError("no test positions remain")
    device = choice.select_device(args.device)
    predictions = {}
    metrics = {}
    temperatures = {}
    for name, filename in (("a", args.model_a), ("b", args.model_b)):
        model, temperature = choice.load_model(filename, device)
        predictions[name] = choice.collect_predictions(model, test, device, args.batch)
        temperatures[name] = temperature
        metrics[name] = choice.policy_metrics(predictions[name], temperature)
    by_game = defaultdict(list)
    for row, (old_logits, best), (new_logits, new_best) in zip(test, predictions["a"], predictions["b"]):
        if best != new_best:
            raise ValueError("models were evaluated on different target moves")
        old_probability = torch.softmax(old_logits / temperatures["a"], dim=0)
        new_probability = torch.softmax(new_logits / temperatures["b"], dim=0)
        top1_gain = int(new_probability.argmax() == best) - int(old_probability.argmax() == best)
        nll_reduction = math.log(max(float(new_probability[best]), 1e-30)) - math.log(max(float(old_probability[best]), 1e-30))
        by_game[row["game"]].append((top1_gain, nll_reduction))
    games = list(by_game)
    rng = random.Random(args.seed)
    samples = []
    for _ in range(args.bootstrap):
        selected = [item for game in (rng.choice(games) for _ in games) for item in by_game[game]]
        samples.append((sum(item[0] for item in selected) / len(selected),
                        sum(item[1] for item in selected) / len(selected)))
    print(json.dumps({
        "positions": len(test), "games": len(games), "bootstrapSamples": args.bootstrap,
        "testSourceIndex": args.test_source_index,
        "modelA": {"path": args.model_a, "sha256": choice.sha256_file(args.model_a), "metrics": metrics["a"]},
        "modelB": {"path": args.model_b, "sha256": choice.sha256_file(args.model_b), "metrics": metrics["b"]},
        "top1GainBminusA": metrics["b"]["top1"] - metrics["a"]["top1"],
        "top1Gain95GameBootstrap": interval([sample[0] for sample in samples]),
        "nllReductionAminusB": metrics["a"]["nll"] - metrics["b"]["nll"],
        "nllReduction95GameBootstrap": interval([sample[1] for sample in samples]),
    }, indent=2))


if __name__ == "__main__":
    main()
