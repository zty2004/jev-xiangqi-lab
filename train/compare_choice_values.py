"""Paired comparison of two Jev scalar value heads on held-out games."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

try:
    from train import choice_model as choice
except ModuleNotFoundError:  # Direct execution from the train directory.
    import choice_model as choice


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]


def predict(model, rows: list[dict], device, batch: int) -> tuple[list[float], list[float], list[float]]:
    dataset = choice.TeacherDataset(rows, model.input_channels, "scalar")
    loader = DataLoader(dataset, batch_size=batch, collate_fn=choice.collate)
    predictions: list[float] = []
    targets: list[float] = []
    weights: list[float] = []
    model.eval()
    with torch.no_grad():
        for boards, moves, _, mask, values, _, value_weights in loader:
            _, output = model(boards.to(device), moves.to(device), mask.to(device))
            predictions.extend(output.cpu().tolist())
            targets.extend(values.tolist())
            weights.extend(value_weights.tolist())
    return predictions, targets, weights


def weighted_metrics(errors: list[tuple[float, float]]) -> dict:
    total = sum(weight for _, weight in errors)
    if total <= 0:
        raise ValueError("comparison contains no weighted value targets")
    return {
        "positions": sum(weight > 0 for _, weight in errors),
        "effectiveWeight": total,
        "mae": sum(abs(error) * weight for error, weight in errors) / total,
        "mse": sum(error * error * weight for error, weight in errors) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--test-source-index", type=int)
    parser.add_argument("--phase", choices=("opening", "middlegame", "endgame"))
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.test_source_index is not None and not 0 <= args.test_source_index < len(args.data):
        parser.error("--test-source-index is outside the teacher sources")

    _, _, _, rows = choice.split_rows(choice.load_teacher_sources(args.data), args.seed)
    if args.phase:
        rows = [row for row in rows if choice.game_phase(row["fen"]) == args.phase]
    if args.test_source_index is not None:
        rows = [row for row in rows if choice.source_index(row) == args.test_source_index]
    if not rows:
        raise ValueError("no held-out positions remain")

    device = choice.select_device(args.device)
    old, _ = choice.load_model(args.model_a, device)
    new, _ = choice.load_model(args.model_b, device)
    if old.value_head != "scalar" or new.value_head != "scalar":
        raise ValueError("paired value comparison requires scalar value heads")
    old_predictions, targets, weights = predict(old, rows, device, args.batch)
    new_predictions, new_targets, new_weights = predict(new, rows, device, args.batch)
    if targets != new_targets or weights != new_weights:
        raise ValueError("models were evaluated against different value targets")

    old_errors = [(prediction - target, weight)
                  for prediction, target, weight in zip(old_predictions, targets, weights)]
    new_errors = [(prediction - target, weight)
                  for prediction, target, weight in zip(new_predictions, targets, weights)]
    old_metrics, new_metrics = weighted_metrics(old_errors), weighted_metrics(new_errors)
    by_game: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row, old_error, new_error in zip(rows, old_errors, new_errors):
        weight = old_error[1]
        if weight > 0:
            by_game[str(row["game"])].append(
                ((abs(old_error[0]) - abs(new_error[0])) * weight,
                 (old_error[0] ** 2 - new_error[0] ** 2) * weight, weight))
    games = list(by_game)
    if not games:
        raise ValueError("comparison contains no held-out games with value targets")
    rng = random.Random(args.seed)
    samples = []
    for _ in range(args.bootstrap):
        selected = [item for game in (rng.choice(games) for _ in games) for item in by_game[game]]
        total = sum(item[2] for item in selected)
        samples.append((sum(item[0] for item in selected) / total,
                        sum(item[1] for item in selected) / total))
    print(json.dumps({
        "schema": "jev-choice-value-comparison-v1",
        "positions": old_metrics["positions"],
        "games": len(games),
        "phase": args.phase,
        "testSourceIndex": args.test_source_index,
        "bootstrapSamples": args.bootstrap,
        "modelA": {"path": args.model_a, "sha256": choice.sha256_file(args.model_a),
                   "metrics": old_metrics},
        "modelB": {"path": args.model_b, "sha256": choice.sha256_file(args.model_b),
                   "metrics": new_metrics},
        "maeReductionAminusB": old_metrics["mae"] - new_metrics["mae"],
        "maeReduction95GameBootstrap": interval([sample[0] for sample in samples]),
        "mseReductionAminusB": old_metrics["mse"] - new_metrics["mse"],
        "mseReduction95GameBootstrap": interval([sample[1] for sample in samples]),
    }, indent=2))


if __name__ == "__main__":
    main()
