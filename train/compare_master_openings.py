"""Compare move models on master-book positions withheld from opening training."""

import argparse
import json
import random

import torch

from choice_model import (collect_predictions, load_model, load_rows,
                          load_teacher_sources, policy_metrics, sha256_file,
                          split_rows)


def position_key(row):
    return " ".join(row["fen"].split()[:2])


def soft_cross_entropy(predictions, rows, temperature):
    total = 0.0
    for (logits, _), row in zip(predictions, rows):
        log_probabilities = torch.log_softmax(logits / temperature, dim=0)
        total -= sum(weight * float(log_probabilities[row["legal"].index(move)])
                     for move, weight in row["policy"].items())
    return total / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--opening-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    training, validation, calibration, test = split_rows(load_teacher_sources(args.teacher), args.seed)
    teacher_keys = {position_key(row) for row in training + validation + calibration + test}
    available = [row for row in load_rows(args.book) if row.get("source") == "opening-book"
                 and row.get("policy") and position_key(row) not in teacher_keys]
    random.Random(args.seed).shuffle(available)
    withheld = available[args.opening_samples:]
    excluded = {position_key(row) for filename in args.exclude for row in load_rows(filename)}
    withheld = [row for row in withheld if position_key(row) not in excluded]
    if not withheld:
        raise ValueError("no independent master-opening positions remain")

    device = torch.device(args.device)
    results = []
    for filename in args.model:
        model, temperature = load_model(filename, device)
        predictions = collect_predictions(model, withheld, device, args.batch)
        results.append({"model": filename, "sha256": sha256_file(filename),
                        "temperature": temperature,
                        "hardBest": policy_metrics(predictions, temperature),
                        "softPolicyCrossEntropy": soft_cross_entropy(predictions, withheld, temperature)})
    report = {"purpose": "Master opening policy holdout; same positions for every model",
              "teacher": [{"file": name, "sha256": sha256_file(name)} for name in args.teacher],
              "book": {"file": args.book, "sha256": sha256_file(args.book)},
              "excluded": [{"file": name, "sha256": sha256_file(name)} for name in args.exclude],
              "seed": args.seed, "openingTrainingSamples": args.opening_samples,
              "availableBookPositions": len(available), "holdoutPositions": len(withheld),
              "models": results,
              "limitations": "Book frequencies reflect recorded master choices, not move quality or match strength."}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"holdoutPositions": len(withheld),
                      "models": [{"model": item["model"], "top1": item["hardBest"]["top1"],
                                  "softPolicyCrossEntropy": item["softPolicyCrossEntropy"]}
                                 for item in results]}), flush=True)


if __name__ == "__main__":
    main()
