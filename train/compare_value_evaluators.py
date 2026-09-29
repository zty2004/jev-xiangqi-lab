"""Compare the trained NNUE value model with the classical evaluator."""

import argparse
import hashlib
import json
import math

import torch
from torch.utils.data import DataLoader

import nnue_value

PHASE_UNITS = {"r": 4, "c": 2, "n": 2, "b": 1, "a": 1, "p": 0, "k": 0}


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase(fen):
    board = nnue_value.parse_board(fen)
    units = sum(PHASE_UNITS[piece.lower()] for piece in board if piece != ".")
    non_kings = sum(piece.lower() != "k" for piece in board if piece != ".")
    fullmove = int(fen.split()[5])
    if units <= 12 or non_kings <= 8:
        return "endgame"
    if fullmove <= 12 and units >= 32:
        return "opening"
    return "middlegame"


def metrics(predictions, targets):
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return {"positions": len(errors), "maeCp": sum(map(abs, errors)) / len(errors),
            "rmseCp": math.sqrt(sum(error * error for error in errors) / len(errors))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output-scale", type=int, default=2000)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    rows = nnue_value.load_rows(args.data)
    _, validation = nnue_value.split_rows(rows, args.seed)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    hidden, head = state["embedding.weight"].shape[1], state["fc1.weight"].shape[0]
    device = nnue_value.select_device(args.device)
    model = nnue_value.NnueValue(hidden, head, args.output_scale).to(device)
    model.load_state_dict(state)
    loader = DataLoader(nnue_value.ValueDataset(validation), batch_size=args.batch, collate_fn=nnue_value.collate)
    neural = []
    model.eval()
    with torch.no_grad():
        for indices, mask, side, _ in loader:
            neural.extend(model(indices.to(device), mask.to(device), side.to(device)).cpu().tolist())
    targets = [row["score"] for row in validation]
    classical = [nnue_value.classical_score(row["fen"]) for row in validation]
    if args.residual:
        neural = [base + correction for base, correction in zip(classical, neural)]
    phases = [phase(row["fen"]) for row in validation]
    report = {"schema": "value-evaluator-comparison-v1", "seed": args.seed,
              "model": {"file": args.model, "sha256": sha256_file(args.model), "hidden": hidden, "head": head,
                        "residual": args.residual},
              "data": [{"file": filename, "sha256": sha256_file(filename)} for filename in args.data],
              "overall": {"classical": metrics(classical, targets), "nnue": metrics(neural, targets)}, "phases": {}}
    for name in ("opening", "middlegame", "endgame"):
        selected = [index for index, value in enumerate(phases) if value == name]
        if selected:
            report["phases"][name] = {"classical": metrics([classical[i] for i in selected], [targets[i] for i in selected]),
                                       "nnue": metrics([neural[i] for i in selected], [targets[i] for i in selected])}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    main()
