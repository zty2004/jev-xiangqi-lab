"""Train an original incrementally updatable Xiangqi value network."""

import argparse
import base64
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PIECES = "KABNRCPkabnrcp"
FEATURES = 9 * len(PIECES) * 90
MAX_PIECES = 32


def parse_board(fen):
    board = []
    for rank in fen.split()[0].split("/"):
        for token in rank:
            board.extend(["."] * int(token) if token.isdigit() else [{"H": "N", "h": "n", "E": "B", "e": "b"}.get(token, token)])
    if len(board) != 90:
        raise ValueError("invalid FEN")
    return board


def feature_indices(fen, perspective):
    board = parse_board(fen)
    king_piece = "K" if perspective == "red" else "k"
    king = board.index(king_piece)
    king = king if perspective == "red" else 89 - king
    mirror = king % 9 > 4
    king_x = 8 - king % 9 if mirror else king % 9
    bucket = (king // 9 - 7) * 3 + king_x - 3
    if not 0 <= bucket < 9:
        raise ValueError("king outside palace")
    result = []
    for square, piece in enumerate(board):
        if piece == ".":
            continue
        normalized_square = square if perspective == "red" else 89 - square
        if mirror:
            normalized_square = normalized_square // 9 * 9 + 8 - normalized_square % 9
        normalized_piece = piece if perspective == "red" else piece.swapcase()
        result.append((bucket * len(PIECES) + PIECES.index(normalized_piece)) * 90 + normalized_square)
    return result


def teacher_score(row):
    candidates = row.get("candidates", [])
    best = next((item for item in candidates if item.get("move") == row.get("best")), None)
    item = best or (candidates[0] if candidates else None)
    if not item:
        return None
    score = float(item["score"])
    if item.get("scoreType") == "mate":
        score = math.copysign(30000, score or 1)
    return max(-2000.0, min(2000.0, score))


def load_rows(filenames):
    rows = []
    for source, filename in enumerate(filenames):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                score = teacher_score(row) if row.get("kind") == "position" else None
                if score is not None:
                    rows.append({**row, "game": f"{source}:{row['game']}", "score": score})
    if not rows:
        raise ValueError("no scored teacher positions")
    return rows


def split_rows(rows, seed):
    games = sorted({row["game"] for row in rows})
    random.Random(seed).shuffle(games)
    validation_games = set(games[::10])
    validation = [row for row in rows if row["game"] in validation_games]
    training = [row for row in rows if row["game"] not in validation_games]
    validation_keys = {" ".join(row["fen"].split()[:2]) for row in validation}
    training = [row for row in training if " ".join(row["fen"].split()[:2]) not in validation_keys]
    if not training or not validation:
        raise ValueError("need multiple games for train/validation split")
    return training, validation


class ValueDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        red = feature_indices(row["fen"], "red")
        black = feature_indices(row["fen"], "black")
        side = 0 if row["fen"].split()[1] == "w" else 1
        return red, black, side, row["score"]


def collate(samples):
    indices = torch.zeros((len(samples), 2, MAX_PIECES), dtype=torch.long)
    mask = torch.zeros((len(samples), 2, MAX_PIECES), dtype=torch.float32)
    for batch, sample in enumerate(samples):
        for perspective, active in enumerate(sample[:2]):
            indices[batch, perspective, :len(active)] = torch.tensor(active)
            mask[batch, perspective, :len(active)] = 1
    return indices, mask, torch.tensor([sample[2] for sample in samples]), torch.tensor([sample[3] for sample in samples], dtype=torch.float32)


class NnueValue(nn.Module):
    def __init__(self, hidden=256, head=32, output_scale=1000):
        super().__init__()
        self.hidden, self.head, self.output_scale = hidden, head, output_scale
        self.embedding = nn.Embedding(FEATURES, hidden)
        self.feature_bias = nn.Parameter(torch.zeros(hidden))
        self.fc1 = nn.Linear(hidden * 4, head)
        self.output = nn.Linear(head * 2, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, indices, mask, side):
        accumulators = self.feature_bias + (self.embedding(indices) * mask.unsqueeze(-1)).sum(dim=2)
        batch = torch.arange(indices.shape[0], device=indices.device)
        own, other = accumulators[batch, side], accumulators[batch, 1 - side]
        own, other = own.clamp(0, 1), other.clamp(0, 1)
        inputs = torch.cat((own, own.square(), other, other.square()), dim=1)
        hidden = self.fc1(inputs).clamp(0, 1)
        raw = self.output(torch.cat((hidden, hidden.square()), dim=1)).squeeze(1)
        return torch.tanh(raw) * self.output_scale


def select_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def metrics(model, loader, device):
    count = absolute = squared = 0.0
    model.eval()
    with torch.no_grad():
        for indices, mask, side, target in loader:
            prediction = model(indices.to(device), mask.to(device), side.to(device)).cpu()
            error = prediction - target
            count += len(target)
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
    return {"positions": int(count), "maeCp": absolute / count, "rmseCp": math.sqrt(squared / count)}


def encode_tensor(tensor):
    values = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return base64.b64encode(values).decode("ascii")


def export_model(model, filename, metadata):
    artifact = {"format": "jev-xiangqi-nnue-v1", "features": FEATURES, "hidden": model.hidden,
                "head": model.head, "outputScale": model.output_scale, **metadata,
                "embedding": encode_tensor(model.embedding.weight), "featureBias": encode_tensor(model.feature_bias),
                "fc1Weight": encode_tensor(model.fc1.weight), "fc1Bias": encode_tensor(model.fc1.bias),
                "outputWeight": encode_tensor(model.output.weight.squeeze(0)),
                "outputBias": float(model.output.bias.detach().cpu())}
    Path(filename).write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")


def train(args):
    torch.manual_seed(args.seed)
    rows = load_rows(args.data)
    training, validation = split_rows(rows, args.seed)
    device = select_device(args.device)
    model = NnueValue(args.hidden, args.head, args.output_scale).to(device)
    train_loader = DataLoader(ValueDataset(training), batch_size=args.batch, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(ValueDataset(validation), batch_size=args.batch, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = None
    print(f"device={device} train={len(training)} validation={len(validation)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices, mask, side, target in train_loader:
            prediction = model(indices.to(device), mask.to(device), side.to(device))
            loss = nn.functional.smooth_l1_loss(prediction / 600, target.to(device) / 600)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result = metrics(model, validation_loader, device)
        print(f"epoch={epoch} mae_cp={result['maeCp']:.2f} rmse_cp={result['rmseCp']:.2f}", flush=True)
        if best is None or result["maeCp"] < best["maeCp"]:
            best = result
            torch.save(model.state_dict(), args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    export_model(model, args.output, {"validation": best, "teacherFiles": [str(item) for item in args.data], "seed": args.seed})
    print(json.dumps({"output": args.output, "validation": best}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--output", default="models/value-nnue.json")
    parser.add_argument("--checkpoint", default="models/value-nnue.pt")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--head", type=int, default=32)
    parser.add_argument("--output-scale", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
