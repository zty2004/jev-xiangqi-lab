"""Train a local Jev-style legal-move choice model from Pikafish teacher rows."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PIECES = "KABNRCPkabnrcp"
FILES = "abcdefghi"


def encode_position(fen):
    board_text, turn = fen.split()[:2]
    board = []
    for rank in board_text.split("/"):
        for token in rank:
            if token.isdigit():
                board.extend(["."] * int(token))
            else:
                board.append({"H": "N", "h": "n", "E": "B", "e": "b"}.get(token, token))
    if len(board) != 90:
        raise ValueError("invalid FEN board")
    if turn == "b":
        board = [piece.swapcase() if piece != "." else "." for piece in reversed(board)]
    planes = torch.zeros((16, 10, 9), dtype=torch.float32)
    for index, piece in enumerate(board):
        if piece != ".":
            planes[PIECES.index(piece), index // 9, index % 9] = 1
    planes[14] = torch.arange(9, dtype=torch.float32).view(1, 9).expand(10, 9) / 8
    planes[15] = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 9) / 9
    return planes


def square_index(name, turn):
    index = (9 - int(name[1])) * 9 + FILES.index(name[0])
    return 89 - index if turn == "b" else index


def encode_moves(moves, turn):
    return torch.tensor([[square_index(move[:2], turn), square_index(move[2:], turn)] for move in moves], dtype=torch.long)


def target_distribution(row):
    moves = row["legal"]
    if row.get("source") == "opening-book":
        targets = torch.tensor([row["policy"].get(move, 0.0) for move in moves], dtype=torch.float32)
        return targets / targets.sum(), 0.0
    targets = torch.zeros(len(moves), dtype=torch.float32)
    candidates = [item for item in row.get("candidates", []) if item["move"] in moves]
    if candidates:
        scores = torch.tensor([30000 * (1 if item["score"] > 0 else -1) if item["scoreType"] == "mate" else item["score"] for item in candidates], dtype=torch.float32)
        weights = torch.softmax((scores - scores.max()).clamp(min=-1500) / 120, dim=0)
        for item, weight in zip(candidates, weights):
            targets[moves.index(item["move"])] += 0.7 * weight
    else:
        targets[moves.index(row["best"])] += 0.7
    targets[moves.index(row["best"])] += 0.3
    value = 0.0
    if candidates:
        first = candidates[0]
        value = float(math.tanh((30000 * (1 if first["score"] > 0 else -1) if first["scoreType"] == "mate" else first["score"]) / 700))
    return targets, value


class TeacherDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        turn = row["fen"].split()[1]
        policy, value = target_distribution(row)
        return encode_position(row["fen"]), encode_moves(row["legal"], turn), policy, torch.tensor(value), row["legal"].index(row["best"]), torch.tensor(0.0 if row.get("source") == "opening-book" else 1.0)


def collate(samples):
    batch = len(samples)
    max_moves = max(sample[1].shape[0] for sample in samples)
    boards = torch.stack([sample[0] for sample in samples])
    moves = torch.zeros((batch, max_moves, 2), dtype=torch.long)
    targets = torch.zeros((batch, max_moves), dtype=torch.float32)
    mask = torch.zeros((batch, max_moves), dtype=torch.bool)
    values = torch.stack([sample[3] for sample in samples])
    best = torch.tensor([sample[4] for sample in samples], dtype=torch.long)
    value_weights = torch.stack([sample[5] for sample in samples])
    for i, sample in enumerate(samples):
        count = sample[1].shape[0]
        moves[i, :count] = sample[1]
        targets[i, :count] = sample[2]
        mask[i, :count] = True
    return boards, moves, targets, mask, values, best, value_weights


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ChoiceNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(nn.Linear(channels * 3, channels * 2), nn.ReLU(), nn.Linear(channels * 2, 1))
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1), nn.Tanh())

    def forward(self, boards, moves, mask):
        features = self.blocks(self.stem(boards))
        flat = features.flatten(2).transpose(1, 2)
        batch, count = moves.shape[:2]
        source_index = moves[:, :, 0].clamp(min=0).unsqueeze(-1).expand(batch, count, flat.shape[-1])
        target_index = moves[:, :, 1].clamp(min=0).unsqueeze(-1).expand(batch, count, flat.shape[-1])
        source = flat.gather(1, source_index)
        target = flat.gather(1, target_index)
        pooled = features.mean(dim=(2, 3))
        pooled_moves = pooled.unsqueeze(1).expand(-1, count, -1)
        logits = self.policy(torch.cat((source, target, pooled_moves), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        value = self.value(pooled).squeeze(-1)
        return logits, value


def load_rows(filename):
    rows = []
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "position" and row.get("best") in row.get("legal", []):
                rows.append(row)
    if not rows:
        raise ValueError("no valid teacher positions")
    return rows


def split_rows(rows, seed):
    games = sorted({row["game"] for row in rows})
    if len(games) >= 2:
        holdout = {game for game in games if game % 10 == 0}
        if not holdout or len(holdout) == len(games):
            holdout = {games[-1]}
        validation = [row for row in rows if row["game"] in holdout]
        validation_keys = {" ".join(row["fen"].split()[:2]) for row in validation}
        training = [row for row in rows if row["game"] not in holdout and " ".join(row["fen"].split()[:2]) not in validation_keys]
        return training, validation
    rng = random.Random(seed)
    copied = rows[:]
    rng.shuffle(copied)
    cutoff = max(1, len(copied) // 5)
    return copied[cutoff:], copied[:cutoff]


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total, hits, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for boards, moves, targets, mask, values, best, value_weights in loader:
            boards, moves, targets, mask, values, best, value_weights = (item.to(device) for item in (boards, moves, targets, mask, values, best, value_weights))
            logits, predicted_value = model(boards, moves, mask)
            policy_loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = (((predicted_value - values) ** 2) * value_weights).sum() / value_weights.sum().clamp(min=1)
            loss = policy_loss + 0.2 * value_loss
            loss_sum += loss.item() * boards.shape[0]
            hits += (logits.argmax(dim=1) == best).sum().item()
            total += boards.shape[0]
    return loss_sum / total, hits / total


def train(args):
    torch.manual_seed(args.seed)
    rows = load_rows(args.data)
    training, validation = split_rows(rows, args.seed)
    if args.opening_data:
        opening_rows = [row for row in load_rows(args.opening_data) if row.get("source") == "opening-book" and row.get("policy")]
        validation_keys = {" ".join(row["fen"].split()[:2]) for row in validation}
        opening_rows = [row for row in opening_rows if " ".join(row["fen"].split()[:2]) not in validation_keys]
        random.Random(args.seed).shuffle(opening_rows)
        training.extend(opening_rows[:args.opening_samples])
    if not training:
        raise ValueError("need at least two teacher positions")
    device = select_device(args.device)
    train_loader = DataLoader(TeacherDataset(training), batch_size=args.batch, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(TeacherDataset(validation), batch_size=args.batch, collate_fn=collate)
    model = ChoiceNet(args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"device={device} train={len(training)} validation={len(validation)}", flush=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for boards, moves, targets, mask, values, _, value_weights in train_loader:
            boards, moves, targets, mask, values, value_weights = (item.to(device) for item in (boards, moves, targets, mask, values, value_weights))
            logits, predicted_value = model(boards, moves, mask)
            value_loss = (((predicted_value - values) ** 2) * value_weights).sum() / value_weights.sum().clamp(min=1)
            loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean() + 0.2 * value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss, accuracy = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} val_loss={validation_loss:.4f} top1={accuracy:.3f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"state_dict": model.state_dict(), "channels": args.channels, "blocks": args.blocks, "epoch": epoch, "validation_loss": validation_loss, "validation_top1": accuracy}, args.output)


def load_model(filename, device):
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    model = ChoiceNet(checkpoint["channels"], checkpoint["blocks"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def rank(model, fen, moves, device):
    if not moves:
        return []
    turn = fen.split()[1]
    board = encode_position(fen).unsqueeze(0).to(device)
    move_tensor = encode_moves(moves, turn).unsqueeze(0).to(device)
    mask = torch.ones((1, len(moves)), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(board, move_tensor, mask)
        probabilities = torch.softmax(logits[0], dim=0).cpu().tolist()
    return {"value": value.item(), "choices": sorted(({"move": move, "probability": probability} for move, probability in zip(moves, probabilities)), key=lambda item: item["probability"], reverse=True)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--opening-data")
    train_parser.add_argument("--opening-samples", type=int, default=2000)
    train_parser.add_argument("--output", default="choice-model.pt")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--channels", type=int, default=64)
    train_parser.add_argument("--blocks", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=20260923)
    train_parser.add_argument("--device", default="auto")
    rank_parser = sub.add_parser("rank")
    rank_parser.add_argument("--model", required=True)
    rank_parser.add_argument("--fen", required=True)
    rank_parser.add_argument("--moves", required=True, help="comma-separated ICCS moves")
    rank_parser.add_argument("--device", default="auto")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        device = select_device(args.device)
        model = load_model(args.model, device)
        if args.command == "rank":
            print(json.dumps(rank(model, args.fen, args.moves.split(","), device), ensure_ascii=False))
        else:
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    print(json.dumps(rank(model, request["fen"], request["moves"], device)), flush=True)
                except Exception as error:
                    print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
