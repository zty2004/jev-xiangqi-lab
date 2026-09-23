"""Train a local Jev-style legal-move choice model from Pikafish teacher rows."""

import argparse
import hashlib
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


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_pieces(fen, perspective):
    board_text, turn = fen.split()[:2]
    if turn not in ("w", "b"):
        raise ValueError("invalid FEN side")
    board = []
    for rank in board_text.split("/"):
        for token in rank:
            if token.isdigit():
                board.extend(["."] * int(token))
            else:
                board.append({"H": "N", "h": "n", "E": "B", "e": "b"}.get(token, token))
    if len(board) != 90:
        raise ValueError("invalid FEN board")
    if perspective == "b":
        board = [piece.swapcase() if piece != "." else "." for piece in reversed(board)]
    return board


def encode_position(fen, previous=None, repetition_count=1, input_channels=16):
    turn = fen.split()[1]
    planes = torch.zeros((input_channels, 10, 9), dtype=torch.float32)
    board = board_pieces(fen, turn)
    for index, piece in enumerate(board):
        if piece != ".":
            planes[PIECES.index(piece), index // 9, index % 9] = 1
    planes[14] = torch.arange(9, dtype=torch.float32).view(1, 9).expand(10, 9) / 8
    planes[15] = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 9) / 9
    if input_channels == 46:
        for frame, older in enumerate(reversed((previous or [])[-2:])):
            for index, piece in enumerate(board_pieces(older, turn)):
                if piece != ".":
                    planes[16 + 14 * frame + PIECES.index(piece), index // 9, index % 9] = 1
        halfmove = int(fen.split()[4]) if len(fen.split()) >= 5 else 0
        planes[44] = min(120, max(0, halfmove)) / 120
        planes[45] = min(3, max(1, repetition_count)) / 3
    elif input_channels != 16:
        raise ValueError("unsupported input channel count")
    return planes


def square_index(name, turn):
    index = (9 - int(name[1])) * 9 + FILES.index(name[0])
    return 89 - index if turn == "b" else index


def encode_moves(moves, turn):
    return torch.tensor([[square_index(move[:2], turn), square_index(move[2:], turn)] for move in moves], dtype=torch.long)


def target_distribution(row, best_weight=0.3):
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
            targets[moves.index(item["move"])] += (1 - best_weight) * weight
    else:
        targets[moves.index(row["best"])] += 1 - best_weight
    targets[moves.index(row["best"])] += best_weight
    value = 0.0
    if candidates:
        first = candidates[0]
        value = float(math.tanh((30000 * (1 if first["score"] > 0 else -1) if first["scoreType"] == "mate" else first["score"]) / 700))
    return targets, value


class TeacherDataset(Dataset):
    def __init__(self, rows, input_channels=16, value_head="scalar", policy_target="soft"):
        self.rows = rows
        self.input_channels = input_channels
        self.value_head = value_head
        self.policy_target = policy_target

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        turn = row["fen"].split()[1]
        policy, value = target_distribution(row, 0.7 if self.policy_target == "dominant" else 0.3)
        if self.policy_target == "best" and row.get("source") != "opening-book":
            policy = torch.zeros_like(policy)
            policy[row["legal"].index(row["best"])] = 1
        if self.value_head == "wdl":
            winner = row.get("winner")
            value = {"draw": 1, "red": 2 if turn == "w" else 0,
                     "black": 2 if turn == "b" else 0}.get(winner, 0)
            weight = float(winner is not None)
            value_tensor = torch.tensor(value, dtype=torch.long)
        else:
            weight = float(row.get("source") != "opening-book")
            value_tensor = torch.tensor(value, dtype=torch.float32)
        return (encode_position(row["fen"], row.get("previous"), row.get("repetitionCount", 1), self.input_channels),
                encode_moves(row["legal"], turn), policy, value_tensor,
                row["legal"].index(row["best"]), torch.tensor(weight))


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
    def __init__(self, channels=64, blocks=4, input_channels=16, value_head="scalar", value_loss_weight=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.value_head = value_head
        self.value_loss_weight = value_loss_weight
        self.stem = nn.Sequential(nn.Conv2d(input_channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(nn.Linear(channels * 3, channels * 2), nn.ReLU(), nn.Linear(channels * 2, 1))
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                   nn.Linear(channels, 3 if value_head == "wdl" else 1),
                                   *([] if value_head == "wdl" else [nn.Tanh()]))

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
        value = self.value(pooled)
        if self.value_head == "scalar":
            value = value.squeeze(-1)
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


def load_teacher_sources(filenames, history_filename=None):
    """Combine teacher files, attaching validated history to the last source."""
    if len(filenames) == 1:
        rows = load_rows(filenames[0])
        return attach_history_labels(rows, history_filename, filenames[0]) if history_filename else rows
    rows = []
    for source_index, filename in enumerate(filenames):
        source_rows = load_rows(filename)
        if history_filename and source_index == len(filenames) - 1:
            source_rows = attach_history_labels(source_rows, history_filename, filename)
        for row in source_rows:
            rows.append({**row, "game": f"{source_index}:{row['game']}"})
    return rows


def attach_history_labels(rows, filename, teacher_filename):
    with open(filename, encoding="utf-8") as handle:
        metadata = json.loads(handle.readline())
        if metadata.get("kind") != "meta" or metadata.get("schema") != "history-labels-v1":
            raise ValueError("unsupported history label file")
        if metadata.get("teacherSha256") != sha256_file(teacher_filename):
            raise ValueError("history labels do not match teacher data")
        labels = {}
        for line in handle:
            item = json.loads(line)
            if item.get("kind") != "history-label":
                raise ValueError("invalid history label record")
            key = (item["game"], item["ply"])
            if key in labels or item.get("winner") not in (None, "red", "black", "draw"):
                raise ValueError("duplicate or invalid history label")
            if not isinstance(item.get("previous"), list) or len(item["previous"]) > 2 or not isinstance(item.get("repetitionCount"), int) or item["repetitionCount"] < 1:
                raise ValueError("invalid history input")
            labels[key] = item
    if len(labels) != len(rows):
        raise ValueError("history label count does not match teacher data")
    enriched = []
    for row in rows:
        item = labels.pop((row["game"], row["ply"]), None)
        if item is None:
            raise ValueError("missing history label")
        enriched.append({**row, "previous": item["previous"], "repetitionCount": item["repetitionCount"], "winner": item["winner"]})
    if labels:
        raise ValueError("orphan history labels")
    return enriched


def split_rows(rows, seed):
    games = sorted({row["game"] for row in rows})
    if len(games) < 4:
        raise ValueError("need at least four teacher games for disjoint train/validation/calibration/test splits")
    shuffled = games[:]
    random.Random(seed).shuffle(shuffled)
    game_sets = [set(shuffled[index::10]) for index in (0, 1, 2)]
    if not all(game_sets):
        raise ValueError("need at least three held-out teacher games")
    # Test has first claim on a repeated position; no identical board/turn can cross splits.
    def unique(items, excluded):
        seen = set(excluded)
        result = []
        for row in items:
            key = " ".join(row["fen"].split()[:2])
            if key not in seen:
                result.append(row)
                seen.add(key)
        return result, seen
    test, test_keys = unique((row for row in rows if row["game"] in game_sets[2]), set())
    calibration, calibration_keys = unique((row for row in rows if row["game"] in game_sets[1]), test_keys)
    validation, validation_keys = unique((row for row in rows if row["game"] in game_sets[0]), calibration_keys)
    training, _ = unique((row for row in rows if all(row["game"] not in group for group in game_sets)), validation_keys)
    if not all((training, validation, calibration, test)):
        raise ValueError("one split is empty after removing repeated positions")
    return training, validation, calibration, test


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def value_loss_for(model, predicted, targets, weights):
    if model.value_head == "wdl":
        errors = nn.functional.cross_entropy(predicted, targets, reduction="none")
    else:
        errors = (predicted - targets).square()
    return (errors * weights).sum() / weights.sum().clamp(min=1)


def evaluate(model, loader, device):
    model.eval()
    total, hits, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for boards, moves, targets, mask, values, best, value_weights in loader:
            boards, moves, targets, mask, values, best, value_weights = (item.to(device) for item in (boards, moves, targets, mask, values, best, value_weights))
            logits, predicted_value = model(boards, moves, mask)
            policy_loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = value_loss_for(model, predicted_value, values, value_weights)
            loss = policy_loss + model.value_loss_weight * value_loss
            loss_sum += loss.item() * boards.shape[0]
            hits += (logits.argmax(dim=1) == best).sum().item()
            total += boards.shape[0]
    return loss_sum / total, hits / total


def collect_predictions(model, rows, device, batch_size):
    loader = DataLoader(TeacherDataset(rows, model.input_channels, model.value_head), batch_size=batch_size, collate_fn=collate)
    predictions = []
    model.eval()
    with torch.no_grad():
        for boards, moves, _, mask, _, best, _ in loader:
            logits, _ = model(boards.to(device), moves.to(device), mask.to(device))
            counts = mask.sum(dim=1).tolist()
            predictions.extend((logits[i, :count].cpu(), int(best[i])) for i, count in enumerate(counts))
    return predictions


def wdl_metrics(model, rows, device, batch_size):
    if model.value_head != "wdl" or model.value_loss_weight == 0:
        return None
    loader = DataLoader(TeacherDataset(rows, model.input_channels, "wdl"), batch_size=batch_size, collate_fn=collate)
    count = hits = nll = brier = 0.0
    model.eval()
    with torch.no_grad():
        for boards, moves, _, mask, values, _, weights in loader:
            _, logits = model(boards.to(device), moves.to(device), mask.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            for index in range(len(boards)):
                if not weights[index]:
                    continue
                target = int(values[index])
                prediction = probabilities[index]
                count += 1
                hits += float(int(prediction.argmax()) == target)
                nll -= math.log(max(float(prediction[target]), 1e-30))
                brier += float((prediction.square().sum() - 2 * prediction[target] + 1).item())
    return {"count": int(count), "accuracy": hits / count, "nll": nll / count, "brier": brier / count} if count else {"count": 0}


def policy_metrics(predictions, temperature):
    if not predictions:
        raise ValueError("no positions to evaluate")
    nll = brier = hits = confidence_sum = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for logits, best in predictions:
        probabilities = torch.softmax(logits / temperature, dim=0)
        chosen = int(probabilities.argmax())
        hit = float(chosen == best)
        confidence = float(probabilities[chosen])
        nll -= math.log(max(float(probabilities[best]), 1e-30))
        brier += float((probabilities.square().sum() - 2 * probabilities[best] + 1).item())
        hits += hit
        confidence_sum += confidence
        bucket = bins[min(9, int(confidence * 10))]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += hit
    count = len(predictions)
    ece = sum(abs(accuracy - confidence) for size, confidence, accuracy in bins if size) / count
    return {"count": count, "top1": hits / count, "nll": nll / count,
            "brier": brier / count, "mean_top_probability": confidence_sum / count, "ece10": ece}


def fit_temperature(predictions):
    # Scalar temperature only. Calibration rows do not update network weights.
    candidates = [math.exp(math.log(0.25) + index * math.log(32) / 160) for index in range(161)]
    return min(candidates, key=lambda value: policy_metrics(predictions, value)["nll"])


def train(args):
    torch.manual_seed(args.seed)
    rows = load_teacher_sources(args.data, args.history_data)
    training, validation, calibration, test = split_rows(rows, args.seed)
    if args.opening_data:
        opening_rows = [row for row in load_rows(args.opening_data) if row.get("source") == "opening-book" and row.get("policy")]
        heldout_keys = {" ".join(row["fen"].split()[:2]) for row in validation + calibration + test}
        training_keys = {" ".join(row["fen"].split()[:2]) for row in training}
        opening_rows = [row for row in opening_rows if " ".join(row["fen"].split()[:2]) not in heldout_keys | training_keys]
        random.Random(args.seed).shuffle(opening_rows)
        training.extend(opening_rows[:args.opening_samples])
    if not training:
        raise ValueError("need at least two teacher positions")
    device = select_device(args.device)
    input_channels = 46 if args.history_data else 16
    if args.value_head == "wdl" and not args.history_data:
        raise ValueError("WDL training requires game outcome labels")
    train_loader = DataLoader(TeacherDataset(training, input_channels, args.value_head, args.policy_target), batch_size=args.batch, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(TeacherDataset(validation, input_channels, args.value_head, args.policy_target), batch_size=args.batch, collate_fn=collate)
    model = ChoiceNet(args.channels, args.blocks, input_channels, args.value_head, args.value_loss_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"device={device} train={len(training)} validation={len(validation)} calibration={len(calibration)} test={len(test)}", flush=True)
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for boards, moves, targets, mask, values, _, value_weights in train_loader:
            boards, moves, targets, mask, values, value_weights = (item.to(device) for item in (boards, moves, targets, mask, values, value_weights))
            logits, predicted_value = model(boards, moves, mask)
            value_loss = value_loss_for(model, predicted_value, values, value_weights)
            loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean() + model.value_loss_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss, accuracy = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} val_loss={validation_loss:.4f} top1={accuracy:.3f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save({"state_dict": model.state_dict(), "channels": args.channels, "blocks": args.blocks,
                        "input_channels": input_channels, "value_head": args.value_head,
                        "value_loss_weight": args.value_loss_weight, "policy_target": args.policy_target,
                        "epoch": epoch, "validation_loss": validation_loss, "validation_top1": accuracy,
                        "seed": args.seed, "teacher_sha256": sha256_file(args.data[0]) if len(args.data) == 1 else None,
                        "teacher_sources": [{"file": str(filename), "sha256": sha256_file(filename)} for filename in args.data],
                        "history_sha256": sha256_file(args.history_data) if args.history_data else None,
                        "opening_sha256": sha256_file(args.opening_data) if args.opening_data else None,
                        "split_sizes": {"train": len(training), "validation": len(validation),
                                        "calibration": len(calibration), "test": len(test)}}, args.output)
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"early_stop={epoch} patience={args.patience}", flush=True)
                break
    model, _ = load_model(args.output, device)
    calibration_predictions = collect_predictions(model, calibration, device, args.batch)
    temperature = fit_temperature(calibration_predictions)
    test_predictions = collect_predictions(model, test, device, args.batch)
    report = {"temperature": temperature, "calibration": policy_metrics(calibration_predictions, temperature),
              "test_uncalibrated": policy_metrics(test_predictions, 1.0), "test_calibrated": policy_metrics(test_predictions, temperature)}
    if args.value_head == "wdl" and args.value_loss_weight > 0:
        report["validation_wdl"] = wdl_metrics(model, validation, device, args.batch)
        report["test_wdl"] = wdl_metrics(model, test, device, args.batch)
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    checkpoint["temperature"] = temperature
    checkpoint["evaluation"] = report
    torch.save(checkpoint, args.output)
    print(json.dumps(report), flush=True)


def load_model(filename, device):
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    model = ChoiceNet(checkpoint["channels"], checkpoint["blocks"],
                      checkpoint.get("input_channels", 16), checkpoint.get("value_head", "scalar"),
                      checkpoint.get("value_loss_weight", 0.2)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("temperature", 1.0))


def rank(model, fen, moves, device, temperature=1.0, history=None):
    if not moves:
        return []
    turn = fen.split()[1]
    current_key = " ".join(fen.split()[:2])
    history = history or []
    previous = history[:-1][-2:] if history and history[-1] == current_key else history[-2:]
    repetition_count = history.count(current_key) + (0 if history and history[-1] == current_key else 1)
    board = encode_position(fen, previous, repetition_count, model.input_channels).unsqueeze(0).to(device)
    move_tensor = encode_moves(moves, turn).unsqueeze(0).to(device)
    mask = torch.ones((1, len(moves)), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(board, move_tensor, mask)
        probabilities = torch.softmax(logits[0] / temperature, dim=0).cpu().tolist()
    concentration = 1.0 if len(moves) == 1 else max(0.0, min(1.0,
        1 - sum(-p * math.log(max(p, 1e-30)) for p in probabilities) / math.log(len(moves))))
    result = {"concentration": concentration,
              "choices": sorted(({"move": move, "probability": probability} for move, probability in zip(moves, probabilities)), key=lambda item: item["probability"], reverse=True)}
    if model.value_head == "wdl" and model.value_loss_weight > 0:
        loss, draw, win = torch.softmax(value[0], dim=0).cpu().tolist()
        result["value"] = win - loss
        result["wdl"] = {"loss": loss, "draw": draw, "win": win}
    elif model.value_loss_weight > 0:
        result["value"] = value.item()
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", action="append", required=True, help="teacher JSONL; repeat to combine files")
    train_parser.add_argument("--opening-data")
    train_parser.add_argument("--history-data", help="validated game-history and outcome labels for the final teacher file")
    train_parser.add_argument("--opening-samples", type=int, default=2000)
    train_parser.add_argument("--value-head", choices=["scalar", "wdl"], default="scalar")
    train_parser.add_argument("--value-loss-weight", type=float, default=0.2)
    train_parser.add_argument("--policy-target", choices=["soft", "dominant", "best"], default="soft")
    train_parser.add_argument("--patience", type=int, default=0, help="stop after this many epochs without validation improvement; 0 disables")
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
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--data", action="append", required=True, help="teacher JSONL; repeat to combine files")
    evaluate_parser.add_argument("--seed", type=int, default=20260923)
    evaluate_parser.add_argument("--batch", type=int, default=128)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--exclude-data", action="append", default=[],
                                 help="exclude FENs found in this JSONL file; repeat for multiple prior training sources")
    evaluate_parser.add_argument("--history-data", help="history labels for the final teacher file")
    args = parser.parse_args()
    if args.command == "train":
        if not 0 <= args.value_loss_weight <= 1:
            raise ValueError("value loss weight must be between 0 and 1")
        train(args)
    else:
        device = select_device(args.device)
        model, temperature = load_model(args.model, device)
        if args.command == "evaluate":
            rows = load_teacher_sources(args.data, args.history_data)
            _, _, _, test = split_rows(rows, args.seed)
            if args.exclude_data:
                excluded = {" ".join(row["fen"].split()[:2]) for filename in args.exclude_data
                            for row in load_rows(filename)}
                test = [row for row in test if " ".join(row["fen"].split()[:2]) not in excluded]
            predictions = collect_predictions(model, test, device, args.batch)
            print(json.dumps({"temperature": temperature, "test": policy_metrics(predictions, temperature),
                              "test_wdl": wdl_metrics(model, test, device, args.batch)}))
        elif args.command == "rank":
            print(json.dumps(rank(model, args.fen, args.moves.split(","), device, temperature), ensure_ascii=False))
        else:
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    print(json.dumps(rank(model, request["fen"], request["moves"], device, temperature,
                                          request.get("history"))), flush=True)
                except Exception as error:
                    print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
