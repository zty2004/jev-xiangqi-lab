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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PIECES = "KABNRCPkabnrcp"
FILES = "abcdefghi"
PHASE_UNITS = {"r": 4, "c": 2, "n": 2, "b": 1, "a": 1, "p": 0, "k": 0}
MOVE_DELTAS = ([(dx, 0) for dx in range(-8, 9) if dx] +
               [(0, dy) for dy in range(-9, 10) if dy] +
               [(dx, dy) for dx, dy in ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                         (1, -2), (1, 2), (2, -1), (2, 1))] +
               [(dx, dy) for dy in (-1, 1) for dx in (-1, 1)] +
               [(dx, dy) for dy in (-2, 2) for dx in (-2, 2)])
MOVE_DELTA_INDEX = {delta: index for index, delta in enumerate(MOVE_DELTAS)}


def game_phase(fen):
    tokens = fen.split()
    board = board_pieces(fen, tokens[1])
    units = sum(PHASE_UNITS[piece.lower()] for piece in board if piece != ".")
    non_kings = sum(piece.lower() != "k" for piece in board if piece != ".")
    fullmove = int(tokens[5]) if len(tokens) >= 6 else 1
    if units <= 12 or non_kings <= 8:
        return "endgame"
    if fullmove <= 12 and units >= 32:
        return "opening"
    return "middlegame"


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


def mirror_move(move):
    return FILES[8 - FILES.index(move[0])] + move[1] + FILES[8 - FILES.index(move[2])] + move[3]


def mirror_fen(fen):
    tokens = fen.split()
    mirrored = []
    for rank in tokens[0].split("/"):
        expanded = []
        for token in rank:
            expanded.extend(["."] * int(token) if token.isdigit() else [token])
        if len(expanded) != 9:
            raise ValueError("invalid FEN rank")
        compressed, empty = "", 0
        for token in reversed(expanded):
            if token == ".":
                empty += 1
            else:
                if empty:
                    compressed += str(empty)
                    empty = 0
                compressed += token
        if empty:
            compressed += str(empty)
        mirrored.append(compressed)
    tokens[0] = "/".join(mirrored)
    return " ".join(tokens)


def encode_moves(moves, turn):
    return torch.tensor([[square_index(move[:2], turn), square_index(move[2:], turn)] for move in moves], dtype=torch.long)


def move_plane_index(source, target):
    delta = (target % 9 - source % 9, target // 9 - source // 9)
    if delta not in MOVE_DELTA_INDEX:
        raise ValueError(f"unsupported Xiangqi move displacement: {delta}")
    return MOVE_DELTA_INDEX[delta]


def target_distribution(row, best_weight=0.3):
    moves = row["legal"]
    if row.get("source") == "opening-book":
        targets = torch.tensor([row["policy"].get(move, 0.0) for move in moves], dtype=torch.float32)
        return targets / targets.sum(), 0.0
    if row.get("searchPolicy"):
        policy = row["searchPolicy"]
        if len({item.get("move") for item in policy}) != len(policy):
            raise ValueError("duplicate self-play search move")
        probabilities = {item.get("move"): item.get("probability") for item in policy}
        if any(move not in moves or not isinstance(probability, (int, float)) or
               not math.isfinite(probability) or probability < 0
               for move, probability in probabilities.items()):
            raise ValueError("invalid self-play search policy")
        targets = torch.tensor([probabilities.get(move, 0.0) for move in moves], dtype=torch.float32)
        if targets.sum() <= 0:
            raise ValueError("empty self-play search policy")
        value = float(math.tanh(float(row.get("value", 0)) / 700))
        return targets / targets.sum(), value
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
    def __init__(self, rows, input_channels=16, value_head="scalar", policy_target="soft", mirror_augmentation=False):
        self.rows = rows
        self.input_channels = input_channels
        self.value_head = value_head
        self.policy_target = policy_target
        self.mirror_augmentation = mirror_augmentation

    def __len__(self):
        return len(self.rows) * (2 if self.mirror_augmentation else 1)

    def __getitem__(self, index):
        mirrored = self.mirror_augmentation and index % 2 == 1
        if self.mirror_augmentation:
            index //= 2
        row = self.rows[index]
        fen = mirror_fen(row["fen"]) if mirrored else row["fen"]
        legal = [mirror_move(move) for move in row["legal"]] if mirrored else row["legal"]
        previous = [mirror_fen(item) for item in row.get("previous", [])] if mirrored else row.get("previous")
        turn = fen.split()[1]
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
        return (encode_position(fen, previous, row.get("repetitionCount", 1), self.input_channels),
                encode_moves(legal, turn), policy, value_tensor,
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
    def __init__(self, channels=64, blocks=4, input_channels=16, value_head="scalar", value_loss_weight=0.2,
                 policy_features="v1"):
        super().__init__()
        self.input_channels = input_channels
        self.value_head = value_head
        self.value_loss_weight = value_loss_weight
        self.policy_features = policy_features
        self.stem = nn.Sequential(nn.Conv2d(input_channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        if policy_features == "attention":
            policy_channels = max(32, channels // 2)
            self.policy_source = nn.Conv2d(channels, policy_channels, 1)
            self.policy_target = nn.Conv2d(channels, policy_channels, 1)
            self.policy_geometry = nn.Linear(4, 1)
            self.policy_scale = math.sqrt(policy_channels)
        elif policy_features == "planes":
            self.policy_planes = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, len(MOVE_DELTAS), 1))
            lookup = torch.full((19, 17), -1, dtype=torch.long)
            for (dx, dy), index in MOVE_DELTA_INDEX.items():
                lookup[dy + 9, dx + 8] = index
            self.register_buffer("move_plane_lookup", lookup)
        else:
            policy_inputs = channels * 3 if policy_features == "v1" else channels * 5 + 4
            self.policy = nn.Sequential(nn.Linear(policy_inputs, channels * 2), nn.ReLU(), nn.Linear(channels * 2, 1))
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                   nn.Linear(channels, 3 if value_head == "wdl" else 1),
                                   *([] if value_head == "wdl" else [nn.Tanh()]))

    def forward(self, boards, moves, mask):
        features = self.blocks(self.stem(boards))
        flat = features.flatten(2).transpose(1, 2)
        pooled = features.mean(dim=(2, 3))
        batch, count = moves.shape[:2]
        source_index = moves[:, :, 0].clamp(min=0).unsqueeze(-1).expand(batch, count, flat.shape[-1])
        target_index = moves[:, :, 1].clamp(min=0).unsqueeze(-1).expand(batch, count, flat.shape[-1])
        source_x, source_y = moves[:, :, 0] % 9, moves[:, :, 0] // 9
        target_x, target_y = moves[:, :, 1] % 9, moves[:, :, 1] // 9
        geometry = torch.stack((source_x / 8, source_y / 9,
                                (target_x - source_x) / 8, (target_y - source_y) / 9), dim=-1)
        if self.policy_features == "attention":
            policy_source = self.policy_source(features).flatten(2).transpose(1, 2)
            policy_target = self.policy_target(features).flatten(2).transpose(1, 2)
            policy_channels = policy_source.shape[-1]
            source = policy_source.gather(1, moves[:, :, 0].clamp(min=0).unsqueeze(-1).expand(batch, count, policy_channels))
            target = policy_target.gather(1, moves[:, :, 1].clamp(min=0).unsqueeze(-1).expand(batch, count, policy_channels))
            logits = (source * target).sum(dim=-1) / self.policy_scale + self.policy_geometry(geometry).squeeze(-1)
        elif self.policy_features == "planes":
            action_planes = self.policy_planes(features).permute(0, 2, 3, 1).reshape(batch, 90, len(MOVE_DELTAS))
            source_actions = action_planes.gather(1, moves[:, :, 0].clamp(min=0).unsqueeze(-1)
                                                  .expand(batch, count, len(MOVE_DELTAS)))
            plane_index = self.move_plane_lookup[(target_y - source_y + 9).clamp(0, 18),
                                                 (target_x - source_x + 8).clamp(0, 16)]
            if bool((plane_index[mask] < 0).any()):
                raise ValueError("legal move cannot be represented by the policy planes")
            logits = source_actions.gather(2, plane_index.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        else:
            source = flat.gather(1, source_index)
            target = flat.gather(1, target_index)
            pooled_moves = pooled.unsqueeze(1).expand(-1, count, -1)
            policy_input = torch.cat((source, target, pooled_moves), dim=-1)
            if self.policy_features == "v2":
                policy_input = torch.cat((policy_input, target - source, target * source, geometry), dim=-1)
            logits = self.policy(policy_input).squeeze(-1)
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


def source_sample_weights(rows, source_weights):
    """Weights affect training draws only; held-out game splits stay unchanged."""
    return [(1.0 if row.get("source") == "opening-book" else
             source_weights.get(int(str(row["game"]).split(":", 1)[0]), 1.0)) *
            float(row.get("trainingWeight", 1.0))
            for row in rows]


def source_index(row):
    game = str(row["game"])
    return int(game.split(":", 1)[0]) if ":" in game else 0


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
    nll = brier = hits = top3_hits = top5_hits = reciprocal_rank = confidence_sum = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for logits, best in predictions:
        probabilities = torch.softmax(logits / temperature, dim=0)
        ranking = torch.argsort(probabilities, descending=True)
        chosen = int(ranking[0])
        hit = float(chosen == best)
        rank = int((ranking == best).nonzero(as_tuple=True)[0][0]) + 1
        confidence = float(probabilities[chosen])
        nll -= math.log(max(float(probabilities[best]), 1e-30))
        brier += float((probabilities.square().sum() - 2 * probabilities[best] + 1).item())
        hits += hit
        top3_hits += float(rank <= 3)
        top5_hits += float(rank <= 5)
        reciprocal_rank += 1 / rank
        confidence_sum += confidence
        bucket = bins[min(9, int(confidence * 10))]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += hit
    count = len(predictions)
    ece = sum(abs(accuracy - confidence) for size, confidence, accuracy in bins if size) / count
    return {"count": count, "top1": hits / count, "top3": top3_hits / count,
            "top5": top5_hits / count, "mean_reciprocal_rank": reciprocal_rank / count, "nll": nll / count,
            "brier": brier / count, "mean_top_probability": confidence_sum / count, "ece10": ece}


def fit_temperature(predictions):
    # Scalar temperature only. Calibration rows do not update network weights.
    candidates = [math.exp(math.log(0.25) + index * math.log(32) / 160) for index in range(161)]
    return min(candidates, key=lambda value: policy_metrics(predictions, value)["nll"])


def train(args):
    torch.manual_seed(args.seed)
    source_weights = {}
    for spec in args.source_weight:
        try:
            index_text, weight_text = spec.split(":", 1)
            index, weight = int(index_text), float(weight_text)
        except ValueError as error:
            raise ValueError(f"invalid source weight {spec!r}; expected INDEX:WEIGHT") from error
        if not 0 <= index < len(args.data) or not math.isfinite(weight) or weight <= 0 or index in source_weights:
            raise ValueError(f"invalid or duplicate source weight {spec!r}")
        source_weights[index] = weight
    rows = load_teacher_sources(args.data, args.history_data)
    training, validation, calibration, test = split_rows(rows, args.seed)
    training_only_sources = set(args.training_only_row_source)
    if training_only_sources:
        validation = [row for row in validation if row.get("source") not in training_only_sources]
        calibration = [row for row in calibration if row.get("source") not in training_only_sources]
        test = [row for row in test if row.get("source") not in training_only_sources]
        if not all((validation, calibration, test)):
            raise ValueError("training-only row sources removed an entire held-out split")
    if args.validation_source_index is not None:
        if not 0 <= args.validation_source_index < len(args.data):
            raise ValueError("validation source index is outside the teacher sources")
        validation = [row for row in validation if source_index(row) == args.validation_source_index]
        if not validation:
            raise ValueError("validation source has no positions in the fixed split")
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
    weighted_rows = source_weights or any(float(row.get("trainingWeight", 1.0)) != 1.0 for row in training)
    sample_weights = source_sample_weights(training, source_weights) if weighted_rows else None
    if sample_weights and args.mirror_augmentation:
        sample_weights = [weight for weight in sample_weights for _ in range(2)]
    sampler = (WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True,
                                     generator=torch.Generator().manual_seed(args.seed)) if sample_weights else None)
    train_dataset = TeacherDataset(training, input_channels, args.value_head, args.policy_target, args.mirror_augmentation)
    loader_options = {"num_workers": args.loader_workers, "pin_memory": device.type == "cuda"}
    if args.loader_workers:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, collate_fn=collate, **loader_options)
    validation_loader = DataLoader(TeacherDataset(validation, input_channels, args.value_head, args.policy_target),
                                   batch_size=args.batch, collate_fn=collate, **loader_options)
    model = ChoiceNet(args.channels, args.blocks, input_channels, args.value_head, args.value_loss_weight,
                      args.policy_features).to(device)
    if args.init_model:
        initial = torch.load(args.init_model, map_location=device, weights_only=True)
        expected = {"channels": args.channels, "blocks": args.blocks, "input_channels": input_channels,
                    "value_head": args.value_head}
        actual = {key: initial.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"initial model architecture mismatch: expected {expected}, got {actual}")
        initialize_from_checkpoint(model, initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max" if args.selection_metric == "top1" else "min",
        factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)
        if args.lr_patience else None)
    print(f"device={device} train={len(training)} validation={len(validation)} calibration={len(calibration)} test={len(test)}", flush=True)
    best_loss = float("inf")
    best_accuracy = -1.0
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} val_loss={validation_loss:.4f} top1={accuracy:.3f} lr={current_lr:.2g}", flush=True)
        improved = (accuracy > best_accuracy if args.selection_metric == "top1" else validation_loss < best_loss)
        if improved:
            best_loss = validation_loss
            best_accuracy = accuracy
            stale_epochs = 0
            torch.save({"state_dict": model.state_dict(), "channels": args.channels, "blocks": args.blocks,
                        "input_channels": input_channels, "value_head": args.value_head,
                        "policy_features": args.policy_features,
                        "value_loss_weight": args.value_loss_weight, "policy_target": args.policy_target,
                        "selection_metric": args.selection_metric,
                        "lr_scheduler": ({"kind": "plateau", "patience": args.lr_patience,
                                          "factor": 0.5, "min_lr": args.min_lr}
                                         if scheduler else None),
                        "mirror_augmentation": args.mirror_augmentation,
                        "epoch": epoch, "validation_loss": validation_loss, "validation_top1": accuracy,
                        "seed": args.seed, "teacher_sha256": sha256_file(args.data[0]) if len(args.data) == 1 else None,
                        "source_weights": source_weights,
                        "training_only_row_sources": sorted(training_only_sources),
                        "initial_model": ({"file": str(args.init_model), "sha256": sha256_file(args.init_model)}
                                          if args.init_model else None),
                        "teacher_sources": [{"file": str(filename), "sha256": sha256_file(filename)} for filename in args.data],
                        "history_sha256": sha256_file(args.history_data) if args.history_data else None,
                        "opening_sha256": sha256_file(args.opening_data) if args.opening_data else None,
                        "split_sizes": {"train": len(training), "validation": len(validation),
                                        "calibration": len(calibration), "test": len(test)},
                        "effective_training_examples": len(train_dataset),
                        "loader_workers": args.loader_workers}, args.output)
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"early_stop={epoch} patience={args.patience}", flush=True)
                break
        if scheduler:
            scheduler.step(accuracy if args.selection_metric == "top1" else validation_loss)
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
                      checkpoint.get("value_loss_weight", 0.2), checkpoint.get("policy_features", "v1")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("temperature", 1.0))


def initialize_from_checkpoint(model, checkpoint):
    source_features = checkpoint.get("policy_features", "v1")
    if source_features == model.policy_features:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return
    if source_features != "v1" or model.policy_features != "v2":
        raise ValueError(f"unsupported policy feature migration: {source_features} to {model.policy_features}")
    source = checkpoint["state_dict"]
    target = model.state_dict()
    for key, value in source.items():
        if key == "policy.0.weight":
            target[key].zero_()
            target[key][:, :value.shape[1]].copy_(value)
        elif target[key].shape == value.shape:
            target[key].copy_(value)
        else:
            raise ValueError(f"cannot migrate checkpoint tensor {key}: {value.shape} to {target[key].shape}")
    model.load_state_dict(target, strict=True)


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
    train_parser.add_argument("--mirror-augmentation", action="store_true",
                              help="add a horizontally mirrored copy of every training position")
    train_parser.add_argument("--source-weight", action="append", default=[],
                              help="training-only sampling weight INDEX:WEIGHT for a --data source; repeatable")
    train_parser.add_argument("--training-only-row-source", action="append", default=[],
                              help="keep rows with this source in training but omit them from held-out metrics")
    train_parser.add_argument("--validation-source-index", type=int,
                              help="choose checkpoints using only this --data source from the fixed validation split")
    train_parser.add_argument("--init-model", help="compatible checkpoint used to initialize fine-tuning")
    train_parser.add_argument("--patience", type=int, default=0, help="stop after this many epochs without validation improvement; 0 disables")
    train_parser.add_argument("--selection-metric", choices=["loss", "top1"], default="loss",
                              help="metric used for checkpoint selection and early stopping")
    train_parser.add_argument("--output", default="choice-model.pt")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=128)
    train_parser.add_argument("--loader-workers", type=int, default=0,
                              help="parallel data-encoding workers; use 0 for in-process loading")
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--lr-patience", type=int, default=0,
                              help="halve learning rate after this many validation plateaus; 0 disables")
    train_parser.add_argument("--min-lr", type=float, default=1e-6)
    train_parser.add_argument("--channels", type=int, default=64)
    train_parser.add_argument("--blocks", type=int, default=4)
    train_parser.add_argument("--policy-features", choices=["v1", "v2", "attention", "planes"], default="v1")
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
    evaluate_parser.add_argument("--phase", choices=["opening", "middlegame", "endgame"],
                                 help="report only this phase from the fixed held-out split")
    evaluate_parser.add_argument("--test-source-index", type=int, action="append",
                                 help="report only these --data sources after making the shared split; repeatable")
    args = parser.parse_args()
    if args.command == "train":
        if not 0 <= args.value_loss_weight <= 1:
            raise ValueError("value loss weight must be between 0 and 1")
        if args.lr_patience < 0 or args.min_lr <= 0 or args.min_lr > args.lr:
            raise ValueError("learning-rate schedule must have nonnegative patience and 0 < min-lr <= lr")
        if args.loader_workers < 0:
            raise ValueError("loader-workers must be nonnegative")
        train(args)
    else:
        device = select_device(args.device)
        model, temperature = load_model(args.model, device)
        if args.command == "evaluate":
            rows = load_teacher_sources(args.data, args.history_data)
            _, _, _, test = split_rows(rows, args.seed)
            if args.test_source_index:
                if any(index < 0 or index >= len(args.data) for index in args.test_source_index):
                    raise ValueError("test source index is outside the teacher sources")
                selected_sources = set(args.test_source_index)
                test = [row for row in test if source_index(row) in selected_sources]
            if args.phase:
                test = [row for row in test if game_phase(row["fen"]) == args.phase]
            if args.exclude_data:
                excluded = {" ".join(row["fen"].split()[:2]) for filename in args.exclude_data
                            for row in load_rows(filename)}
                test = [row for row in test if " ".join(row["fen"].split()[:2]) not in excluded]
            predictions = collect_predictions(model, test, device, args.batch)
            print(json.dumps({"temperature": temperature, "phase": args.phase,
                              "test": policy_metrics(predictions, temperature),
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
