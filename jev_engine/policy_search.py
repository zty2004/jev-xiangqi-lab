"""PUCT search that applies the Jev policy and value at every expanded node."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from jev_engine.xiangqi import (Move, Position, legal_moves, make_move, move_name,
                                parse_fen, to_fen)
from train.choice_model import (encode_moves, encode_position, load_model,
                                select_device)


class PolicyValueEvaluator(Protocol):
    def evaluate_batch(self, positions: list[Position], moves: list[list[Move]]) \
            -> list[tuple[list[float], float]]:
        """Return legal-move priors and side-to-move values in [-1, 1]."""


class TorchChoiceEvaluator:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.device = select_device(device)
        self.model, self.temperature = load_model(checkpoint, self.device)

    def evaluate_batch(self, positions: list[Position], moves: list[list[Move]]):
        if len(positions) != len(moves) or not positions:
            raise ValueError("positions and legal moves must form a nonempty batch")
        boards = torch.stack([
            encode_position(to_fen(position), input_channels=self.model.input_channels)
            for position in positions
        ]).to(self.device)
        counts = [len(items) for items in moves]
        maximum = max(counts)
        move_tensor = torch.zeros((len(positions), maximum, 2), dtype=torch.long)
        mask = torch.zeros((len(positions), maximum), dtype=torch.bool)
        for index, (position, items) in enumerate(zip(positions, moves)):
            turn = "w" if position.side == "red" else "b"
            names = [move_name(move) for move in items]
            encoded = encode_moves(names, turn)
            move_tensor[index, :len(items)] = encoded
            mask[index, :len(items)] = True
        with torch.no_grad():
            logits, values = self.model(boards, move_tensor.to(self.device), mask.to(self.device))
            probabilities = torch.softmax(logits / self.temperature, dim=1).cpu()
            values = values.cpu()
        return [(probabilities[index, :count].tolist(),
                 float(values[index].item()))
                for index, count in enumerate(counts)]


@dataclass
class Node:
    position: Position
    prior: float = 1.0
    move: Move | None = None
    parent: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0
    expanded: bool = False
    children: list["Node"] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    move: str | None
    value: float
    visits: int
    expanded_nodes: int
    model_evaluations: int
    maximum_depth: int
    elapsed_ms: float
    policy: tuple[tuple[str, int, float, float], ...]


def terminal_value(position: Position, moves: list[Move]) -> float | None:
    own_king = "K" if position.side == "red" else "k"
    enemy_king = "k" if position.side == "red" else "K"
    if own_king not in position.board:
        return -1.0
    if enemy_king not in position.board:
        return 1.0
    if not moves:
        return -1.0
    if position.halfmove >= 120:
        return 0.0
    return None


class PuctSearch:
    def __init__(self, evaluator: PolicyValueEvaluator, c_puct: float = 1.5,
                 batch_size: int = 16, maximum_depth: int = 64):
        if c_puct <= 0 or batch_size < 1 or maximum_depth < 1:
            raise ValueError("invalid PUCT search configuration")
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.depth_limit = maximum_depth
        self.expanded_nodes = 0
        self.model_evaluations = 0
        self.maximum_depth = 0

    def select_child(self, node: Node) -> Node | None:
        available = [child for child in node.children if not child.virtual_visits]
        if not available:
            return None
        parent_visits = max(1, node.visits + node.virtual_visits)
        scale = math.sqrt(parent_visits)
        return max(available, key=lambda child:
                   -child.mean_value + self.c_puct * child.prior * scale /
                   (1 + child.visits + child.virtual_visits))

    def select_leaf(self, root: Node) -> tuple[list[Node], list[Move], float | None, bool] | None:
        node, path, depth = root, [root], 0
        while node.expanded and node.children and depth < self.depth_limit:
            child = self.select_child(node)
            if child is None:
                return None
            node = child
            path.append(node)
            depth += 1
        self.maximum_depth = max(self.maximum_depth, depth)
        moves = legal_moves(node.position)
        result = terminal_value(node.position, moves)
        for item in path:
            item.virtual_visits += 1
        return path, moves, result, depth < self.depth_limit

    @staticmethod
    def clear_virtual(path: list[Node]) -> None:
        for node in path:
            node.virtual_visits -= 1

    @staticmethod
    def backup(path: list[Node], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def expand(self, node: Node, moves: list[Move], priors: list[float]) -> None:
        if len(moves) != len(priors) or any(not math.isfinite(item) or item < 0 for item in priors):
            raise ValueError("evaluator returned an invalid legal-move distribution")
        total = sum(priors)
        if total <= 0:
            raise ValueError("evaluator returned an empty legal-move distribution")
        node.children = [Node(make_move(node.position, move), prior / total, move, node)
                         for move, prior in zip(moves, priors)]
        node.expanded = True
        self.expanded_nodes += 1

    def run(self, position: Position, simulations: int) -> SearchResult:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        started = time.perf_counter()
        self.expanded_nodes = self.model_evaluations = self.maximum_depth = 0
        root = Node(position)
        root_moves = legal_moves(position)
        terminal = terminal_value(position, root_moves)
        if terminal is not None:
            return SearchResult(None, terminal, 0, 0, 0, 0,
                                (time.perf_counter() - started) * 1000, ())
        root_priors, _ = self.evaluator.evaluate_batch([position], [root_moves])[0]
        self.model_evaluations += 1
        self.expand(root, root_moves, root_priors)

        completed = 0
        while completed < simulations:
            pending = []
            target = min(self.batch_size, simulations - completed)
            for _ in range(target):
                selected = self.select_leaf(root)
                if selected is None:
                    break
                path, moves, value, may_expand = selected
                if value is not None:
                    self.clear_virtual(path)
                    self.backup(path, value)
                    completed += 1
                else:
                    pending.append((path, moves, may_expand))
            if not pending:
                if completed >= simulations:
                    break
                continue
            positions = [path[-1].position for path, _, _ in pending]
            move_batches = [moves for _, moves, _ in pending]
            evaluations = self.evaluator.evaluate_batch(positions, move_batches)
            self.model_evaluations += len(evaluations)
            for (path, moves, may_expand), (priors, value) in zip(pending, evaluations):
                leaf = path[-1]
                self.clear_virtual(path)
                if may_expand and not leaf.expanded:
                    self.expand(leaf, moves, priors)
                self.backup(path, max(-1.0, min(1.0, value)))
                completed += 1

        ordered = sorted(root.children,
                         key=lambda child: (child.visits, child.prior), reverse=True)
        best = ordered[0] if ordered else None
        policy = tuple((move_name(child.move), child.visits, child.prior,
                        -child.mean_value) for child in ordered)
        return SearchResult(move_name(best.move) if best else None,
                            root.mean_value, root.visits, self.expanded_nodes,
                            self.model_evaluations, self.maximum_depth,
                            (time.perf_counter() - started) * 1000, policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fen", default=None)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-depth", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = str(Path(args.model))
    search = PuctSearch(TorchChoiceEvaluator(checkpoint, args.device), args.c_puct,
                        args.batch_size, args.maximum_depth)
    result = search.run(parse_fen(args.fen) if args.fen else parse_fen(), args.simulations)
    print(json.dumps({**result.__dict__, "policy": result.policy}, ensure_ascii=False))


if __name__ == "__main__":
    main()
