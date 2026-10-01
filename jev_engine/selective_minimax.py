"""Batched Jev-guided selective minimax search.

Every expanded node is evaluated by the Jev policy/value network.  The root
keeps every legal move so recommendations remain complete; internal nodes keep
the model's most probable replies and minimax assumes the opponent chooses the
worst retained continuation.  This is a speculative, policy-pruned tree rather
than visit averaging.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from jev_engine.policy_search import PolicyValueEvaluator, TorchChoiceEvaluator, terminal_value
from jev_engine.xiangqi import Move, Position, legal_moves, make_move, move_name, parse_fen


@dataclass
class Node:
    position: Position
    prior: float = 1.0
    move: Move | None = None
    depth: int = 0
    model_value: float = 0.0
    backed_value: float = 0.0
    terminal: bool = False
    children: list["Node"] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    move: str | None
    value: float
    expanded_nodes: int
    model_evaluations: int
    maximum_depth: int
    elapsed_ms: float
    # move, combined root score, minimax value, Jev root probability
    policy: tuple[tuple[str, float, float, float], ...]


class SelectiveMinimaxSearch:
    def __init__(self, evaluator: PolicyValueEvaluator, internal_width: int = 4,
                 evaluation_batch_size: int = 256, root_prior_weight: float = 0.0):
        if internal_width < 1 or evaluation_batch_size < 1 or root_prior_weight < 0:
            raise ValueError("invalid selective minimax configuration")
        self.evaluator = evaluator
        self.internal_width = internal_width
        self.batch_size = evaluation_batch_size
        self.root_prior_weight = root_prior_weight

    def evaluate_nodes(self, nodes: list[Node], move_batches: list[list[Move]]) \
            -> list[tuple[list[float], float]]:
        results = []
        for start in range(0, len(nodes), self.batch_size):
            end = start + self.batch_size
            results.extend(self.evaluator.evaluate_batch(
                [node.position for node in nodes[start:end]], move_batches[start:end]))
        return results

    def run(self, position: Position, maximum_depth: int) -> SearchResult:
        if maximum_depth < 1:
            raise ValueError("maximum depth must be positive")
        started = time.perf_counter()
        root = Node(position)
        levels: list[list[Node]] = [[root]]
        expanded_nodes = model_evaluations = 0

        for depth in range(maximum_depth + 1):
            frontier = levels[depth]
            evaluated_nodes: list[Node] = []
            evaluated_moves: list[list[Move]] = []
            moves_by_node: dict[int, list[Move]] = {}
            for node in frontier:
                moves = legal_moves(node.position)
                exact = terminal_value(node.position, moves)
                if exact is not None:
                    node.terminal = True
                    node.model_value = node.backed_value = exact
                else:
                    evaluated_nodes.append(node)
                    evaluated_moves.append(moves)
                    moves_by_node[id(node)] = moves
            if evaluated_nodes:
                evaluations = self.evaluate_nodes(evaluated_nodes, evaluated_moves)
                model_evaluations += len(evaluations)
                for node, moves, (priors, value) in zip(evaluated_nodes, evaluated_moves, evaluations):
                    if len(priors) != len(moves) or any(not math.isfinite(item) or item < 0 for item in priors):
                        raise ValueError("evaluator returned an invalid legal-move distribution")
                    total = sum(priors)
                    if total <= 0 or not math.isfinite(value):
                        raise ValueError("evaluator returned an invalid policy or value")
                    priors = [item / total for item in priors]
                    node.model_value = node.backed_value = max(-1.0, min(1.0, value))
                    if depth >= maximum_depth:
                        continue
                    ranked = sorted(zip(moves, priors), key=lambda item: item[1], reverse=True)
                    selected = ranked if node is root else ranked[:self.internal_width]
                    node.children = [Node(make_move(node.position, move), prior, move, depth + 1)
                                     for move, prior in selected]
                    expanded_nodes += 1
            if depth < maximum_depth:
                next_frontier = [child for node in frontier for child in node.children]
                if not next_frontier:
                    break
                levels.append(next_frontier)

        for frontier in reversed(levels[:-1]):
            for node in frontier:
                if node.children:
                    node.backed_value = max(-child.backed_value for child in node.children)

        if not root.children:
            return SearchResult(None, root.backed_value, expanded_nodes, model_evaluations,
                                len(levels) - 1, (time.perf_counter() - started) * 1000, ())
        maximum_prior = max(child.prior for child in root.children)
        ranked_root = []
        for child in root.children:
            minimax = -child.backed_value
            prior_adjustment = self.root_prior_weight * math.log(max(child.prior, 1e-30) / maximum_prior)
            ranked_root.append((child, minimax + prior_adjustment, minimax))
        ranked_root.sort(key=lambda item: (item[1], item[0].prior), reverse=True)
        policy = tuple((move_name(child.move), combined, minimax, child.prior)
                       for child, combined, minimax in ranked_root)
        return SearchResult(policy[0][0], root.backed_value, expanded_nodes, model_evaluations,
                            len(levels) - 1, (time.perf_counter() - started) * 1000, policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fen")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--internal-width", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--root-prior-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    search = SelectiveMinimaxSearch(
        TorchChoiceEvaluator(str(Path(args.model)), args.device), args.internal_width,
        args.batch_size, args.root_prior_weight)
    result = search.run(parse_fen(args.fen) if args.fen else parse_fen(), args.depth)
    print({
        "move": result.move, "value": result.value, "depth": result.maximum_depth,
        "expandedNodes": result.expanded_nodes, "modelEvaluations": result.model_evaluations,
        "elapsedMs": result.elapsed_ms, "policy": result.policy,
    })


if __name__ == "__main__":
    main()
