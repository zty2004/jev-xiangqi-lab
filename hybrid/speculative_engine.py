"""Fuse a fast Jev policy draft with a full Pikafish target search.

The draft proposes root moves. Pikafish first verifies those moves with a
restricted MultiPV search, retaining the resulting transposition-table work,
then performs an unrestricted search and may reject every draft proposal.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.choice_model import load_model, rank, select_device  # noqa: E402

MOVE = re.compile(r"^[a-i][0-9][a-i][0-9]$")
PERFT_MOVE = re.compile(r"^([a-i][0-9][a-i][0-9]):\s+1$")


@dataclass
class SearchResult:
    move: str | None
    ponder: str | None = None
    depth: int = 0
    selective_depth: int = 0
    score_type: str | None = None
    score: int | None = None
    nodes: int = 0
    pv: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass
class SpeculativeResult:
    move: str | None
    draft: tuple[tuple[str, float], ...]
    verification: SearchResult | None
    target: SearchResult
    disposition: str
    elapsed_ms: float


def _integer(line: str, field: str, default: int = 0) -> int:
    match = re.search(rf"(?:^|\s){re.escape(field)}\s+(-?\d+)(?:\s|$)", line)
    return int(match.group(1)) if match else default


def parse_info(line: str, previous: SearchResult | None = None) -> SearchResult:
    result = previous or SearchResult(None)
    score = re.search(r"\bscore\s+(cp|mate)\s+(-?\d+)", line)
    pv_match = re.search(r"\spv\s+(.+)$", line)
    pv = tuple(token for token in pv_match.group(1).split() if MOVE.match(token)) if pv_match else result.pv
    return SearchResult(
        move=result.move,
        ponder=result.ponder,
        depth=_integer(line, "depth", result.depth),
        selective_depth=_integer(line, "seldepth", result.selective_depth),
        score_type=score.group(1) if score else result.score_type,
        score=int(score.group(2)) if score else result.score,
        nodes=_integer(line, "nodes", result.nodes),
        pv=pv,
        elapsed_ms=result.elapsed_ms,
    )


class PikafishProcess:
    def __init__(self, executable: str, threads: int = 1, hash_mb: int = 256):
        self.executable = str(Path(executable).resolve())
        self.process = subprocess.Popen(
            [self.executable], cwd=str(Path(self.executable).parent), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until(lambda line: line == "uciok")
        self.set_option("Threads", threads)
        self.set_option("Hash", hash_mb)
        self.set_option("Move Overhead", 0)
        self._send("isready")
        self._read_until(lambda line: line == "readyok")

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Pikafish exited with {self.process.returncode}: {error[-2000:]}")
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, predicate) -> list[str]:
        assert self.process.stdout is not None
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                error = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"Pikafish closed its output: {error[-2000:]}")
            line = line.rstrip("\r\n")
            lines.append(line)
            if predicate(line):
                return lines

    def set_option(self, name: str, value) -> None:
        self._send(f"setoption name {name} value {value}")

    def clear_hash(self) -> None:
        self._send("setoption name Clear Hash")
        self._send("isready")
        self._read_until(lambda line: line == "readyok")

    def set_position(self, position_command: str) -> None:
        if not position_command.startswith("position "):
            raise ValueError("position command must start with 'position '")
        self._send(position_command)

    def fen(self) -> str:
        self._send("d")
        lines = self._read_until(lambda line: line.startswith("Checkers:"))
        fen = next((line[5:].strip() for line in lines if line.startswith("Fen: ")), None)
        if not fen:
            raise RuntimeError("Pikafish did not return a FEN")
        return fen

    def legal_moves(self) -> list[str]:
        self._send("go perft 1")
        lines = self._read_until(lambda line: line.startswith("Nodes searched:"))
        return [match.group(1) for line in lines if (match := PERFT_MOVE.match(line))]

    def search(self, movetime_ms: int, searchmoves: list[str] | None = None,
               multipv: int = 1) -> SearchResult:
        if movetime_ms < 1:
            raise ValueError("movetime must be positive")
        self.set_option("MultiPV", max(1, multipv))
        suffix = f" searchmoves {' '.join(searchmoves)}" if searchmoves else ""
        started = time.monotonic()
        self._send(f"go movetime {int(movetime_ms)}{suffix}")
        lines = self._read_until(lambda line: line.startswith("bestmove "))
        info = SearchResult(None)
        for line in lines:
            if line.startswith("info ") and " multipv 1 " in f" {line} ":
                info = parse_info(line, info)
        tokens = lines[-1].split()
        move = tokens[1] if len(tokens) > 1 and MOVE.match(tokens[1]) else None
        ponder = tokens[3] if len(tokens) > 3 and tokens[2] == "ponder" and MOVE.match(tokens[3]) else None
        return SearchResult(move, ponder, info.depth, info.selective_depth, info.score_type,
                            info.score, info.nodes, info.pv, (time.monotonic() - started) * 1000)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class JevDraftModel:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.device = select_device(device)
        self.model, self.temperature = load_model(checkpoint, self.device)

    def choices(self, fen: str, legal_moves: list[str], history: list[str] | None = None):
        return rank(self.model, fen, legal_moves, self.device, self.temperature, history)["choices"]


class SpeculativeEngine:
    def __init__(self, target: PikafishProcess, draft: JevDraftModel, top_k: int = 8,
                 verification_fraction: float = 0.15, safety_ms: int = 25):
        if top_k < 1 or not 0 < verification_fraction < 0.5:
            raise ValueError("invalid speculative search configuration")
        self.target = target
        self.draft = draft
        self.top_k = top_k
        self.verification_fraction = verification_fraction
        self.safety_ms = safety_ms

    def choose(self, position_command: str, movetime_ms: int,
               history: list[str] | None = None) -> SpeculativeResult:
        started = time.monotonic()
        deadline = started + movetime_ms / 1000
        self.target.set_position(position_command)
        fen = self.target.fen()
        legal = self.target.legal_moves()
        if not legal:
            empty = SearchResult(None)
            return SpeculativeResult(None, (), None, empty, "no-legal-move",
                                     (time.monotonic() - started) * 1000)
        choices = self.draft.choices(fen, legal, history)
        proposals = choices[: min(self.top_k, len(choices))]
        draft = tuple((item["move"], float(item["probability"])) for item in proposals)

        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000) - self.safety_ms)
        verification_ms = max(1, min(remaining_ms // 3,
                                     int(movetime_ms * self.verification_fraction)))
        verification = None
        if len(legal) > 1 and remaining_ms >= 3:
            verification = self.target.search(verification_ms, [move for move, _ in draft], len(draft))

        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000) - self.safety_ms)
        target = self.target.search(remaining_ms)
        if target.move == draft[0][0]:
            disposition = "accepted-top1"
        elif target.move in {move for move, _ in draft}:
            disposition = "corrected-within-draft"
        else:
            disposition = "rejected-fallback"
        return SpeculativeResult(target.move, draft, verification, target, disposition,
                                 (time.monotonic() - started) * 1000)


def run_uci(args) -> None:
    target = PikafishProcess(args.pikafish, args.threads, args.hash)
    engine = SpeculativeEngine(target, JevDraftModel(args.model, args.device), args.top_k,
                               args.verification_fraction)
    position = "position startpos"
    try:
        for raw in sys.stdin:
            command = raw.strip()
            if command == "uci":
                print("id name Jev-Pikafish Speculative", flush=True)
                print("id author Jev Xiangqi Lab", flush=True)
                print("uciok", flush=True)
            elif command == "isready":
                print("readyok", flush=True)
            elif command.startswith("position "):
                position = command
            elif command == "ucinewgame":
                target.clear_hash()
            elif command.startswith("go "):
                match = re.search(r"\bmovetime\s+(\d+)", command)
                if not match:
                    print("info string Jev-Pikafish currently requires go movetime", flush=True)
                    print("bestmove 0000", flush=True)
                    continue
                result = engine.choose(position, int(match.group(1)))
                print("info string " + str({
                    "speculation": result.disposition,
                    "draft": [move for move, _ in result.draft],
                    "elapsedMs": round(result.elapsed_ms, 1),
                }), flush=True)
                target_info = result.target
                score = f" score {target_info.score_type} {target_info.score}" if target_info.score_type else ""
                pv = f" pv {' '.join(target_info.pv)}" if target_info.pv else ""
                print(f"info depth {target_info.depth} seldepth {target_info.selective_depth}{score} "
                      f"nodes {target_info.nodes}{pv}", flush=True)
                print(f"bestmove {result.move or '0000'}", flush=True)
            elif command == "quit":
                break
    finally:
        target.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pikafish", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--verification-fraction", type=float, default=0.15)
    run_uci(parser.parse_args())


if __name__ == "__main__":
    main()
