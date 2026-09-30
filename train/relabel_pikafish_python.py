"""Re-label existing teacher positions with reproducible Pikafish MultiPV/PV data.

Each invocation owns one persistent Pikafish process and one deterministic
shard. Run several shards on distinct CPU cores, then combine them by
``sourceIndex``. This training path is Python plus native C++ Pikafish only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

MOVE = re.compile(r"^[a-i][0-9][a-i][0-9]$")


def sha256_file(filename: str) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pikafish_info(line: str):
    if not line.startswith("info ") or " multipv " not in line or " pv " not in line:
        return None
    fields = {}
    for name in ("depth", "seldepth", "nodes", "multipv"):
        match = re.search(rf"\b{name}\s+(\d+)", line)
        fields[name] = int(match.group(1)) if match else 0
    score = re.search(r"\bscore\s+(cp|mate)\s+(-?\d+)", line)
    pv = [token for token in line.split(" pv ", 1)[1].split() if MOVE.match(token)]
    if not fields["depth"] or not fields["multipv"] or not score or not pv:
        return None
    return {"depth": fields["depth"], "selectiveDepth": fields["seldepth"],
            "nodes": fields["nodes"], "rank": fields["multipv"], "move": pv[0],
            "pv": pv, "score": int(score.group(2)), "scoreType": score.group(1)}


def select_multipv(lines: list[str], best: str) -> dict:
    by_depth = {}
    searched_nodes = selective_depth = 0
    for line in lines:
        info = parse_pikafish_info(line)
        if not info:
            continue
        searched_nodes = max(searched_nodes, info["nodes"])
        selective_depth = max(selective_depth, info["selectiveDepth"])
        by_depth.setdefault(info["depth"], {})[info["rank"]] = info
    if not by_depth:
        raise ValueError("Pikafish returned no complete MultiPV information")

    def unique_count(depth):
        return len({item["move"] for item in by_depth[depth].values()})

    selected_depth = max(by_depth, key=lambda depth: (unique_count(depth), depth))
    candidates, seen = [], set()
    for rank, item in sorted(by_depth[selected_depth].items()):
        if item["move"] in seen:
            continue
        seen.add(item["move"])
        candidates.append({key: value for key, value in item.items()
                           if key not in ("depth", "selectiveDepth", "nodes")})
    return {"best": best, "depth": selected_depth, "selectiveDepth": selective_depth,
            "nodes": searched_nodes, "candidates": candidates}


class PikafishTeacher:
    def __init__(self, executable: str, hash_mb: int = 64, multipv: int = 8):
        self.executable = str(Path(executable).resolve())
        self.multipv = multipv
        self.process = subprocess.Popen(
            [self.executable], cwd=str(Path(self.executable).parent), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.read_until(lambda line: line == "uciok")
        self.send("setoption name Threads value 1")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name MultiPV value {multipv}")
        self.send("isready")
        self.read_until(lambda line: line == "readyok")

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, predicate) -> list[str]:
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

    def analyse(self, fen: str, nodes: int) -> dict:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}")
        lines = self.read_until(lambda line: line.startswith("bestmove "))
        best = lines[-1].split()[1]
        if not MOVE.match(best):
            raise ValueError(f"Pikafish returned invalid best move: {best}")
        return select_multipv(lines, best)

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def load_source(filename: str, limit: int | None) -> tuple[dict, list[dict]]:
    with open(filename, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    metadata = rows[0] if rows and rows[0].get("kind") == "meta" else {}
    positions = [row for row in rows if row.get("kind") == "position"]
    if limit is not None:
        positions = positions[:limit]
    if not positions:
        raise ValueError("source contains no positions")
    return metadata, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pikafish", required=True)
    parser.add_argument("--nodes", type=int, default=250000)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.nodes < 1 or args.multipv < 1 or args.hash < 1 or args.shard_count < 1 or
            not 0 <= args.shard_index < args.shard_count or
            (args.limit is not None and args.limit < 1)):
        raise ValueError("invalid relabel configuration")

    source_meta, positions = load_source(args.input, args.limit)
    tasks = [(index, row) for index, row in enumerate(positions)
             if index % args.shard_count == args.shard_index]
    metadata = {
        "kind": "meta", "schema": "jev-pikafish-pv-relabel-v1",
        "source": str(Path(args.input)), "sourceSha256": sha256_file(args.input),
        "sourceMetadata": source_meta, "teacherBinarySha256": sha256_file(args.pikafish),
        "nodes": args.nodes, "multipv": args.multipv, "hashMb": args.hash,
        "shardIndex": args.shard_index, "shardCount": args.shard_count,
        "sourcePositions": len(positions), "shardPositions": len(tasks),
    }
    output = Path(args.output)
    completed = 0
    if output.exists():
        if not args.resume:
            raise ValueError(f"output exists: {output}; pass --resume or choose another path")
        with output.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        if not existing or existing[0] != metadata:
            raise ValueError("resume metadata mismatch")
        completed = len(existing) - 1
        for actual, (source_index, _) in zip(existing[1:], tasks):
            if actual.get("sourceIndex") != source_index:
                raise ValueError("resume source index mismatch")
    else:
        output.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")

    teacher = PikafishTeacher(args.pikafish, args.hash, args.multipv)
    try:
        with output.open("a", encoding="utf-8") as handle:
            for offset, (source_index, row) in enumerate(tasks[completed:], completed + 1):
                analysis = teacher.analyse(row["fen"], args.nodes)
                legal = set(row["legal"])
                if analysis["best"] not in legal:
                    raise ValueError(f"illegal best move at source index {source_index}")
                candidates = [item for item in analysis["candidates"] if item["move"] in legal]
                labelled = {**row, "sourceIndex": source_index, "best": analysis["best"],
                            "candidates": candidates, "depth": analysis["depth"],
                            "selectiveDepth": analysis["selectiveDepth"], "nodes": analysis["nodes"]}
                handle.write(json.dumps(labelled, ensure_ascii=False) + "\n")
                handle.flush()
                if offset % 100 == 0 or offset == len(tasks):
                    print(f"shard={args.shard_index} labelled={offset}/{len(tasks)}", flush=True)
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
