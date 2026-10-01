"""Self-contained Xiangqi rules used by the Python Jev search backend."""

from __future__ import annotations

from dataclasses import dataclass

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
FILES = "abcdefghi"


@dataclass(frozen=True)
class Position:
    board: tuple[str, ...]
    side: str
    halfmove: int = 0
    fullmove: int = 1


@dataclass(frozen=True)
class Move:
    source: int
    target: int
    piece: str = "."
    captured: str = "."


def inside(x: int, y: int) -> bool:
    return 0 <= x < 9 and 0 <= y < 10


def square(x: int, y: int) -> int:
    return y * 9 + x


def side_of(piece: str) -> str | None:
    if piece == ".":
        return None
    return "red" if piece.isupper() else "black"


def opponent(side: str) -> str:
    return "black" if side == "red" else "red"


def palace(x: int, y: int, side: str) -> bool:
    return 3 <= x <= 5 and (7 <= y <= 9 if side == "red" else 0 <= y <= 2)


def parse_fen(fen: str = START_FEN) -> Position:
    tokens = fen.strip().split()
    if not tokens:
        raise ValueError("invalid Xiangqi FEN")
    ranks = tokens[0].split("/")
    turn = tokens[1] if len(tokens) > 1 else "w"
    if len(ranks) != 10 or turn not in ("w", "r", "b"):
        raise ValueError("invalid Xiangqi FEN")
    board: list[str] = []
    aliases = {"h": "n", "H": "N", "e": "b", "E": "B"}
    for rank in ranks:
        expanded: list[str] = []
        for token in rank:
            if token.isdigit() and token != "0":
                expanded.extend("." for _ in range(int(token)))
            elif token in "rnbakcpheRNBAKCPHE":
                expanded.append(aliases.get(token, token))
            else:
                raise ValueError("invalid Xiangqi FEN piece")
        if len(expanded) != 9:
            raise ValueError("invalid Xiangqi FEN rank")
        board.extend(expanded)
    if board.count("K") != 1 or board.count("k") != 1:
        raise ValueError("invalid Xiangqi FEN kings")
    halfmove = int(tokens[4]) if len(tokens) >= 5 else 0
    fullmove = int(tokens[5]) if len(tokens) >= 6 else 1
    return Position(tuple(board), "black" if turn == "b" else "red",
                    max(0, halfmove), max(1, fullmove))


def to_fen(position: Position) -> str:
    ranks = []
    for y in range(10):
        text, empty = "", 0
        for x in range(9):
            piece = position.board[square(x, y)]
            if piece == ".":
                empty += 1
            else:
                if empty:
                    text += str(empty)
                    empty = 0
                text += piece
        if empty:
            text += str(empty)
        ranks.append(text)
    turn = "w" if position.side == "red" else "b"
    return f"{'/'.join(ranks)} {turn} - - {position.halfmove} {position.fullmove}"


def square_name(index: int) -> str:
    return FILES[index % 9] + str(9 - index // 9)


def square_index(name: str) -> int:
    if len(name) != 2 or name[0] not in FILES or name[1] not in "0123456789":
        return -1
    return square(FILES.index(name[0]), 9 - int(name[1]))


def move_name(move: Move) -> str:
    return square_name(move.source) + square_name(move.target)


def pseudo_moves(position: Position, side: str | None = None,
                 captures_only: bool = False) -> list[Move]:
    board = position.board
    side = side or position.side
    moves: list[Move] = []

    def add(source: int, x: int, y: int) -> None:
        if not inside(x, y):
            return
        target = square(x, y)
        captured = board[target]
        if side_of(captured) == side or (captures_only and captured == "."):
            return
        moves.append(Move(source, target, board[source], captured))

    for source, piece in enumerate(board):
        if side_of(piece) != side:
            continue
        kind, x, y = piece.lower(), source % 9, source // 9
        if kind == "k":
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if palace(x + dx, y + dy, side):
                    add(source, x + dx, y + dy)
            for direction in (-1, 1):
                yy = y + direction
                while inside(x, yy):
                    target = board[square(x, yy)]
                    if target != ".":
                        if target.lower() == "k" and side_of(target) != side:
                            add(source, x, yy)
                        break
                    yy += direction
        elif kind == "a":
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                if palace(x + dx, y + dy, side):
                    add(source, x + dx, y + dy)
        elif kind == "b":
            for dx, dy in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                target_y = y + dy
                if (inside(x + dx, target_y)
                        and (target_y >= 5 if side == "red" else target_y <= 4)
                        and board[square(x + dx // 2, y + dy // 2)] == "."):
                    add(source, x + dx, target_y)
        elif kind == "n":
            steps = ((1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1),
                     (-1, -2, 0, -1), (2, 1, 1, 0), (2, -1, 1, 0),
                     (-2, 1, -1, 0), (-2, -1, -1, 0))
            for dx, dy, leg_x, leg_y in steps:
                if (inside(x + dx, y + dy)
                        and board[square(x + leg_x, y + leg_y)] == "."):
                    add(source, x + dx, y + dy)
        elif kind in ("r", "c"):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                screened = False
                xx, yy = x + dx, y + dy
                while inside(xx, yy):
                    target = board[square(xx, yy)]
                    if kind == "r":
                        if target == ".":
                            if not captures_only:
                                add(source, xx, yy)
                        else:
                            if side_of(target) != side:
                                add(source, xx, yy)
                            break
                    elif not screened:
                        if target == ".":
                            if not captures_only:
                                add(source, xx, yy)
                        else:
                            screened = True
                    elif target != ".":
                        if side_of(target) != side:
                            add(source, xx, yy)
                        break
                    xx, yy = xx + dx, yy + dy
        elif kind == "p":
            add(source, x, y + (-1 if side == "red" else 1))
            if y <= 4 if side == "red" else y >= 5:
                add(source, x - 1, y)
                add(source, x + 1, y)
    return moves


def make_move(position: Position, move: Move) -> Position:
    board = list(position.board)
    piece, captured = board[move.source], board[move.target]
    board[move.target], board[move.source] = piece, "."
    halfmove = 0 if captured != "." or piece.lower() == "p" else position.halfmove + 1
    fullmove = position.fullmove + int(position.side == "black")
    return Position(tuple(board), opponent(position.side), halfmove, fullmove)


def is_in_check(position: Position, side: str | None = None) -> bool:
    board = position.board
    side = side or position.side
    king_piece = "K" if side == "red" else "k"
    try:
        king = board.index(king_piece)
    except ValueError:
        return True
    enemy = opponent(side)
    upper = enemy == "red"
    rook, cannon = (("R", "C") if upper else ("r", "c"))
    general, pawn = (("K", "P") if upper else ("k", "p"))
    horse, advisor, elephant = (("N", "A", "B") if upper else ("n", "a", "b"))
    x, y = king % 9, king // 9

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        screened, distance = False, 0
        xx, yy = x + dx, y + dy
        while inside(xx, yy):
            distance += 1
            attacker = board[square(xx, yy)]
            if attacker == ".":
                xx, yy = xx + dx, yy + dy
                continue
            if not screened:
                if attacker == rook or (attacker == general and
                                        (dx == 0 or (distance == 1 and palace(x, y, enemy)))):
                    return True
                screened = True
            else:
                if attacker == cannon:
                    return True
                break
            xx, yy = xx + dx, yy + dy

    pawn_y = y + (1 if enemy == "red" else -1)
    if inside(x, pawn_y) and board[square(x, pawn_y)] == pawn:
        return True
    if y <= 4 if enemy == "red" else y >= 5:
        if inside(x - 1, y) and board[square(x - 1, y)] == pawn:
            return True
        if inside(x + 1, y) and board[square(x + 1, y)] == pawn:
            return True

    horse_steps = ((1, 2), (-1, 2), (1, -2), (-1, -2),
                   (2, 1), (2, -1), (-2, 1), (-2, -1))
    for dx, dy in horse_steps:
        source_x, source_y = x - dx, y - dy
        if not inside(source_x, source_y) or board[square(source_x, source_y)] != horse:
            continue
        leg_x = source_x + (0 if abs(dx) == 1 else (1 if dx > 0 else -1))
        leg_y = source_y + (0 if abs(dy) == 1 else (1 if dy > 0 else -1))
        if board[square(leg_x, leg_y)] == ".":
            return True

    if palace(x, y, enemy):
        for dx in (-1, 1):
            for dy in (-1, 1):
                if inside(x + dx, y + dy) and board[square(x + dx, y + dy)] == advisor:
                    return True
    if y >= 5 if enemy == "red" else y <= 4:
        for dx in (-2, 2):
            for dy in (-2, 2):
                if (inside(x + dx, y + dy)
                        and board[square(x + dx, y + dy)] == elephant
                        and board[square(x + dx // 2, y + dy // 2)] == "."):
                    return True
    return False


def legal_moves(position: Position) -> list[Move]:
    return [move for move in pseudo_moves(position)
            if not is_in_check(make_move(position, move), position.side)]


def play_move(position: Position, notation: str) -> Position:
    move = next((candidate for candidate in legal_moves(position)
                 if move_name(candidate) == notation), None)
    if move is None:
        raise ValueError("illegal move")
    return make_move(position, move)


def position_key(position: Position) -> str:
    return " ".join(to_fen(position).split()[:2])


def perft(position: Position, depth: int) -> int:
    if depth <= 0:
        return 1
    return sum(perft(make_move(position, move), depth - 1)
               for move in legal_moves(position))
