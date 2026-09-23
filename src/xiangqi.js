export const START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1';

const files = 'abcdefghi';
const inside = (x, y) => x >= 0 && x < 9 && y >= 0 && y < 10;
const at = (x, y) => y * 9 + x;
const sideOf = piece => piece === '.' ? null : piece.charCodeAt(0) <= 90 ? 'red' : 'black';
const opponent = side => side === 'red' ? 'black' : 'red';
const palace = (x, y, side) => x >= 3 && x <= 5 && (side === 'red' ? y >= 7 && y <= 9 : y >= 0 && y <= 2);

export function parseFen(fen = START_FEN) {
  const [rows, turn = 'w', , , halfmove = '0', fullmove = '1'] = fen.trim().split(/\s+/);
  const ranks = rows?.split('/');
  if (ranks?.length !== 10 || !['w', 'r', 'b'].includes(turn)) throw new Error('Invalid Xiangqi FEN');
  const board = [];
  for (const rank of ranks) {
    const before = board.length;
    for (const token of rank) {
      if (/[1-9]/.test(token)) board.push(...Array(Number(token)).fill('.'));
      else if (/[rnbakcpheRNBAKCPHE]/.test(token)) board.push(({ h: 'n', H: 'N', e: 'b', E: 'B' })[token] || token);
      else throw new Error('Invalid Xiangqi FEN piece');
    }
    if (board.length - before !== 9) throw new Error('Invalid Xiangqi FEN rank');
  }
  if (board.length !== 90 || board.filter(p => p === 'K').length !== 1 || board.filter(p => p === 'k').length !== 1) throw new Error('Invalid Xiangqi FEN kings');
  return { board, side: turn === 'b' ? 'black' : 'red', halfmove: Number(halfmove) || 0, fullmove: Number(fullmove) || 1 };
}

export function toFen(position) {
  const ranks = [];
  for (let y = 0; y < 10; y++) {
    let rank = '', empty = 0;
    for (let x = 0; x < 9; x++) {
      const piece = position.board[at(x, y)];
      if (piece === '.') empty++;
      else { if (empty) rank += empty; empty = 0; rank += piece; }
    }
    if (empty) rank += empty;
    ranks.push(rank);
  }
  return `${ranks.join('/')} ${position.side === 'red' ? 'w' : 'b'} - - ${position.halfmove} ${position.fullmove}`;
}

export function squareName(index) { return files[index % 9] + (9 - Math.floor(index / 9)); }
export function squareIndex(name) {
  if (!/^[a-i][0-9]$/.test(name)) return -1;
  return at(files.indexOf(name[0]), 9 - Number(name[1]));
}
export const moveName = move => squareName(move.from) + squareName(move.to);

export function pseudoMoves(position, side = position.side, capturesOnly = false) {
  const { board } = position;
  const moves = [];
  const add = (from, x, y) => {
    if (!inside(x, y)) return;
    const to = at(x, y), target = board[to];
    if (sideOf(target) === side || (capturesOnly && target === '.')) return;
    moves.push({ from, to, piece: board[from], captured: target });
  };
  for (let from = 0; from < 90; from++) {
    const piece = board[from];
    if (sideOf(piece) !== side) continue;
    const type = piece.toLowerCase(), x = from % 9, y = Math.floor(from / 9);
    if (type === 'k') {
      for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) if (palace(x + dx, y + dy, side)) add(from, x + dx, y + dy);
      for (const dir of [-1, 1]) {
        for (let yy = y + dir; inside(x, yy); yy += dir) {
          if (board[at(x, yy)] === '.') continue;
          if (board[at(x, yy)].toLowerCase() === 'k' && sideOf(board[at(x, yy)]) !== side) add(from, x, yy);
          break;
        }
      }
    } else if (type === 'a') {
      for (const [dx, dy] of [[1,1],[1,-1],[-1,1],[-1,-1]]) if (palace(x + dx, y + dy, side)) add(from, x + dx, y + dy);
    } else if (type === 'b') {
      for (const [dx, dy] of [[2,2],[2,-2],[-2,2],[-2,-2]]) {
        const yy = y + dy;
        if (inside(x + dx, yy) && (side === 'red' ? yy >= 5 : yy <= 4) && board[at(x + dx / 2, y + dy / 2)] === '.') add(from, x + dx, yy);
      }
    } else if (type === 'n') {
      for (const [dx, dy, lx, ly] of [[1,2,0,1],[-1,2,0,1],[1,-2,0,-1],[-1,-2,0,-1],[2,1,1,0],[2,-1,1,0],[-2,1,-1,0],[-2,-1,-1,0]]) {
        if (inside(x + dx, y + dy) && board[at(x + lx, y + ly)] === '.') add(from, x + dx, y + dy);
      }
    } else if (type === 'r' || type === 'c') {
      for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        let screened = false;
        for (let xx = x + dx, yy = y + dy; inside(xx, yy); xx += dx, yy += dy) {
          const target = board[at(xx, yy)];
          if (type === 'r') {
            if (target === '.') { if (!capturesOnly) add(from, xx, yy); }
            else { if (sideOf(target) !== side) add(from, xx, yy); break; }
          } else if (!screened) {
            if (target === '.') { if (!capturesOnly) add(from, xx, yy); }
            else screened = true;
          } else if (target !== '.') {
            if (sideOf(target) !== side) add(from, xx, yy);
            break;
          }
        }
      }
    } else if (type === 'p') {
      add(from, x, y + (side === 'red' ? -1 : 1));
      const crossed = side === 'red' ? y <= 4 : y >= 5;
      if (crossed) { add(from, x - 1, y); add(from, x + 1, y); }
    }
  }
  return moves;
}

export function makeMove(position, move) {
  const board = position.board.slice();
  const captured = board[move.to], piece = board[move.from];
  board[move.to] = piece;
  board[move.from] = '.';
  return { board, side: opponent(position.side), halfmove: captured !== '.' || piece.toLowerCase() === 'p' ? 0 : position.halfmove + 1, fullmove: position.fullmove + (position.side === 'black' ? 1 : 0) };
}

export function isInCheck(position, side = position.side) {
  const { board } = position;
  const king = board.indexOf(side === 'red' ? 'K' : 'k');
  if (king < 0) return true;
  const enemy = opponent(side), upper = enemy === 'red';
  const rook = upper ? 'R' : 'r', cannon = upper ? 'C' : 'c';
  const general = upper ? 'K' : 'k', pawn = upper ? 'P' : 'p';
  const horse = upper ? 'N' : 'n', advisor = upper ? 'A' : 'a', elephant = upper ? 'B' : 'b';
  const x = king % 9, y = Math.floor(king / 9);

  // Look outward from the king. This avoids generating every enemy capture
  // during each legality check and search node.
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    let screened = false, distance = 0;
    for (let xx = x + dx, yy = y + dy; inside(xx, yy); xx += dx, yy += dy) {
      distance++;
      const attacker = board[at(xx, yy)];
      if (attacker === '.') continue;
      if (!screened) {
        if (attacker === rook || (attacker === general &&
            (dx === 0 || (distance === 1 && palace(x, y, enemy))))) return true;
        screened = true;
      } else {
        if (attacker === cannon) return true;
        break;
      }
    }
  }

  const pawnY = y + (enemy === 'red' ? 1 : -1);
  if (inside(x, pawnY) && board[at(x, pawnY)] === pawn) return true;
  if (enemy === 'red' ? y <= 4 : y >= 5) {
    if (inside(x - 1, y) && board[at(x - 1, y)] === pawn) return true;
    if (inside(x + 1, y) && board[at(x + 1, y)] === pawn) return true;
  }

  for (const [dx, dy] of [[1, 2], [-1, 2], [1, -2], [-1, -2],
                          [2, 1], [2, -1], [-2, 1], [-2, -1]]) {
    const sx = x - dx, sy = y - dy;
    if (!inside(sx, sy) || board[at(sx, sy)] !== horse) continue;
    const legX = sx + (Math.abs(dx) === 2 ? Math.sign(dx) : 0);
    const legY = sy + (Math.abs(dy) === 2 ? Math.sign(dy) : 0);
    if (board[at(legX, legY)] === '.') return true;
  }

  if (palace(x, y, enemy)) {
    for (const dx of [-1, 1]) for (const dy of [-1, 1])
      if (inside(x + dx, y + dy) && board[at(x + dx, y + dy)] === advisor) return true;
  }
  if (enemy === 'red' ? y >= 5 : y <= 4) {
    for (const dx of [-2, 2]) for (const dy of [-2, 2])
      if (inside(x + dx, y + dy) && board[at(x + dx, y + dy)] === elephant &&
          board[at(x + dx / 2, y + dy / 2)] === '.') return true;
  }
  return false;
}

export function legalMoves(position) {
  return pseudoMoves(position).filter(move => !isInCheck(makeMove(position, move), position.side));
}

export function playMove(position, notation) {
  const move = legalMoves(position).find(candidate => moveName(candidate) === notation);
  if (!move) throw new Error('Illegal move');
  return makeMove(position, move);
}

export function repetitionResult(history = []) {
  const key = history.at(-1);
  if (!key) return null;
  const occurrences = [];
  for (let index = 0; index < history.length; index++) if (history[index] === key) occurrences.push(index);
  if (occurrences.length < 3) return null;

  // A repeated position alone is not enough to call a draw in Xiangqi. Inspect
  // the moves between the last three occurrences and identify a side that
  // checked on every one of its turns throughout both repetitions.
  const start = occurrences.at(-3);
  const moves = { red: 0, black: 0 }, checks = { red: 0, black: 0 };
  for (let index = start + 1; index < history.length; index++) {
    const position = parseFen(`${history[index]} - - 0 1`);
    const mover = position.side === 'red' ? 'black' : 'red';
    moves[mover]++;
    if (isInCheck(position)) checks[mover]++;
  }
  const redPerpetual = moves.red > 0 && checks.red === moves.red;
  const blackPerpetual = moves.black > 0 && checks.black === moves.black;
  if (redPerpetual && !blackPerpetual) return { winner: 'black', reason: '红方长将判负' };
  if (blackPerpetual && !redPerpetual) return { winner: 'red', reason: '黑方长将判负' };
  return { winner: null, reason: '三次重复局面' };
}

export function gameResult(position, history = []) {
  const key = positionKey(position);
  if (history.at(-1) === key) {
    const repetition = repetitionResult(history);
    if (repetition) return repetition;
  }
  if (position.halfmove >= 120) return { winner: null, reason: '六十回合未吃子或走兵' };
  if (!legalMoves(position).length) return { winner: opponent(position.side), reason: isInCheck(position) ? '将死' : '困毙' };
  return null;
}

export function positionKey(position) { return toFen(position).split(' ').slice(0, 2).join(' '); }
