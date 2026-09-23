export const START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1';

const files = 'abcdefghi';
const inside = (x, y) => x >= 0 && x < 9 && y >= 0 && y < 10;
const at = (x, y) => y * 9 + x;
const red = piece => piece !== '.' && piece === piece.toUpperCase();
const sideOf = piece => piece === '.' ? null : red(piece) ? 'red' : 'black';
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
  const king = position.board.indexOf(side === 'red' ? 'K' : 'k');
  if (king < 0) return true;
  return pseudoMoves(position, opponent(side), true).some(move => move.to === king);
}

export function legalMoves(position) {
  return pseudoMoves(position).filter(move => !isInCheck(makeMove(position, move), position.side));
}

export function playMove(position, notation) {
  const move = legalMoves(position).find(candidate => moveName(candidate) === notation);
  if (!move) throw new Error('Illegal move');
  return makeMove(position, move);
}

export function gameResult(position, history = []) {
  const key = toFen(position).split(' ').slice(0, 2).join(' ');
  if (history.filter(item => item === key).length >= 3) return { winner: null, reason: '三次重复局面' };
  if (position.halfmove >= 120) return { winner: null, reason: '六十回合未吃子或走兵' };
  if (!legalMoves(position).length) return { winner: opponent(position.side), reason: isInCheck(position) ? '将死' : '困毙' };
  return null;
}

export function positionKey(position) { return toFen(position).split(' ').slice(0, 2).join(' '); }
