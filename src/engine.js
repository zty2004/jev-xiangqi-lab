import { isInCheck, legalMoves, makeMove, moveName, positionKey, pseudoMoves } from './xiangqi.js';

const VALUES = { k: 20000, r: 1000, c: 470, n: 430, b: 220, a: 220, p: 120 };
const MATE = 30000;
const INF = 32000;

export function evaluate(position) {
  let score = 0;
  for (let i = 0; i < 90; i++) {
    const piece = position.board[i];
    if (piece === '.') continue;
    const side = piece === piece.toUpperCase() ? 'red' : 'black';
    const y = Math.floor(i / 9), x = i % 9, advance = side === 'red' ? 9 - y : y;
    let value = VALUES[piece.toLowerCase()];
    if (piece.toLowerCase() === 'p') value += advance * 9 + (advance >= 5 ? 60 + (4 - Math.abs(x - 4)) * 6 : 0);
    if (piece.toLowerCase() === 'n') value += (4 - Math.abs(x - 4)) * 8 + (4.5 - Math.abs(y - 4.5)) * 5;
    if (piece.toLowerCase() === 'r' || piece.toLowerCase() === 'c') value += (4 - Math.abs(x - 4)) * 5;
    score += side === 'red' ? value : -value;
  }
  return position.side === 'red' ? score : -score;
}

class Stopped extends Error {}

export function chooseMove(position, options = {}) {
  const timeMs = Math.max(10, Number(options.timeMs) || 1000);
  const maxDepth = Math.max(1, Math.min(20, Number(options.maxDepth) || 12));
  const start = performance.now(), deadline = start + timeMs;
  const table = new Map(), history = new Map(), killers = Array.from({ length: 64 }, () => []);
  const rootMoves = legalMoves(position);
  const priors = options.priors || null;
  const fullRootScores = Boolean(options.fullRootScores);
  if (!rootMoves.length) return { move: null, depth: 0, score: -MATE, nodes: 0, timeMs: 0, pv: [] };
  const multiPv = fullRootScores ? rootMoves.length : Math.max(1, Math.min(rootMoves.length, Number(options.multiPv) || 1));
  let nodes = 0, completed = { move: rootMoves[0], depth: 0, score: evaluate(position), pv: [moveName(rootMoves[0])], candidates: rootMoves.map(move => ({ move: moveName(move), score: -evaluate(makeMove(position, move)) })).sort((a, b) => b.score - a.score).slice(0, 8) };
  const repetition = options.history ? [...options.history] : [];

  function checkTime() {
    if ((nodes & 1023) === 0 && performance.now() >= deadline) throw new Stopped();
  }
  function ordered(moves, ttMove, ply) {
    return moves.sort((a, b) => priority(b, ttMove, ply) - priority(a, ttMove, ply));
  }
  function priority(move, ttMove, ply) {
    const key = moveName(move);
    if (key === ttMove) return 2_000_000;
    if (ply === 0 && priors) return Math.max(0, Math.min(1, priors.get(key) || 0)) * 1_000_000 +
      (move.captured !== '.' ? 250_000 + VALUES[move.captured.toLowerCase()] * 10 : 0);
    if (move.captured !== '.') return 1_000_000 + VALUES[move.captured.toLowerCase()] * 10 - VALUES[move.piece.toLowerCase()];
    if (killers[ply]?.includes(key)) return 500_000;
    return history.get(key) || 0;
  }
  function quiescence(pos, alpha, beta, ply) {
    nodes++; checkTime();
    if (ply >= 24) return evaluate(pos);
    const checked = isInCheck(pos);
    const stand = evaluate(pos);
    if (!checked) {
      if (stand >= beta) return beta;
      if (stand > alpha) alpha = stand;
    }
    const source = checked ? legalMoves(pos) : pseudoMoves(pos, pos.side, true).filter(m => !isInCheck(makeMove(pos, m), pos.side));
    if (checked && !source.length) return -MATE + ply;
    for (const move of ordered(source, null, ply)) {
      const score = -quiescence(makeMove(pos, move), -beta, -alpha, ply + 1);
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }
  function negamax(pos, depth, alpha, beta, ply) {
    nodes++; checkTime();
    const key = positionKey(pos);
    if (repetition.filter(item => item === key).length >= 2 || pos.halfmove >= 120) return 0;
    if (depth <= 0) return quiescence(pos, alpha, beta, ply);
    const entry = table.get(key), originalAlpha = alpha;
    if (entry && entry.depth >= depth) {
      if (entry.flag === 'exact') return entry.score;
      if (entry.flag === 'lower') alpha = Math.max(alpha, entry.score);
      else beta = Math.min(beta, entry.score);
      if (alpha >= beta) return entry.score;
    }
    const moves = legalMoves(pos);
    if (!moves.length) return -MATE + ply;
    let best = -INF, bestMove = null;
    repetition.push(key);
    try {
      for (const move of ordered(moves, entry?.move, ply)) {
        const score = -negamax(makeMove(pos, move), depth - 1, -beta, -alpha, ply + 1);
        if (score > best) { best = score; bestMove = moveName(move); }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
          if (move.captured === '.') {
            const k = killers[ply] || (killers[ply] = []);
            if (!k.includes(bestMove)) k.unshift(bestMove);
            k.length = Math.min(2, k.length);
            history.set(bestMove, (history.get(bestMove) || 0) + depth * depth);
          }
          break;
        }
      }
    } finally { repetition.pop(); }
    if (table.size > 200_000) table.clear();
    table.set(key, { depth, score: best, move: bestMove, flag: best <= originalAlpha ? 'upper' : best >= beta ? 'lower' : 'exact' });
    return best;
  }

  function continuation(move, depth) {
    const line = [moveName(move)];
    let next = makeMove(position, move);
    for (let remaining = depth - 1; remaining > 0; remaining--) {
      const entry = table.get(positionKey(next));
      if (!entry?.move || entry.depth < remaining || entry.flag !== 'exact') break;
      const reply = legalMoves(next).find(item => moveName(item) === entry.move);
      if (!reply) break;
      line.push(entry.move);
      next = makeMove(next, reply);
    }
    return line;
  }

  for (let depth = 1; depth <= maxDepth; depth++) {
    try {
      let scores = [];
      const rootKey = positionKey(position), ttMove = table.get(rootKey)?.move || (completed.depth ? moveName(completed.move) : null);
      repetition.push(rootKey);
      try {
        for (const move of ordered(rootMoves, ttMove, 0)) {
          const next = makeMove(position, move);
          if (scores.length >= multiPv) {
            const threshold = scores[multiPv - 1].score;
            const probe = -negamax(next, depth - 1, -INF, -threshold, 1);
            if (probe <= threshold) continue;
          }
          const score = -negamax(next, depth - 1, -INF, INF, 1);
          scores.push({ move: moveName(move), score, pv: continuation(move, depth) });
          scores.sort((a, b) => b.score - a.score);
          scores.length = Math.min(scores.length, multiPv);
        }
      } finally { repetition.pop(); }
      const best = scores[0];
      completed = { move: rootMoves.find(move => moveName(move) === best.move), depth,
        score: best.score, pv: best.pv, candidates: scores.slice(0, 8) };
      table.set(rootKey, { depth, score: best.score, move: best.move, flag: 'exact' });
    } catch (error) {
      if (!(error instanceof Stopped)) throw error;
      break;
    }
    if (performance.now() >= deadline) break;
  }
  return { ...completed, nodes, timeMs: Math.round(performance.now() - start) };
}
