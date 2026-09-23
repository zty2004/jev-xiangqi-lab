import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chooseMove } from '../src/engine.js';
import { START_FEN, gameResult, isInCheck, legalMoves, makeMove, moveName, parseFen, playMove, positionKey, pseudoMoves, squareIndex, toFen } from '../src/xiangqi.js';

function perft(position, depth) {
  if (depth === 0) return 1;
  return legalMoves(position).reduce((count, move) => count + perft(makeMove(position, move), depth - 1), 0);
}

test('opening and tactical legal-move trees match independent perft results', () => {
  const cases = [
    [START_FEN, [44, 1920, 79666]],
    ['r1ea1a3/4kh3/2h1e4/pHp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2EAKAE2 w - - 0 1', [38, 1128, 43929]],
    ['1ceak4/9/h2a5/2p1p3p/5cp2/2h2H3/6PCP/3AE4/2C6/3A1K1H1 w - - 0 1', [7, 281, 8620]],
  ];
  for (const [fen, expected] of cases) {
    const position = parseFen(fen);
    expected.forEach((count, index) => assert.equal(perft(position, index + 1), count, `${fen} depth ${index + 1}`));
  }
});

test('FEN round trip and legal move validation', () => {
  const position = parseFen();
  assert.equal(toFen(position), START_FEN);
  assert.throws(() => playMove(position, 'a0a9'), /Illegal move/);
  const next = playMove(position, 'b2e2');
  assert.equal(next.side, 'black');
  assert.ok(legalMoves(next).length > 0);
  assert.equal(toFen(parseFen(toFen(next))), toFen(next));
});

test('facing generals count as check and illegal exposure is filtered', () => {
  const position = parseFen('4k4/9/9/9/9/4R4/9/9/9/4K4 w - - 0 1');
  assert.equal(isInCheck(position), false);
  assert.ok(!legalMoves(position).some(move => moveName(move) === 'e4d4'));
  assert.ok(!legalMoves(position).some(move => moveName(move) === 'e4f4'));
});

test('direct king attack detection agrees with capture generation across recorded games', () => {
  const games = readFileSync(new URL('../data/ai-games-160.jsonl', import.meta.url), 'utf8')
    .trim().split('\n').map(JSON.parse).filter(row => row.kind === 'game');
  let states = 0;
  for (const game of games) {
    let position = parseFen(game.startFen);
    for (const notation of [null, ...game.moves]) {
      if (notation) position = makeMove(position, { from: squareIndex(notation.slice(0, 2)), to: squareIndex(notation.slice(2)) });
      for (const side of ['red', 'black']) {
        const king = position.board.indexOf(side === 'red' ? 'K' : 'k');
        const enemy = side === 'red' ? 'black' : 'red';
        const reference = king < 0 || pseudoMoves(position, enemy, true).some(move => move.to === king);
        assert.equal(isInCheck(position, side), reference, `game ${game.game}, state ${states}, ${side}`);
      }
      states++;
    }
  }
  assert.equal(states, 17827);
});

test('search selects a legal move and reports completed depth', () => {
  const position = parseFen();
  const result = chooseMove(position, { timeMs: 300, maxDepth: 3 });
  assert.ok(result.depth >= 1);
  assert.ok(legalMoves(position).some(move => moveName(move) === moveName(result.move)));
});

test('teaching analysis ranks distinct legal moves with full root scores', () => {
  const position = parseFen();
  const result = chooseMove(position, { timeMs: 300, maxDepth: 2, fullRootScores: true });
  const allowed = new Set(legalMoves(position).map(moveName));
  assert.ok(result.candidates.length >= 5);
  assert.equal(new Set(result.candidates.map(item => item.move)).size, result.candidates.length);
  assert.ok(result.candidates.every(item => allowed.has(item.move) && Number.isFinite(item.score)));
  assert.ok(result.candidates.every((item, index) => !index || result.candidates[index - 1].score >= item.score));
});

test('multi-line search returns legal continuations and the same top moves as full search', () => {
  const position = parseFen();
  const full = chooseMove(position, { timeMs: 10000, maxDepth: 3, fullRootScores: true });
  const lines = chooseMove(position, { timeMs: 10000, maxDepth: 3, multiPv: 3 });
  assert.equal(lines.depth, 3);
  assert.deepEqual(lines.candidates.map(item => [item.move, item.score]),
    full.candidates.slice(0, 3).map(item => [item.move, item.score]));
  for (const candidate of lines.candidates) {
    assert.equal(candidate.pv[0], candidate.move);
    assert.ok(candidate.pv.length > 1, 'search should show more than the next move');
    let current = position;
    for (const notation of candidate.pv) current = playMove(current, notation);
  }
});

function replay(fen, moves) {
  let position = parseFen(fen);
  const history = [positionKey(position)];
  for (const notation of moves) {
    position = playMove(position, notation);
    history.push(positionKey(position));
  }
  return { position, history };
}

test('the side giving perpetual check loses instead of receiving a repetition draw', () => {
  const cases = [
    ['4k4/4R4/9/9/9/9/9/9/9/5K3 b - - 0 1', ['e9d9', 'e8d8', 'd9e9', 'd8e8'], 'black', '红方长将判负'],
    ['5k3/9/9/9/9/9/9/9/4r4/4K4 w - - 0 1', ['e0d0', 'e1d1', 'd0e0', 'd1e1'], 'red', '黑方长将判负'],
  ];
  for (const [fen, cycle, winner, reason] of cases) {
    const once = replay(fen, cycle);
    assert.equal(gameResult(once.position, once.history), null);
    const twice = replay(fen, [...cycle, ...cycle]);
    assert.deepEqual(gameResult(twice.position, twice.history), { winner, reason });
  }
});

test('quiet repetition remains a draw and search penalizes a third checking cycle', () => {
  const quiet = replay('4k4/9/9/9/9/9/9/9/9/R4K3 b - - 0 1',
    ['e9d9', 'a0a1', 'd9e9', 'a1a0', 'e9d9', 'a0a1', 'd9e9', 'a1a0']);
  assert.deepEqual(gameResult(quiet.position, quiet.history), { winner: null, reason: '三次重复局面' });

  const checking = replay('4k4/4R4/9/9/9/9/9/9/9/5K3 b - - 0 1',
    ['e9d9', 'e8d8', 'd9e9', 'd8e8', 'e9d9', 'e8d8', 'd9e9']);
  const result = chooseMove(checking.position, { timeMs: 3000, maxDepth: 1,
    fullRootScores: true, history: checking.history.slice(0, -1) });
  assert.ok(result.candidates.find(item => item.move === 'd8e8').score < -29000);
});
