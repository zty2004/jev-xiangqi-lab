import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseMove } from '../src/engine.js';
import { START_FEN, isInCheck, legalMoves, makeMove, moveName, parseFen, playMove, toFen } from '../src/xiangqi.js';

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

test('search selects a legal move and reports completed depth', () => {
  const position = parseFen();
  const result = chooseMove(position, { timeMs: 300, maxDepth: 3 });
  assert.ok(result.depth >= 1);
  assert.ok(legalMoves(position).some(move => moveName(move) === moveName(result.move)));
});
