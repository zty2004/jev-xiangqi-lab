import assert from 'node:assert/strict';
import test from 'node:test';
import { chineseMove } from '../src/chinese-notation.js';
import { loadOpeningLines } from '../src/opening-book.js';
import { parseFen, playMove } from '../src/xiangqi.js';

test('traditional Xiangqi notation converts to legal coordinate moves', () => {
  const start = parseFen();
  assert.equal(chineseMove(start, '炮二平五'), 'b2e2');
  const after = playMove(start, 'b2e2');
  assert.equal(chineseMove(after, '馬８進７'), 'b9c7');
  assert.equal(chineseMove(after, '炮二平五'), 'h7e7');
  assert.throws(() => chineseMove(after, '炮一平五'), /matched 0 legal moves/);
});

test('CCPD opening book has validated distinct prefixes for paired matches', () => {
  const lines = loadOpeningLines(new URL('../data/openings.json', import.meta.url), 8);
  assert.ok(lines.length > 100);
  assert.equal(new Set(lines.map(line => line.join(' '))).size, lines.length);
  assert.ok(lines.every(line => line.length === 8));
});
