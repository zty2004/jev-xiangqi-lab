import assert from 'node:assert/strict';
import test from 'node:test';
import { chineseMove, formatChineseMove } from '../src/chinese-notation.js';
import { loadMasterOpeningBook, loadOpeningLines, masterOpeningCandidates } from '../src/opening-book.js';
import { parseFen, playMove } from '../src/xiangqi.js';

test('traditional Xiangqi notation converts to legal coordinate moves', () => {
  const start = parseFen();
  assert.equal(chineseMove(start, '炮二平五'), 'b2e2');
  const after = playMove(start, 'b2e2');
  assert.equal(chineseMove(after, '馬８進７'), 'b9c7');
  assert.equal(chineseMove(after, '炮二平五'), 'h7e7');
  assert.throws(() => chineseMove(after, '炮一平五'), /matched 0 legal moves/);
});

test('Chinese move labels identify legal moves for both sides', () => {
  const start = parseFen();
  assert.equal(formatChineseMove(start, 'b2e2'), '炮二平五');
  const after = playMove(start, 'b2e2');
  assert.equal(formatChineseMove(after, 'b9c7'), '马８进７');
  assert.equal(chineseMove(after, formatChineseMove(after, 'b9c7')), 'b9c7');
});

test('master book gives the screen horse reply to the central cannon, including the mirror', () => {
  const book = loadMasterOpeningBook(new URL('../data/master-opening-book.json', import.meta.url));
  const left = masterOpeningCandidates(playMove(parseFen(), 'b2e2'), book);
  assert.deepEqual(left.map(item => item.move), ['b9c7']);
  assert.ok(left[0].screenHorseRepertoire);
  const right = masterOpeningCandidates(playMove(parseFen(), 'h2e2'), book);
  assert.deepEqual(right.map(item => item.move), ['h9g7']);
  assert.ok(right[0].mirrored);
  let mainline = parseFen();
  for (const move of ['b2e2', 'b9c7', 'b0c2', 'a9b9', 'a0b0']) mainline = playMove(mainline, move);
  assert.deepEqual(masterOpeningCandidates(mainline, book).map(item => item.move), ['h9g7']);
});

test('CCPD opening book has validated distinct prefixes for paired matches', () => {
  const lines = loadOpeningLines(new URL('../data/openings.json', import.meta.url), 8);
  assert.ok(lines.length > 100);
  assert.equal(new Set(lines.map(line => line.join(' '))).size, lines.length);
  assert.ok(lines.every(line => line.length === 8));
});
