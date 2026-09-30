import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePikafishInfo, pikafishBudget } from '../src/pikafish-teacher.js';

test('Pikafish teacher accepts reproducible node budgets while keeping movetime compatibility', () => {
  assert.deepEqual(pikafishBudget(500), { command: 'go movetime 500', timeout: 20500, moveTime: 500, nodes: null });
  assert.deepEqual(pikafishBudget({ nodes: 50000, timeoutMs: 9000 }),
    { command: 'go nodes 50000', timeout: 9000, moveTime: null, nodes: 50000 });
  assert.throws(() => pikafishBudget({ nodes: 50000, moveTime: 500 }), /exactly one/);
});

test('Pikafish teacher preserves the complete principal variation', () => {
  assert.deepEqual(parsePikafishInfo(
    'info depth 12 seldepth 27 multipv 2 score cp 33 nodes 250123 nps 900000 pv b2e2 h9g7 b0c2'), {
    depth: 12, selectiveDepth: 27, nodes: 250123, rank: 2, move: 'b2e2',
    pv: ['b2e2', 'h9g7', 'b0c2'], score: 33, scoreType: 'cp'
  });
  assert.equal(parsePikafishInfo('bestmove b2e2'), null);
});
