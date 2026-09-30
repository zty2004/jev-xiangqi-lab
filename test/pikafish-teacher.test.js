import test from 'node:test';
import assert from 'node:assert/strict';
import { pikafishBudget } from '../src/pikafish-teacher.js';

test('Pikafish teacher accepts reproducible node budgets while keeping movetime compatibility', () => {
  assert.deepEqual(pikafishBudget(500), { command: 'go movetime 500', timeout: 20500, moveTime: 500, nodes: null });
  assert.deepEqual(pikafishBudget({ nodes: 50000, timeoutMs: 9000 }),
    { command: 'go nodes 50000', timeout: 9000, moveTime: null, nodes: 50000 });
  assert.throws(() => pikafishBudget({ nodes: 50000, moveTime: 500 }), /exactly one/);
});
