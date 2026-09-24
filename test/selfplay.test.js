import assert from 'node:assert/strict';
import test from 'node:test';
import { randomGenerator, samplePolicy, scorePolicy } from '../selfplay.js';

test('self-play score policy is normalized and prefers stronger search moves', () => {
  const policy = scorePolicy([{ move: 'a0a1', score: 120 }, { move: 'b0c2', score: 40 }, { move: 'h0g2', score: -50 }], 80);
  assert.ok(policy[0].probability > policy[1].probability);
  assert.ok(policy[1].probability > policy[2].probability);
  assert.ok(Math.abs(policy.reduce((sum, item) => sum + item.probability, 0) - 1) < 1e-12);
});

test('self-play sampling is reproducible from its seed', () => {
  const policy = [{ move: 'a', probability: 0.2 }, { move: 'b', probability: 0.8 }];
  const first = randomGenerator(17), second = randomGenerator(17);
  assert.deepEqual(Array.from({ length: 20 }, () => samplePolicy(policy, first)),
    Array.from({ length: 20 }, () => samplePolicy(policy, second)));
});
