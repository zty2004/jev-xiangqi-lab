import test from 'node:test';
import assert from 'node:assert/strict';
import { rankRecommendations } from '../src/recommendation-ranking.js';

test('recommendations are ranked by search score without mutating the search snapshot', () => {
  const candidates = [
    { move: 'a0a1', score: 33 },
    { move: 'b0b1', score: -7 },
    { move: 'c0c1', score: 7 },
    { move: 'd0d1', score: 31 },
    { move: 'e0e1', score: 37 }
  ];

  assert.deepEqual(rankRecommendations(candidates).map(item => item.score), [37, 33, 31, 7, -7]);
  assert.deepEqual(candidates.map(item => item.score), [33, -7, 7, 31, 37]);
});

test('equal scores preserve the search order', () => {
  const candidates = [{ move: 'a0a1', score: 8 }, { move: 'b0b1', score: 8 }];
  assert.deepEqual(rankRecommendations(candidates).map(item => item.move), ['a0a1', 'b0b1']);
});
