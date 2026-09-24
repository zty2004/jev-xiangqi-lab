import assert from 'node:assert/strict';
import test from 'node:test';
import { NnueEvaluator, NNUE_FEATURES, featureIndex } from '../src/nnue-evaluator.js';
import { legalMoves, makeMove, parseFen } from '../src/xiangqi.js';

function model(hidden = 4, head = 3) {
  const embedding = new Float32Array(NNUE_FEATURES * hidden);
  for (let i = 0; i < embedding.length; i++) embedding[i] = ((i * 17) % 29 - 14) / 1000;
  return { features: NNUE_FEATURES, hidden, head, outputScale: 2000, embedding,
    featureBias: new Float32Array(hidden).fill(0.5),
    fc1Weight: new Float32Array(head * hidden * 4).fill(0.02),
    fc1Bias: new Float32Array(head).fill(0.1),
    outputWeight: new Float32Array(head * 2).fill(0.05), outputBias: -0.1 };
}

test('NNUE uses symmetric feature identities from each player perspective', () => {
  const position = parseFen();
  assert.equal(featureIndex(position, 'red', 'R', 81), featureIndex(position, 'black', 'r', 8));
  assert.equal(featureIndex(position, 'red', 'p', 27), featureIndex(position, 'black', 'P', 62));
});

test('incremental accumulator exactly matches a full refresh', () => {
  const evaluator = new NnueEvaluator(model());
  let position = parseFen(), state = evaluator.createState(position);
  for (let ply = 0; ply < 20; ply++) {
    const moves = legalMoves(position);
    const move = moves[(ply * 13 + 7) % moves.length];
    const next = makeMove(position, move);
    state = evaluator.updateState(position, move, next, state);
    const refreshed = evaluator.createState(next);
    for (const side of ['red', 'black'])
      for (let unit = 0; unit < state[side].length; unit++)
        assert.ok(Math.abs(state[side][unit] - refreshed[side][unit]) < 1e-6);
    assert.equal(evaluator.evaluate(next, state), evaluator.evaluate(next, refreshed));
    position = next;
  }
});
