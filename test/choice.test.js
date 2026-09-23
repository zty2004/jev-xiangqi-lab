import assert from 'node:assert/strict';
import test from 'node:test';
import { validateChoiceResponse } from '../src/local-choice.js';

test('choice interface covers precisely the legal move set', () => {
  const moves = ['b2e2', 'h2e2'];
  const response = { choices: [{ move: 'b2e2', probability: 0.7 }, { move: 'h2e2', probability: 0.3 }] };
  assert.equal(validateChoiceResponse(response, moves), response);
  assert.throws(() => validateChoiceResponse({ choices: [{ move: 'b2e2', probability: 1 }] }, moves));
  assert.throws(() => validateChoiceResponse({ choices: [{ move: 'a0a1', probability: 0.7 }, response.choices[1]] }, moves));
});
