import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { currentChoiceModel, currentNnueModel } from '../src/model-selection.js';

test('promoted choice model is the default and can be disabled or overridden', () => {
  assert.equal(path.basename(currentChoiceModel()), 'choice-selfplay-round1-finetune-gpu0.pt');
  assert.equal(currentChoiceModel('off'), null);
  assert.equal(currentChoiceModel('./models/custom.pt'), path.resolve('./models/custom.pt'));
});

test('promoted residual value model is the default and can be disabled or overridden', () => {
  assert.equal(path.basename(currentNnueModel()), 'value-nnue-residual-32x16-gpu.json');
  assert.equal(currentNnueModel('off'), null);
  assert.equal(currentNnueModel('./models/custom.json'), path.resolve('./models/custom.json'));
});
