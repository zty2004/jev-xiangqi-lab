import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { currentChoiceModel } from '../src/model-selection.js';

test('promoted choice model is the default and can be disabled or overridden', () => {
  assert.equal(path.basename(currentChoiceModel()), 'choice-master-opening-gpu0.pt');
  assert.equal(currentChoiceModel('off'), null);
  assert.equal(currentChoiceModel('./models/custom.pt'), path.resolve('./models/custom.pt'));
});
