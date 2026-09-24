import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

export function currentChoiceModel(environmentValue = process.env.CHOICE_MODEL) {
  if (environmentValue === 'off') return null;
  if (environmentValue) return path.resolve(environmentValue);
  const manifestPath = path.join(root, 'models', 'current.json');
  if (!existsSync(manifestPath)) return null;
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (!manifest.model || !/^[A-Za-z0-9._-]+\.pt$/.test(manifest.model) ||
      !/^[a-f0-9]{64}$/.test(manifest.sha256 || '')) throw new Error('Invalid current model manifest');
  const modelPath = path.join(root, 'models', manifest.model);
  if (!existsSync(modelPath)) throw new Error(`Current choice model is missing: ${modelPath}`);
  const actual = createHash('sha256').update(readFileSync(modelPath)).digest('hex');
  if (actual !== manifest.sha256) throw new Error(`Current choice model hash mismatch: ${modelPath}`);
  return modelPath;
}

export function currentNnueModel(environmentValue = process.env.NNUE_MODEL) {
  if (environmentValue === 'off') return null;
  if (environmentValue) return path.resolve(environmentValue);
  const manifestPath = path.join(root, 'models', 'current-value.json');
  if (!existsSync(manifestPath)) return null;
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (!manifest.model || !/^[A-Za-z0-9._-]+\.json$/.test(manifest.model) ||
      !/^[a-f0-9]{64}$/.test(manifest.sha256 || '')) throw new Error('Invalid current value model manifest');
  const modelPath = path.join(root, 'models', manifest.model);
  if (!existsSync(modelPath)) throw new Error(`Current value model is missing: ${modelPath}`);
  const actual = createHash('sha256').update(readFileSync(modelPath)).digest('hex');
  if (actual !== manifest.sha256) throw new Error(`Current value model hash mismatch: ${modelPath}`);
  return modelPath;
}
