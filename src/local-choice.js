import { spawn } from 'node:child_process';
import readline from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

export function validateChoiceResponse(response, moves) {
  if (response.error || !Array.isArray(response.choices)) throw new Error(response.error || 'Invalid local choice response');
  const allowed = new Set(moves), seen = new Set();
  let total = 0;
  for (const item of response.choices) {
    if (!allowed.has(item.move) || seen.has(item.move) || !Number.isFinite(item.probability) || item.probability < 0 || item.probability > 1)
      throw new Error('Local choice returned an invalid move distribution');
    seen.add(item.move);
    total += item.probability;
  }
  if (seen.size !== allowed.size || Math.abs(total - 1) > 0.001) throw new Error('Local choice distribution does not cover all legal moves');
  return response;
}

export class LocalChoice {
  constructor(modelPath, python = process.env.PYTHON || 'python3') {
    this.process = spawn(python, ['-u', path.join(root, 'train/choice_model.py'), 'serve', '--model', modelPath], { cwd: root, stdio: ['pipe', 'pipe', 'pipe'] });
    this.pending = [];
    this.stderr = '';
    this.process.stderr.on('data', chunk => { this.stderr = (this.stderr + chunk.toString()).slice(-2000); });
    readline.createInterface({ input: this.process.stdout }).on('line', line => {
      const pending = this.pending.shift();
      if (!pending) return;
      try { pending.resolve(JSON.parse(line)); }
      catch (error) { pending.reject(error); }
    });
    this.process.on('error', error => this.pending.splice(0).forEach(item => item.reject(error)));
    this.initialized = this.read(30000).then(message => { if (!message.ready) throw new Error('Local choice model did not start'); });
  }
  read(timeoutMs) {
    return new Promise((resolve, reject) => {
      const pending = { resolve: value => { clearTimeout(timer); resolve(value); }, reject: error => { clearTimeout(timer); reject(error); } };
      const timer = setTimeout(() => {
        this.pending = this.pending.filter(item => item !== pending);
        reject(new Error(`Local choice model timed out: ${this.stderr}`));
      }, timeoutMs);
      this.pending.push(pending);
    });
  }
  async ready() { await this.initialized; }
  async rank(fen, moves, timeoutMs = 10000) {
    await this.ready();
    const result = this.read(timeoutMs);
    this.process.stdin.write(JSON.stringify({ fen, moves }) + '\n');
    const response = await result;
    return validateChoiceResponse(response, moves);
  }
  close() { this.process.kill(); }
}
