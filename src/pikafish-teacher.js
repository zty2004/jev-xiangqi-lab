import { spawn } from 'node:child_process';
import path from 'node:path';
import readline from 'node:readline';

export class PikafishTeacher {
  constructor(filename) {
    this.process = spawn(filename, [], { cwd: path.dirname(path.resolve(filename)), stdio: ['pipe', 'pipe', 'pipe'] });
    this.waiters = [];
    this.stderr = '';
    readline.createInterface({ input: this.process.stdout }).on('line', line => {
      for (const waiter of [...this.waiters]) waiter(line);
    });
    this.process.stderr.on('data', chunk => { this.stderr = (this.stderr + chunk.toString()).slice(-2000); });
  }

  send(command) { this.process.stdin.write(command + '\n'); }

  until(predicate, timeout = 10000, collect = null) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter(item => item !== onLine);
        reject(new Error(`Pikafish timed out: ${this.stderr}`));
      }, timeout);
      const onLine = line => {
        collect?.(line);
        if (!predicate(line)) return;
        clearTimeout(timer);
        this.waiters = this.waiters.filter(item => item !== onLine);
        resolve(line);
      };
      this.waiters.push(onLine);
    });
  }

  async ready() {
    const uci = this.until(line => line === 'uciok');
    this.send('uci');
    await uci;
    this.send('setoption name MultiPV value 8');
    const ready = this.until(line => line === 'readyok');
    this.send('isready');
    await ready;
  }

  async analyse(fen, ms, historyMoves = null) {
    const byDepth = new Map();
    const finished = this.until(line => line.startsWith('bestmove '), ms + 20000, line => {
      if (!line.startsWith('info ') || !line.includes(' multipv ') || !line.includes(' pv ')) return;
      const depth = Number(line.match(/\bdepth (\d+)/)?.[1]);
      const rank = Number(line.match(/\bmultipv (\d+)/)?.[1]);
      const score = line.match(/\bscore (cp|mate) (-?\d+)/);
      const move = line.match(/\bpv ([a-i][0-9][a-i][0-9])/);
      if (!depth || !rank || !score || !move) return;
      if (!byDepth.has(depth)) byDepth.set(depth, new Map());
      byDepth.get(depth).set(rank, { move: move[1], score: Number(score[2]), scoreType: score[1] });
    });
    this.send(historyMoves ? `position startpos${historyMoves.length ? ` moves ${historyMoves.join(' ')}` : ''}` : `position fen ${fen}`);
    this.send(`go movetime ${ms}`);
    const line = await finished;
    const best = line.split(/\s+/)[1];
    const uniqueCount = depth => new Set([...byDepth.get(depth).values()].map(item => item.move)).size;
    const selectedDepth = [...byDepth.keys()].sort((a, b) => uniqueCount(b) - uniqueCount(a) || b - a)[0] || 0;
    const seen = new Set();
    const candidates = [...(byDepth.get(selectedDepth)?.entries() || [])]
      .sort((a, b) => a[0] - b[0])
      .filter(([, item]) => { if (seen.has(item.move)) return false; seen.add(item.move); return true; })
      .map(([rank, item]) => ({ rank, ...item }));
    return { best, depth: selectedDepth, candidates };
  }

  close() {
    if (!this.process.killed) {
      this.send('quit');
      this.process.kill();
    }
  }
}
