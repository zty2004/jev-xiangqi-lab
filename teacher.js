import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createWriteStream, readFileSync } from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';
import { loadOpeningLines } from './src/opening-book.js';

const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const positions = Number(option('--positions', 1000));
const moveTime = Number(option('--movetime', 200));
const maxPlies = Number(option('--maxplies', 120));
const seed = Number(option('--seed', 20260923));
const output = path.resolve(option('--output', 'teacher-data.jsonl'));
const openingBook = option('--openings');
const openingPlies = Number(option('--opening-plies', 8));
if (!binary || !Number.isInteger(positions) || positions < 1 || !Number.isInteger(moveTime) || moveTime < 20) {
  console.error('Usage: node teacher.js --pikafish /absolute/path/to/pikafish [--positions 1000] [--movetime 200] [--seed 20260923] [--output teacher-data.jsonl]');
  process.exit(2);
}

function randomGenerator(value) {
  let state = value >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const random = randomGenerator(seed);

class Teacher {
  constructor(filename) {
    this.process = spawn(filename, [], { cwd: path.dirname(path.resolve(filename)), stdio: ['pipe', 'pipe', 'pipe'] });
    this.waiters = []; this.lines = []; this.stderr = '';
    readline.createInterface({ input: this.process.stdout }).on('line', line => {
      for (const waiter of [...this.waiters]) waiter(line);
    });
    this.process.stderr.on('data', chunk => { this.stderr = (this.stderr + chunk.toString()).slice(-2000); });
  }
  send(command) { this.process.stdin.write(command + '\n'); }
  until(predicate, timeout = 10000, collect = null) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.waiters = this.waiters.filter(item => item !== onLine); reject(new Error(`Pikafish timed out: ${this.stderr}`)); }, timeout);
      const onLine = line => {
        collect?.(line);
        if (!predicate(line)) return;
        clearTimeout(timer); this.waiters = this.waiters.filter(item => item !== onLine); resolve(line);
      };
      this.waiters.push(onLine);
    });
  }
  async ready() {
    const uci = this.until(line => line === 'uciok'); this.send('uci'); await uci;
    this.send('setoption name MultiPV value 8');
    const ready = this.until(line => line === 'readyok'); this.send('isready'); await ready;
  }
  async analyse(fen, ms) {
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
    this.send(`position fen ${fen}`); this.send(`go movetime ${ms}`);
    const line = await finished, best = line.split(/\s+/)[1];
    const selectedDepth = [...byDepth.keys()].sort((a, b) => byDepth.get(b).size - byDepth.get(a).size || b - a)[0] || 0;
    const candidates = [...(byDepth.get(selectedDepth)?.entries() || [])].sort((a, b) => a[0] - b[0]).map(([rank, item]) => ({ rank, ...item }));
    return { best, depth: selectedDepth, candidates };
  }
  close() { this.send('quit'); this.process.kill(); }
}

const openings = openingBook ? loadOpeningLines(openingBook, openingPlies) :
  [[], ['b2e2', 'b7e7'], ['h2e2', 'h7e7'], ['b0c2', 'b9c7'], ['h0g2', 'h9g7']];
if (openingBook) {
  const openingRandom = randomGenerator(seed ^ 0x5eeda11);
  for (let i = openings.length - 1; i > 0; i--) {
    const j = Math.floor(openingRandom() * (i + 1));
    [openings[i], openings[j]] = [openings[j], openings[i]];
  }
}
function advance(position, notation) {
  const move = legalMoves(position).find(item => moveName(item) === notation);
  if (!move) throw new Error(`Illegal teacher move ${notation} in ${toFen(position)}`);
  return makeMove(position, move);
}

async function main() {
  const teacher = new Teacher(binary), writer = createWriteStream(output, { flags: 'w' });
  try {
    await teacher.ready();
    writer.write(JSON.stringify({ kind: 'meta', model: 'Pikafish', positions, moveTime, seed,
      teacherBinarySha256: createHash('sha256').update(readFileSync(binary)).digest('hex'),
      openingBook: openingBook || null, openingPlies: openingBook ? openingPlies : null, openingLines: openings.length,
      openingBookHash: openingBook ? createHash('sha256').update(readFileSync(openingBook)).digest('hex') : null,
      generatedAt: new Date().toISOString() }) + '\n');
    let written = 0;
    for (let game = 0; written < positions; game++) {
      teacher.send('ucinewgame');
      let position = parseFen(), history = [positionKey(position)], ply = 0;
      for (const notation of openings[game % openings.length]) { position = advance(position, notation); history.push(positionKey(position)); ply++; }
      while (written < positions && ply < maxPlies && !gameResult(position, history)) {
        const legal = legalMoves(position).map(moveName), fen = toFen(position);
        const analysis = await teacher.analyse(fen, moveTime);
        if (!legal.includes(analysis.best)) throw new Error(`Pikafish returned illegal move ${analysis.best} in ${fen}`);
        const candidates = analysis.candidates.filter(item => legal.includes(item.move));
        writer.write(JSON.stringify({ kind: 'position', game, ply, fen, legal, best: analysis.best, candidates, depth: analysis.depth }) + '\n');
        written++;
        const varied = random() < 0.2 && candidates.length >= 2;
        const played = varied ? candidates[Math.min(candidates.length - 1, Math.floor(random() * 3))].move : analysis.best;
        position = advance(position, played); history.push(positionKey(position)); ply++;
        if (written % 100 === 0) console.log(`${written}/${positions} positions, ${game + 1} games`);
      }
    }
    console.log(`Saved ${written} positions to ${output}`);
  } finally { teacher.close(); writer.end(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
