import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, statSync } from 'node:fs';
import readline from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';
import { loadOpeningLines } from './src/opening-book.js';
import { currentChoiceModel } from './src/model-selection.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
}
const pikafishPath = option('--pikafish', process.env.PIKAFISH_PATH);
const gameCount = Number(option('--games', 4));
const startGame = Number(option('--start-game', 0));
const moveTime = Number(option('--movetime', 5000));
const maxPlies = Number(option('--maxplies', 240));
const outputPath = path.resolve(option('--output', 'benchmark-results.jsonl'));
const openingBook = option('--openings');
const openingPlies = Number(option('--opening-plies', 8));
const choiceModelPath = currentChoiceModel();
if (!pikafishPath || !Number.isInteger(gameCount) || gameCount < 2 || gameCount % 2 ||
    !Number.isInteger(startGame) || startGame < 0 || startGame >= gameCount ||
    !Number.isInteger(moveTime) || moveTime < 10) {
  console.error('Usage: node benchmark.js --pikafish /path/to/pikafish [--games EVEN_NUMBER] [--start-game INDEX] [--movetime 5000] [--output results.jsonl]');
  process.exit(2);
}

async function fileHash(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

class UciEngine {
  constructor(command, commandArgs, name, cwd = root) {
    this.name = name;
    this.process = spawn(command, commandArgs, { cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    this.lines = [];
    this.waiters = [];
    this.errors = '';
    readline.createInterface({ input: this.process.stdout }).on('line', line => {
      if (line.startsWith('id name ')) this.idName = line.slice(8);
      const index = this.waiters.findIndex(waiter => waiter.test(line));
      if (index >= 0) this.waiters.splice(index, 1)[0].resolve(line);
      else { this.lines.push(line); if (this.lines.length > 100) this.lines.shift(); }
    });
    this.process.stderr.on('data', chunk => { this.errors = (this.errors + chunk.toString()).slice(-2000); });
  }
  wait(test, timeoutMs) {
    const index = this.lines.findIndex(test);
    if (index >= 0) return Promise.resolve(this.lines.splice(index, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = { test, resolve: line => { clearTimeout(timer); resolve(line); } };
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter(item => item !== waiter);
        reject(new Error(`${this.name} timed out. ${this.errors}`));
      }, timeoutMs);
      this.waiters.push(waiter);
    });
  }
  send(line) { this.process.stdin.write(`${line}\n`); }
  async ready() {
    this.send('uci');
    await this.wait(line => line === 'uciok', 10000);
    this.send('isready');
    await this.wait(line => line === 'readyok', 10000);
  }
  async move(fen, ms, historyMoves = null) {
    this.lines = [];
    this.send(historyMoves ? `position startpos${historyMoves.length ? ` moves ${historyMoves.join(' ')}` : ''}` : `position fen ${fen}`);
    this.send(`go movetime ${ms}`);
    const line = await this.wait(line => line.startsWith('bestmove '), ms + 20000);
    return line.split(/\s+/)[1];
  }
  close() {
    this.send('quit');
    setTimeout(() => { if (!this.process.killed) this.process.kill(); }, 1000).unref();
  }
}

const openings = openingBook ? loadOpeningLines(openingBook, openingPlies) : [
  [],
  ['b2e2', 'b7e7'],
  ['h2e2', 'h7e7'],
  ['b0c2', 'b9c7'],
  ['h0g2', 'h9g7'],
];

function apply(position, notation) {
  const move = legalMoves(position).find(candidate => moveName(candidate) === notation);
  if (!move) throw new Error(`Illegal move: ${notation} in ${toFen(position)}`);
  return makeMove(position, move);
}

async function main() {
  statSync(pikafishPath);
  const baseline = new UciEngine(process.execPath, [path.join(root, 'uci.js')], 'baseline');
  const pikafish = new UciEngine(pikafishPath, [], 'Pikafish', path.dirname(path.resolve(pikafishPath)));
  const out = createWriteStream(outputPath, { flags: 'a' });
  try {
    await Promise.all([baseline.ready(), pikafish.ready()]);
    const settings = { moveTime, gameCount, startGame, maxPlies, openings, nodeVersion: process.version,
      openingBookHash: openingBook ? await fileHash(openingBook) : null,
      masterOpeningBookHash: process.env.OPENING_BOOK === 'off' ? null :
        await fileHash(process.env.OPENING_BOOK || path.join(root, 'data/master-opening-book.json')),
      openingPlies: openingBook ? openingPlies : null,
      baselineName: baseline.idName, pikafishName: pikafish.idName,
      baselineHash: await fileHash(path.join(root, 'src/engine.js')),
      uciHash: await fileHash(path.join(root, 'uci.js')),
      benchmarkHarnessHash: await fileHash(path.join(root, 'benchmark.js')),
      rulesHash: await fileHash(path.join(root, 'src/xiangqi.js')),
      choiceModelHash: choiceModelPath ? await fileHash(choiceModelPath) : null,
      choiceCodeHash: choiceModelPath ? await fileHash(path.join(root, 'train/choice_model.py')) : null,
      choiceAdapterHash: choiceModelPath ? await fileHash(path.join(root, 'src/local-choice.js')) : null,
      pikafishHash: await fileHash(pikafishPath) };
    out.write(JSON.stringify({ kind: 'run', date: new Date().toISOString(), settings }) + '\n');
    for (let game = startGame; game < gameCount; game++) {
      const opening = openings[Math.floor(game / 2) % openings.length];
      const baselineSide = game % 2 === 0 ? 'red' : 'black';
      let position = parseFen(), moves = [], history = [positionKey(position)], result = null;
      baseline.send('ucinewgame'); pikafish.send('ucinewgame');
      for (const notation of opening) {
        position = apply(position, notation); moves.push(notation); history.push(positionKey(position));
      }
      for (let ply = moves.length; ply < maxPlies; ply++) {
        result = gameResult(position, history);
        if (result) break;
        const engine = position.side === baselineSide ? baseline : pikafish;
        let notation;
        try { notation = await engine.move(toFen(position), moveTime, moves); }
        catch (error) {
          out.write(JSON.stringify({ kind: 'aborted', game: game + 1, baselineSide, opening,
            reason: `${engine.name} error: ${error.message}`, plies: moves.length, moves,
            fen: toFen(position) }) + '\n');
          console.error(`Game ${game + 1}/${gameCount} aborted at ply ${moves.length}: ${engine.name} error: ${error.message}`);
          process.exitCode = 1;
          return;
        }
        if (!legalMoves(position).some(move => moveName(move) === notation)) {
          result = { winner: position.side === 'red' ? 'black' : 'red', reason: `${engine.name} illegal move: ${notation}` }; break;
        }
        position = apply(position, notation);
        moves.push(notation);
        history.push(positionKey(position));
      }
      result ||= gameResult(position, history) || { winner: null, reason: 'maximum plies' };
      const row = { kind: 'game', game: game + 1, baselineSide, opening, result,
        baselineScore: result.winner === null ? 0.5 : result.winner === baselineSide ? 1 : 0,
        plies: moves.length, moves, finalFen: toFen(position) };
      out.write(JSON.stringify(row) + '\n');
      console.log(`Game ${game + 1}/${gameCount}: ${row.baselineScore} (${result.reason}), ${moves.length} plies`);
    }
    console.log(`Results: ${outputPath}`);
  } finally {
    baseline.close(); pikafish.close(); out.end();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
