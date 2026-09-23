import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';
import { PikafishTeacher } from './src/pikafish-teacher.js';

const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const input = path.resolve(option('--input', 'data/ccpd-computer-games.jsonl'));
const output = path.resolve(option('--output', 'data/teacher-ccpd-competition.jsonl'));
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const limit = Number(option('--positions', 1200));
const moveTime = Number(option('--movetime', 500));
const seed = Number(option('--seed', 20260923));
const includeHumanMachine = args.includes('--include-human-machine');
const planOnly = args.includes('--plan-only');
const resume = args.includes('--resume');
const excludeFiles = args.flatMap((item, index) => item === '--exclude-teacher' ? [path.resolve(args[index + 1])] : []);
if (!Number.isInteger(limit) || limit < 1 || !Number.isInteger(moveTime) || moveTime < 20 ||
    !Number.isInteger(seed) || (resume && planOnly) || (!planOnly && !binary)) {
  console.error('Usage: node label-ccpd-games.js --pikafish /path/to/Pikafish --positions 1200 --movetime 500 --exclude-teacher data/teacher.jsonl [--resume] [--plan-only]');
  process.exit(2);
}

function sha256(buffer) { return createHash('sha256').update(buffer).digest('hex'); }
function fileSha256(filename) { return sha256(readFileSync(filename)); }
function jsonLines(filename) { return readFileSync(filename, 'utf8').trim().split('\n').map(JSON.parse); }
function randomGenerator(value) {
  let state = value >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function shuffle(items, random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}
function phase(ply) { return ply < 32 ? 0 : ply < 80 ? 1 : 2; }

function selectPositions() {
  const rows = jsonLines(input);
  if (rows.shift()?.kind !== 'meta') throw new Error('Missing game metadata');
  const exclude = new Set();
  for (const filename of excludeFiles) {
    for (const row of jsonLines(filename)) {
      if (row.kind === 'position') exclude.add(row.fen.split(/\s+/).slice(0, 2).join(' '));
    }
  }
  const seen = new Set(exclude);
  const byPhase = Array.from({ length: 3 }, () => new Map());
  const skipped = { existing: 0, duplicate: 0, terminalTail: 0 };
  let games = 0;
  for (const row of rows) {
    if (row.kind !== 'game' || (!includeHumanMachine && row.sourceCategory !== '電腦對局競賽')) continue;
    games++;
    let position = parseFen(row.startFen);
    const history = [positionKey(position)];
    for (let ply = 0; ply < row.moves.length; ply++) {
      if (gameResult(position, history)) {
        skipped.terminalTail += row.moves.length - ply;
        break;
      }
      const fen = toFen(position);
      const key = positionKey(position);
      if (exclude.has(key)) skipped.existing++;
      else if (seen.has(key)) skipped.duplicate++;
      else if (ply >= 8) {
        const legal = legalMoves(position).map(moveName);
        if (!legal.includes(row.moves[ply])) throw new Error(`Illegal move in source game ${row.game}, ply ${ply}`);
        const candidate = { game: row.game, ply, fen, legal, played: row.moves[ply],
          sourceCategory: row.sourceCategory, sourceFile: row.sourceFile, sourceSha256: row.sourceSha256,
          historyMoves: row.moves.slice(0, ply) };
        const group = byPhase[phase(ply)];
        if (!group.has(row.game)) group.set(row.game, []);
        group.get(row.game).push(candidate);
        seen.add(key);
      }
      const move = legalMoves(position).find(item => moveName(item) === row.moves[ply]);
      if (!move) throw new Error(`Illegal move in source game ${row.game}, ply ${ply}`);
      position = makeMove(position, move);
      history.push(positionKey(position));
    }
  }
  const available = byPhase.map(group => [...group.values()].reduce((sum, items) => sum + items.length, 0));
  if (available.reduce((a, b) => a + b, 0) < limit) throw new Error(`Only ${available.reduce((a, b) => a + b, 0)} eligible positions for ${limit} requested`);
  const random = randomGenerator(seed);
  const queues = byPhase.map(group => {
    const entries = [...group.entries()];
    shuffle(entries, random);
    for (const [, items] of entries) shuffle(items, random);
    return entries;
  });
  const targets = [Math.floor(limit * 0.25), Math.floor(limit * 0.4)];
  targets.push(limit - targets[0] - targets[1]);
  const selected = [], counts = [0, 0, 0];
  for (let p = 0; p < 3; p++) {
    let cursor = 0;
    while (counts[p] < targets[p] && queues[p].some(([, items]) => items.length)) {
      const [, items] = queues[p][cursor % queues[p].length];
      cursor++;
      if (!items.length) continue;
      selected.push(items.pop());
      counts[p]++;
    }
  }
  if (selected.length < limit) throw new Error(`Phase quota unavailable: selected ${selected.length} of ${limit}`);
  shuffle(selected, random);
  return { selected, summary: { sourceGames: games, excludedPositions: exclude.size, eligibleByPhase: available,
    selectedByPhase: counts, selectedGames: new Set(selected.map(row => row.game)).size, skipped } };
}

async function main() {
  const { selected, summary } = selectPositions();
  if (planOnly) {
    console.log(JSON.stringify({ ...summary, positions: selected.length, selectionSha256: sha256(JSON.stringify(selected)) }));
    return;
  }
  const metadata = { kind: 'meta', model: 'Pikafish', source: 'CCPD computer competition',
    sourceFile: path.basename(input), sourceSha256: fileSha256(input),
    teacherBinarySha256: fileSha256(binary), positions: selected.length, moveTime, seed,
    includeHumanMachine, excludeSources: excludeFiles.map(filename => ({ file: path.basename(filename), sha256: fileSha256(filename) })),
    selectionSha256: sha256(JSON.stringify(selected)), sampling: 'game-round-robin phase quotas 25/40/35', ...summary };
  let done = 0;
  if (resume) {
    if (!existsSync(output)) throw new Error('Cannot resume missing output');
    const prior = jsonLines(output);
    if (JSON.stringify(prior.shift()) !== JSON.stringify(metadata)) throw new Error('Resume metadata mismatch');
    for (const row of prior) {
      const expected = selected[done++];
      if (!expected || row.kind !== 'position' || row.game !== expected.game || row.ply !== expected.ply || row.fen !== expected.fen ||
          !expected.legal.includes(row.best)) throw new Error(`Invalid resume row ${done}`);
    }
  } else {
    if (existsSync(output)) throw new Error(`Output exists: ${output}. Use --resume or a new path.`);
    writeFileSync(output, JSON.stringify(metadata) + '\n');
  }
  const teacher = new PikafishTeacher(binary);
  try {
    await teacher.ready();
    for (let i = done; i < selected.length; i++) {
      const sample = selected[i];
      teacher.send('ucinewgame');
      const analysis = await teacher.analyse(sample.fen, moveTime, sample.historyMoves);
      if (!sample.legal.includes(analysis.best)) throw new Error(`Illegal teacher move at sample ${i}`);
      const { historyMoves, ...row } = sample;
      appendFileSync(output, JSON.stringify({ kind: 'position', ...row, best: analysis.best,
        candidates: analysis.candidates.filter(item => sample.legal.includes(item.move)), depth: analysis.depth }) + '\n');
      if ((i + 1) % 100 === 0 || i + 1 === selected.length) console.log(`${i + 1}/${selected.length} labelled`);
    }
  } finally { teacher.close(); }
  console.log(JSON.stringify({ output, ...summary, positions: selected.length }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
