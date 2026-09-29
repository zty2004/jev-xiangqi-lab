import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { PikafishTeacher } from './src/pikafish-teacher.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = path.resolve(option('--input', 'data/selfplay-round1-positions.jsonl'));
const output = path.resolve(option('--output', 'data/teacher-selfplay-round1-pikafish.jsonl'));
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const limit = Number(option('--positions', 2400));
const moveTime = Number(option('--movetime', 1000));
const seed = Number(option('--seed', 20260930));
const resume = args.includes('--resume');
const planOnly = args.includes('--plan-only');
const excludeFiles = args.flatMap((item, index) => item === '--exclude-teacher' ? [path.resolve(args[index + 1])] : []);

if (!Number.isInteger(limit) || limit < 1 || !Number.isInteger(moveTime) || moveTime < 20 ||
    !Number.isInteger(seed) || (resume && planOnly) || (!planOnly && !binary)) {
  console.error('Usage: node label-selfplay-pikafish.js --pikafish /path/to/Pikafish [--positions 2400] [--movetime 1000] [--exclude-teacher data/teacher.jsonl] [--resume]');
  process.exit(2);
}

function fileHash(filename) { return createHash('sha256').update(readFileSync(filename)).digest('hex'); }
function jsonLines(filename) { return readFileSync(filename, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse); }
function positionKey(fen) { return fen.split(/\s+/).slice(0, 2).join(' '); }
function randomGenerator(value) {
  let state = value >>> 0;
  return () => { state += 0x6D2B79F5; let next = state; next = Math.imul(next ^ next >>> 15, next | 1); next ^= next + Math.imul(next ^ next >>> 7, next | 61); return ((next ^ next >>> 14) >>> 0) / 4294967296; };
}
function shuffle(items, random) {
  for (let index = items.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [items[index], items[other]] = [items[other], items[index]];
  }
}

function selectPositions() {
  const rows = jsonLines(input);
  const sourceMeta = rows.shift();
  if (sourceMeta?.kind !== 'meta' || sourceMeta.schema !== 'jev-selfplay-v1') throw new Error('Expected Jev self-play metadata');
  const excluded = new Set();
  for (const filename of excludeFiles) {
    for (const row of jsonLines(filename)) if (row.kind === 'position') excluded.add(positionKey(row.fen));
  }
  const seen = new Set(excluded), phases = new Map([['opening', []], ['middlegame', []], ['endgame', []]]);
  const skipped = { excluded: 0, duplicate: 0, malformed: 0 };
  for (const row of rows) {
    if (row.kind !== 'position' || !Array.isArray(row.legal) || !row.legal.length || !phases.has(row.phase)) { skipped.malformed++; continue; }
    const key = positionKey(row.fen);
    if (excluded.has(key)) { skipped.excluded++; continue; }
    if (seen.has(key)) { skipped.duplicate++; continue; }
    seen.add(key);
    phases.get(row.phase).push({ game: row.game, ply: row.ply, fen: row.fen, legal: row.legal, phase: row.phase });
  }
  const quotas = { opening: Math.floor(limit * 0.25), middlegame: Math.floor(limit * 0.50) };
  quotas.endgame = limit - quotas.opening - quotas.middlegame;
  const random = randomGenerator(seed), selected = [];
  for (const [phase, quota] of Object.entries(quotas)) {
    const candidates = phases.get(phase);
    shuffle(candidates, random);
    if (candidates.length < quota) throw new Error(`Only ${candidates.length} ${phase} positions, need ${quota}`);
    selected.push(...candidates.slice(0, quota));
  }
  shuffle(selected, random);
  return { selected, summary: { source: path.basename(input), sourceSha256: fileHash(input), eligibleByPhase: Object.fromEntries([...phases].map(([phase, rows]) => [phase, rows.length])), selectedByPhase: quotas, excludedPositions: excluded.size, skipped } };
}

async function main() {
  const { selected, summary } = selectPositions();
  const metadata = { kind: 'meta', model: 'Pikafish', source: 'Jev self-play positions re-labelled by Pikafish',
    teacherBinarySha256: binary ? fileHash(binary) : null, positions: selected.length, moveTime, seed,
    sampling: 'unique self-play positions, phase quotas 25/50/25',
    excludeSources: excludeFiles.map(filename => ({ file: path.basename(filename), sha256: fileHash(filename) })), ...summary };
  if (planOnly) { console.log(JSON.stringify(metadata)); return; }
  let done = 0;
  if (resume) {
    if (!existsSync(output)) throw new Error('Cannot resume missing output');
    const prior = jsonLines(output);
    if (JSON.stringify(prior.shift()) !== JSON.stringify(metadata)) throw new Error('Resume metadata mismatch');
    for (const row of prior) {
      const expected = selected[done++];
      if (!expected || row.kind !== 'position' || row.game !== expected.game || row.ply !== expected.ply || row.fen !== expected.fen || !expected.legal.includes(row.best)) throw new Error(`Invalid resume row ${done}`);
    }
  } else {
    if (existsSync(output)) throw new Error(`Output exists: ${output}. Use --resume or a new path.`);
    writeFileSync(output, JSON.stringify(metadata) + '\n');
  }
  const teacher = new PikafishTeacher(binary);
  try {
    await teacher.ready();
    for (let index = done; index < selected.length; index++) {
      const sample = selected[index];
      teacher.send('ucinewgame');
      const analysis = await teacher.analyse(sample.fen, moveTime);
      if (!sample.legal.includes(analysis.best)) throw new Error(`Illegal teacher move at sample ${index}`);
      appendFileSync(output, JSON.stringify({ kind: 'position', ...sample, source: 'selfplay-relabelled-pikafish',
        best: analysis.best, candidates: analysis.candidates.filter(item => sample.legal.includes(item.move)), depth: analysis.depth }) + '\n');
      if ((index + 1) % 100 === 0 || index + 1 === selected.length) console.log(`${index + 1}/${selected.length} labelled`);
    }
  } finally { teacher.close(); }
  console.log(JSON.stringify({ output, positions: selected.length, ...summary }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
