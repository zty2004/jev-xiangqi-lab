import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gamePhase } from './src/engine.js';
import { PikafishTeacher } from './src/pikafish-teacher.js';
import { parseFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback = null) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = path.resolve(option('--input', 'data/teacher-ccpd-middle-4000-pikafish.jsonl'));
const output = path.resolve(option('--output', 'reports/pikafish-teacher-consistency.json'));
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const positions = Number(option('--positions', 100));
const trials = Number(option('--trials', 2));
const seed = Number(option('--seed', 20260930));
const phase = option('--phase', 'middlegame');
const nodeValue = option('--nodes');
const timeValue = option('--movetime');
const nodes = nodeValue === null ? null : Number(nodeValue);
const moveTime = timeValue === null ? null : Number(timeValue);
if (!binary || !Number.isInteger(positions) || positions < 1 || !Number.isInteger(trials) || trials < 2 || trials > 4 ||
    !Number.isInteger(seed) || !['opening', 'middlegame', 'endgame'].includes(phase) ||
    (nodes === null) === (moveTime === null) || (nodes !== null && (!Number.isInteger(nodes) || nodes < 1)) ||
    (moveTime !== null && (!Number.isInteger(moveTime) || moveTime < 20))) {
  console.error('Usage: node audit-pikafish-consistency.js --pikafish /path/to/Pikafish (--nodes 50000 | --movetime 500) [--positions 100 --trials 2 --phase middlegame]');
  process.exit(2);
}

const bytes = readFileSync(input);
const rows = bytes.toString('utf8').trim().split('\n').filter(Boolean).map(JSON.parse)
  .filter(row => row.kind === 'position' && gamePhase(parseFen(row.fen)) === phase);
let randomState = seed >>> 0;
function random() {
  randomState += 0x6D2B79F5;
  let value = randomState;
  value = Math.imul(value ^ value >>> 15, value | 1);
  value ^= value + Math.imul(value ^ value >>> 7, value | 61);
  return ((value ^ value >>> 14) >>> 0) / 4294967296;
}
for (let index = rows.length - 1; index > 0; index--) {
  const other = Math.floor(random() * (index + 1));
  [rows[index], rows[other]] = [rows[other], rows[index]];
}
const distinctGames = [], repeatedGames = [], seenGames = new Set();
for (const row of rows) {
  const game = String(row.game);
  (seenGames.has(game) ? repeatedGames : distinctGames).push(row);
  seenGames.add(game);
}
const samples = [...distinctGames, ...repeatedGames].slice(0, positions);
if (samples.length < positions) throw new Error(`Only ${samples.length} eligible ${phase} positions`);

const teachers = Array.from({ length: trials }, () => new PikafishTeacher(binary));
const records = [];
try {
  await Promise.all(teachers.map(teacher => teacher.ready()));
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index];
    const results = await Promise.all(teachers.map(async teacher => {
      teacher.send('ucinewgame');
      return teacher.analyse(sample.fen, nodes === null ? moveTime : { nodes });
    }));
    if (results.some(result => !sample.legal.includes(result.best))) throw new Error(`Illegal teacher move at sample ${index}`);
    records.push({ game: sample.game, ply: sample.ply, expected: sample.best,
      best: results.map(result => result.best), depths: results.map(result => result.depth),
      selectiveDepths: results.map(result => result.selectiveDepth), searchedNodes: results.map(result => result.nodes) });
    if ((index + 1) % 10 === 0 || index + 1 === samples.length) console.log(`${index + 1}/${samples.length} audited`);
  }
} finally { teachers.forEach(teacher => teacher.close()); }

const unanimous = records.filter(row => new Set(row.best).size === 1).length;
const storedAgreement = records.map(row => row.best.filter(best => best === row.expected).length / trials);
const allDepths = records.flatMap(row => row.depths), allSelectiveDepths = records.flatMap(row => row.selectiveDepths);
const report = { kind: 'pikafish-consistency-audit', input: path.basename(input),
  inputSha256: createHash('sha256').update(bytes).digest('hex'), phase, positions, trials, seed,
  budget: nodes === null ? { moveTime } : { nodes }, distinctGames: new Set(samples.map(row => row.game)).size,
  unanimousBest: unanimous, unanimousBestRate: unanimous / records.length,
  storedLabelAgreement: storedAgreement.reduce((sum, value) => sum + value, 0) / records.length,
  depth: { min: Math.min(...allDepths), max: Math.max(...allDepths), mean: allDepths.reduce((a, b) => a + b, 0) / allDepths.length },
  selectiveDepth: { min: Math.min(...allSelectiveDepths), max: Math.max(...allSelectiveDepths), mean: allSelectiveDepths.reduce((a, b) => a + b, 0) / allSelectiveDepths.length },
  mismatches: records.filter(row => new Set(row.best).size > 1 || row.best.some(best => best !== row.expected)) };
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output, unanimousBestRate: report.unanimousBestRate,
  storedLabelAgreement: report.storedLabelAgreement, depth: report.depth }));
