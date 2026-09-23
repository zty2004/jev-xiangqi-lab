import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { loadMasterOpeningBook, masterOpeningCandidates } from './src/opening-book.js';
import { legalMoves, moveName, parseFen, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const bookPath = option('--book', 'data/master-opening-book.json');
const output = option('--output', 'data/master-opening-positions.jsonl');
const minGames = Number(option('--min-games', 5));
if (!Number.isInteger(minGames) || minGames < 1) throw new Error('invalid minimum game count');

const raw = await readFile(bookPath), source = JSON.parse(raw);
const book = loadMasterOpeningBook(bookPath);
const rows = [{ kind: 'meta', source: source.source, sourceUrl: source.sourceUrl,
  sourceCommit: source.sourceCommit, license: source.license,
  bookSha256: createHash('sha256').update(raw).digest('hex'), minGames }];
for (const entry of source.entries) {
  const position = parseFen(entry.position);
  const totalGames = entry.moves.reduce((sum, item) => sum + item.masterGames, 0);
  if (totalGames < minGames) continue;
  const candidates = masterOpeningCandidates(position, book, { minGames });
  if (!candidates.length) continue;
  const legal = legalMoves(position).map(moveName);
  const useScreenHorse = candidates[0].screenHorseRepertoire;
  const weights = candidates.map(item => useScreenHorse ? item.screenHorseGames : item.masterGames);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const policy = Object.fromEntries(candidates.map((item, index) => [item.move, weights[index] / totalWeight]));
  const best = candidates[0].move;
  if (!legal.includes(best)) throw new Error(`illegal master-book move ${best} in ${entry.position}`);
  rows.push({ kind: 'position', source: 'opening-book', fen: toFen(position), legal, best,
    policy, count: totalGames, screenHorseRepertoire: useScreenHorse });
}
await writeFile(output, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
console.log(JSON.stringify({ positions: rows.length - 1, output, minGames }));
