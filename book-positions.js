import { readFile, writeFile } from 'node:fs/promises';
import { legalMoves, moveName, parseFen, playMove, positionKey, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const bookPath = option('--openings', 'data/openings.json');
const output = option('--output', 'data/opening-positions.jsonl');
const maxPlies = Number(option('--maxplies', 24));
if (!Number.isInteger(maxPlies) || maxPlies < 1) throw new Error('invalid max plies');

const book = JSON.parse(await readFile(bookPath, 'utf8'));
if (book.kind !== 'xiangqi-opening-book') throw new Error('invalid opening book');
const positions = new Map();
for (const entry of book.entries) {
  let position = parseFen();
  for (const notation of entry.moves.slice(0, maxPlies)) {
    const key = positionKey(position);
    if (!positions.has(key)) positions.set(key, { fen: toFen(position), counts: new Map() });
    const record = positions.get(key);
    record.counts.set(notation, (record.counts.get(notation) || 0) + 1);
    position = playMove(position, notation);
  }
}
const rows = [{ kind: 'meta', source: book.source, sourceUrl: book.sourceUrl, license: book.license,
  sourceHash: book.sourceHash, entries: book.entries.length, maxPlies }];
for (const record of positions.values()) {
  const position = parseFen(record.fen);
  const legal = legalMoves(position).map(moveName);
  const total = [...record.counts.values()].reduce((sum, count) => sum + count, 0);
  const policy = Object.fromEntries([...record.counts].map(([move, count]) => [move, count / total]));
  const best = [...record.counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
  if (!legal.includes(best)) throw new Error(`illegal book move ${best} in ${record.fen}`);
  rows.push({ kind: 'position', source: 'opening-book', fen: record.fen, legal, best, policy, count: total });
}
await writeFile(output, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
console.log(`Saved ${rows.length - 1} unique opening positions to ${output}`);
