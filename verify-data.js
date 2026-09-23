import { createReadStream } from 'node:fs';
import readline from 'node:readline';
import { legalMoves, moveName, parseFen } from './src/xiangqi.js';

const filename = process.argv[2];
if (!filename) {
  console.error('Usage: node verify-data.js data/teacher.jsonl');
  process.exit(2);
}

let meta = null, count = 0, duplicateCount = 0, candidateCount = 0;
const positions = new Set(), games = new Set();
for await (const line of readline.createInterface({ input: createReadStream(filename), crlfDelay: Infinity })) {
  const row = JSON.parse(line);
  if (row.kind === 'meta') {
    if (meta || count) throw new Error('Duplicate or misplaced metadata');
    meta = row;
    continue;
  }
  if (row.kind !== 'position' || !meta) throw new Error('Unexpected dataset record');
  const legal = legalMoves(parseFen(row.fen)).map(moveName).sort();
  if (legal.length !== row.legal?.length || legal.some((move, index) => move !== [...row.legal].sort()[index]))
    throw new Error(`Legal move mismatch at position ${count + 1}`);
  if (!legal.includes(row.best)) throw new Error(`Illegal teacher best move at position ${count + 1}`);
  if (row.played && !legal.includes(row.played)) throw new Error(`Illegal played move at position ${count + 1}`);
  const candidates = row.candidates || [];
  if (candidates.some(item => !legal.includes(item.move) || !Number.isFinite(item.score)))
    throw new Error(`Invalid teacher candidate at position ${count + 1}`);
  const key = row.fen.split(/\s+/).slice(0, 2).join(' ');
  if (positions.has(key)) duplicateCount++;
  positions.add(key);
  games.add(row.game);
  candidateCount += candidates.length;
  count++;
}
if (!meta || (Number.isInteger(meta.positions) && count !== meta.positions))
  throw new Error(`Expected ${meta?.positions} positions, found ${count}`);
console.log(JSON.stringify({ positions: count, games: games.size, uniquePositions: positions.size,
  duplicates: duplicateCount, meanCandidates: candidateCount / count,
  openingBookHash: meta.openingBookHash || null }));
