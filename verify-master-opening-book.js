import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { legalMoves, moveName, parseFen } from './src/xiangqi.js';

const [bookFile = 'data/master-opening-book.json', reportFile = 'reports/master-opening-book.json'] = process.argv.slice(2);
const bytes = readFileSync(bookFile), book = JSON.parse(bytes), report = JSON.parse(readFileSync(reportFile));
const hash = createHash('sha256').update(bytes).digest('hex');
if (book.kind !== 'xiangqi-master-opening-book' || hash !== report.bookSha256 ||
    book.sourceTreeSha256 !== report.sourceTreeSha256 || book.positions !== book.entries.length)
  throw new Error('Invalid opening book metadata or hash');
const positions = new Set();
let choices = 0, screenHorseChoices = 0;
for (const entry of book.entries) {
  if (positions.has(entry.position) || !entry.moves.length) throw new Error(`Duplicate or empty position ${entry.position}`);
  positions.add(entry.position);
  const legal = new Set(legalMoves(parseFen(entry.position)).map(moveName));
  const moves = new Set();
  for (const item of entry.moves) {
    if (moves.has(item.move) || !legal.has(item.move) || !Number.isInteger(item.masterGames) || item.masterGames < 1 ||
        !Number.isInteger(item.screenHorseGames) || item.screenHorseGames < 0 || item.screenHorseGames > item.masterGames ||
        !Array.isArray(item.sourceExamples) || item.sourceExamples.length < 1 || item.sourceExamples.length > 2)
      throw new Error(`Invalid opening move ${item.move} at ${entry.position}`);
    moves.add(item.move);
    choices++;
    screenHorseChoices += Number(item.screenHorseGames > 0);
  }
}
console.log(JSON.stringify({ games: book.games, positions: positions.size, choices, screenHorseChoices,
  bookSha256: hash, sourceTreeSha256: book.sourceTreeSha256 }));
