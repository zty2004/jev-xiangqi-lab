import { readFileSync } from 'node:fs';
import { legalMoves, moveName, parseFen, playMove } from './xiangqi.js';

export function loadOpeningLines(filename, plies = 8) {
  if (!Number.isInteger(plies) || plies < 2) throw new Error('opening plies must be at least 2');
  const book = JSON.parse(readFileSync(filename, 'utf8'));
  if (book.kind !== 'xiangqi-opening-book' || !Array.isArray(book.entries)) throw new Error('invalid opening book');
  const seen = new Set(), lines = [];
  for (const entry of book.entries) {
    if (!Array.isArray(entry.moves) || entry.moves.length < plies) continue;
    const moves = entry.moves.slice(0, plies), key = moves.join(' ');
    if (seen.has(key)) continue;
    let position = parseFen();
    for (const notation of moves) {
      if (!legalMoves(position).some(move => moveName(move) === notation)) throw new Error(`illegal book move ${notation} in ${entry.sourceFile}`);
      position = playMove(position, notation);
    }
    seen.add(key);
    lines.push(moves);
  }
  if (!lines.length) throw new Error('opening book has no usable lines');
  return lines;
}
