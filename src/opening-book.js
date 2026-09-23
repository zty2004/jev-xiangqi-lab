import { readFileSync } from 'node:fs';
import { legalMoves, moveName, parseFen, playMove, positionKey } from './xiangqi.js';

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

export function loadMasterOpeningBook(filename) {
  const book = JSON.parse(readFileSync(filename, 'utf8'));
  if (book.kind !== 'xiangqi-master-opening-book' || !Number.isInteger(book.maxPlies) || !Array.isArray(book.entries))
    throw new Error('invalid master opening book');
  return { maxPlies: book.maxPlies, source: book.source, sourceUrl: book.sourceUrl,
    positions: new Map(book.entries.map(entry => [entry.position, entry.moves])) };
}

export function mirrorMove(notation) {
  const files = 'abcdefghi';
  return files[8 - files.indexOf(notation[0])] + notation[1] + files[8 - files.indexOf(notation[2])] + notation[3];
}

function mirrorPositionKey(position) {
  const board = position.board.map((_, index) => position.board[Math.floor(index / 9) * 9 + 8 - index % 9]);
  return positionKey({ ...position, board });
}

export function masterOpeningCandidates(position, book, { minGames = 5, maxMoves = 3 } = {}) {
  if (!book || !Number.isInteger(position.fullmove) ||
      2 * (position.fullmove - 1) + Number(position.side === 'black') >= book.maxPlies) return [];
  const direct = book.positions.get(positionKey(position));
  const mirrored = book.positions.get(mirrorPositionKey(position));
  const support = items => items?.reduce((total, item) => total + item.masterGames, 0) || 0;
  const useMirror = support(mirrored) > support(direct);
  const entries = useMirror ? mirrored : direct;
  if (!entries?.length) return [];
  const legal = new Set(legalMoves(position).map(moveName));
  const choices = entries.map(item => ({ ...item, move: useMirror ? mirrorMove(item.move) : item.move }))
    .filter(item => legal.has(item.move));
  const screenHorse = position.side === 'black' && choices.some(item => item.screenHorseGames >= minGames);
  const eligible = choices.filter(item => (screenHorse ? item.screenHorseGames : item.masterGames) >= minGames)
    .sort((a, b) => (screenHorse ? b.screenHorseGames - a.screenHorseGames : b.masterGames - a.masterGames) ||
      a.move.localeCompare(b.move));
  if (!eligible.length) return [];
  const first = screenHorse ? eligible[0].screenHorseGames : eligible[0].masterGames;
  return eligible.filter(item => (screenHorse ? item.screenHorseGames : item.masterGames) >= first * 0.25)
    .slice(0, maxMoves).map(item => ({ ...item, mirrored: useMirror, screenHorseRepertoire: screenHorse }));
}
