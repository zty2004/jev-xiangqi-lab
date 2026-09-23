import { createReadStream } from 'node:fs';
import readline from 'node:readline';
import { gameResult, legalMoves, moveName, parseFen, playMove, positionKey, toFen } from './src/xiangqi.js';

const [gamesFile, teacherFile] = process.argv.slice(2);
if (!gamesFile || !teacherFile) {
  console.error('Usage: node verify-games.js data/games.jsonl data/teacher.jsonl');
  process.exit(2);
}

async function* records(filename) {
  for await (const line of readline.createInterface({ input: createReadStream(filename), crlfDelay: Infinity }))
    if (line.trim()) yield JSON.parse(line);
}

const games = new Map(), positions = new Map(), uniqueGames = new Set();
let gamesMeta = null, plies = 0, terminalGames = 0;
for await (const row of records(gamesFile)) {
  if (row.kind === 'meta') {
    if (gamesMeta || games.size) throw new Error('Duplicate or misplaced game metadata');
    gamesMeta = row;
    continue;
  }
  if (!gamesMeta || row.kind !== 'game' || games.has(row.game) || !Array.isArray(row.moves))
    throw new Error('Invalid game record');
  let position = parseFen(row.startFen), history = [positionKey(position)];
  if (row.opening.some((move, index) => row.moves[index] !== move)) throw new Error(`Opening mismatch in game ${row.game}`);
  const states = [];
  for (const [ply, notation] of row.moves.entries()) {
    if (gameResult(position, history)) throw new Error(`Moves after game end in game ${row.game}`);
    const legal = legalMoves(position).map(moveName);
    if (!legal.includes(notation)) throw new Error(`Illegal move ${notation} in game ${row.game}, ply ${ply}`);
    states.push(toFen(position));
    position = playMove(position, notation);
    history.push(positionKey(position));
  }
  if (toFen(position) !== row.finalFen) throw new Error(`Final FEN mismatch in game ${row.game}`);
  const actual = gameResult(position, history) || { winner: null, reason: 'maxPlies', censored: true };
  if (JSON.stringify(actual) !== JSON.stringify(row.result)) throw new Error(`Result mismatch in game ${row.game}`);
  if (!actual.censored) terminalGames++;
  if (!row.moves.length || uniqueGames.has(row.moves.join(' '))) throw new Error(`Empty or duplicate game ${row.game}`);
  uniqueGames.add(row.moves.join(' '));
  games.set(row.game, row);
  positions.set(row.game, states);
  plies += row.moves.length;
}
if (!gamesMeta || (Number.isInteger(gamesMeta.games) && gamesMeta.games !== games.size))
  throw new Error('Game count mismatch');

let teacherMeta = null, teacherPositions = 0;
const seen = new Set();
for await (const row of records(teacherFile)) {
  if (row.kind === 'meta') {
    if (teacherMeta || teacherPositions) throw new Error('Duplicate or misplaced teacher metadata');
    teacherMeta = row;
    continue;
  }
  const game = games.get(row.game), states = positions.get(row.game);
  if (!teacherMeta || row.kind !== 'position' || !game || !states ||
      row.ply < game.opening.length || row.ply >= game.moves.length ||
      row.fen !== states[row.ply] || row.played !== game.moves[row.ply])
    throw new Error(`Teacher/game mismatch at row ${teacherPositions + 1}`);
  const key = `${row.game}:${row.ply}`;
  if (seen.has(key)) throw new Error(`Duplicate teacher position ${key}`);
  seen.add(key);
  teacherPositions++;
}
if (!teacherMeta || teacherMeta.teacherBinarySha256 !== gamesMeta.teacherBinarySha256 ||
    teacherMeta.openingBookHash !== gamesMeta.openingBookHash ||
    teacherMeta.repetitionRule !== gamesMeta.repetitionRule ||
    (Number.isInteger(teacherMeta.positions) && teacherMeta.positions !== teacherPositions) ||
    teacherPositions !== plies - [...games.values()].reduce((sum, game) => sum + game.opening.length, 0))
  throw new Error('Teacher/game metadata or position count mismatch');

console.log(JSON.stringify({ games: games.size, terminalGames, censoredGames: games.size - terminalGames,
  plies, teacherPositions, uniqueGames: uniqueGames.size,
  teacherBinarySha256: gamesMeta.teacherBinarySha256,
  openingBookHash: gamesMeta.openingBookHash }));
