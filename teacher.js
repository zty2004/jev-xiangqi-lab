import { createHash } from 'node:crypto';
import { createWriteStream, readFileSync } from 'node:fs';
import path from 'node:path';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';
import { loadOpeningLines } from './src/opening-book.js';
import { PikafishTeacher } from './src/pikafish-teacher.js';

const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const gameLimit = args.includes('--games') ? Number(option('--games')) : null;
const positions = gameLimit === null ? Number(option('--positions', 1000)) : null;
const moveTime = Number(option('--movetime', 200));
const maxPlies = Number(option('--maxplies', 120));
const seed = Number(option('--seed', 20260923));
const output = path.resolve(option('--output', 'teacher-data.jsonl'));
const gamesOutput = args.includes('--games-output') ? path.resolve(option('--games-output')) : null;
const openingBook = option('--openings');
const openingPlies = Number(option('--opening-plies', 8));
if (!binary || (gameLimit === null && (!Number.isInteger(positions) || positions < 1)) ||
    (gameLimit !== null && (!Number.isInteger(gameLimit) || gameLimit < 1 || !gamesOutput || args.includes('--positions'))) ||
    (gameLimit === null && gamesOutput) ||
    !Number.isInteger(moveTime) || moveTime < 20) {
  console.error('Usage: node teacher.js --pikafish /absolute/path/to/pikafish [--positions 1000 | --games 100 --games-output data/games.jsonl] [--movetime 200] [--output data/teacher.jsonl]');
  process.exit(2);
}

function randomGenerator(value) {
  let state = value >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const random = randomGenerator(seed);

const openings = openingBook ? loadOpeningLines(openingBook, openingPlies) :
  [[], ['b2e2', 'b7e7'], ['h2e2', 'h7e7'], ['b0c2', 'b9c7'], ['h0g2', 'h9g7']];
if (openingBook) {
  const openingRandom = randomGenerator(seed ^ 0x5eeda11);
  for (let i = openings.length - 1; i > 0; i--) {
    const j = Math.floor(openingRandom() * (i + 1));
    [openings[i], openings[j]] = [openings[j], openings[i]];
  }
}
function advance(position, notation) {
  const move = legalMoves(position).find(item => moveName(item) === notation);
  if (!move) throw new Error(`Illegal teacher move ${notation} in ${toFen(position)}`);
  return makeMove(position, move);
}

async function main() {
  const teacher = new PikafishTeacher(binary), writer = createWriteStream(output, { flags: 'w' });
  const gameWriter = gamesOutput ? createWriteStream(gamesOutput, { flags: 'w' }) : null;
  try {
    await teacher.ready();
    const teacherBinarySha256 = createHash('sha256').update(readFileSync(binary)).digest('hex');
    const metadata = { model: 'Pikafish', positions, games: gameLimit, moveTime, seed,
      repetitionRule: 'single-side-perpetual-check-loss-v1',
      teacherBinarySha256,
      openingBook: openingBook || null, openingPlies: openingBook ? openingPlies : null, openingLines: openings.length,
      openingBookHash: openingBook ? createHash('sha256').update(readFileSync(openingBook)).digest('hex') : null,
      generatedAt: new Date().toISOString() };
    writer.write(JSON.stringify({ kind: 'meta', ...metadata }) + '\n');
    gameWriter?.write(JSON.stringify({ kind: 'meta', ...metadata, format: 'iccs-game-v1' }) + '\n');
    let written = 0;
    for (let game = 0; gameLimit === null ? written < positions : game < gameLimit; game++) {
      teacher.send('ucinewgame');
      let position = parseFen(), history = [positionKey(position)], ply = 0;
      const opening = openings[game % openings.length], moves = [...opening];
      for (const notation of opening) { position = advance(position, notation); history.push(positionKey(position)); ply++; }
      while ((gameLimit !== null || written < positions) && ply < maxPlies && !gameResult(position, history)) {
        const legal = legalMoves(position).map(moveName), fen = toFen(position);
        const analysis = await teacher.analyse(fen, moveTime, moves);
        if (!legal.includes(analysis.best)) throw new Error(`Pikafish returned illegal move ${analysis.best} in ${fen}`);
        const candidates = analysis.candidates.filter(item => legal.includes(item.move));
        const varied = random() < 0.2 && candidates.length >= 2;
        const played = varied ? candidates[Math.min(candidates.length - 1, Math.floor(random() * 3))].move : analysis.best;
        writer.write(JSON.stringify({ kind: 'position', game, ply, fen, legal, best: analysis.best, played, candidates, depth: analysis.depth }) + '\n');
        written++;
        moves.push(played);
        position = advance(position, played); history.push(positionKey(position)); ply++;
        if (written % 100 === 0) console.log(`${written} positions, ${game + 1} games`);
      }
      const result = gameResult(position, history);
      gameWriter?.write(JSON.stringify({ kind: 'game', game, startFen: toFen(parseFen()), opening,
        moves, finalFen: toFen(position), result: result || { winner: null, reason: 'maxPlies', censored: true } }) + '\n');
    }
    console.log(`Saved ${written} positions to ${output}${gamesOutput ? ` and games to ${gamesOutput}` : ''}`);
  } finally { teacher.close(); writer.end(); gameWriter?.end(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
