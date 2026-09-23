import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { gameResult, parseFen, playMove, positionKey, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
}
const gamesPath = option('--games', 'data/ai-games-160.jsonl');
const teacherPath = option('--teacher', 'data/teacher-games-160.jsonl');
const outputPath = option('--output', 'data/history-labels-160.jsonl');
const reportPath = option('--report', 'reports/history-labels-160.json');

function readJsonl(filename) {
  const content = readFileSync(filename);
  return { rows: content.toString('utf8').trim().split('\n').map(JSON.parse),
    sha256: createHash('sha256').update(content).digest('hex') };
}

const gamesSource = readJsonl(gamesPath), teacherSource = readJsonl(teacherPath);
const games = new Map();
for (const game of gamesSource.rows.filter(row => row.kind === 'game')) {
  if (games.has(game.game)) throw new Error(`Duplicate game ${game.game}`);
  let position = parseFen(game.startFen);
  const states = [position];
  for (const move of game.moves) {
    position = playMove(position, move);
    states.push(position);
  }
  if (toFen(position) !== game.finalFen) throw new Error(`Final FEN mismatch in game ${game.game}`);
  const history = states.map(positionKey);
  const result = gameResult(position, history);
  if (game.result.censored) {
    if (result) throw new Error(`Censored game ${game.game} actually has a terminal result`);
  } else if (!result || result.winner !== game.result.winner) {
    throw new Error(`Result mismatch in game ${game.game}`);
  }
  games.set(game.game, { game, states, history });
}

const output = [], keys = new Set();
const report = { schema: 'history-labels-v1', gamesPath, teacherPath,
  gamesSha256: gamesSource.sha256, teacherSha256: teacherSource.sha256,
  games: games.size, positions: 0, terminalPositions: 0, censoredPositions: 0,
  positionsWithTwoPriorBoards: 0, repeatedPositions: 0, maxRepetitionCount: 0,
  redWinPositions: 0, blackWinPositions: 0, drawPositions: 0 };
output.push(JSON.stringify({ kind: 'meta', schema: report.schema,
  gamesSha256: report.gamesSha256, teacherSha256: report.teacherSha256 }));
for (const row of teacherSource.rows.filter(item => item.kind === 'position')) {
  const entry = games.get(row.game);
  if (!entry) throw new Error(`Missing game ${row.game}`);
  const { game, states, history } = entry;
  if (!Number.isInteger(row.ply) || row.ply < 0 || row.ply >= game.moves.length) throw new Error(`Invalid ply in game ${row.game}`);
  const id = `${row.game}:${row.ply}`;
  if (keys.has(id)) throw new Error(`Duplicate teacher position ${id}`);
  keys.add(id);
  if (toFen(states[row.ply]) !== row.fen || game.moves[row.ply] !== row.played) throw new Error(`Teacher row differs from game ${id}`);
  const previous = history.slice(Math.max(0, row.ply - 2), row.ply);
  const repetitionCount = history.slice(0, row.ply + 1).filter(key => key === history[row.ply]).length;
  const winner = game.result.censored ? null : game.result.winner === null ? 'draw' : game.result.winner;
  output.push(JSON.stringify({ kind: 'history-label', game: row.game, ply: row.ply,
    previous, repetitionCount, winner }));
  report.positions++;
  if (winner === null) report.censoredPositions++;
  else {
    report.terminalPositions++;
    if (winner === 'draw') report.drawPositions++;
    else report[`${winner}WinPositions`]++;
  }
  if (previous.length === 2) report.positionsWithTwoPriorBoards++;
  if (repetitionCount > 1) report.repeatedPositions++;
  report.maxRepetitionCount = Math.max(report.maxRepetitionCount, repetitionCount);
}
if (!report.games || !report.positions || report.positions !== teacherSource.rows.filter(item => item.kind === 'position').length)
  throw new Error('Missing game or position records');
writeFileSync(outputPath, `${output.join('\n')}\n`);
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report));
