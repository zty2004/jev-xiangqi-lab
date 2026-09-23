import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { formatChineseMove } from './src/chinese-notation.js';
import { loadMasterOpeningBook, masterOpeningCandidates } from './src/opening-book.js';
import { parseFen, playMove, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = option('--input');
const output = option('--output');
if (!input || !output) throw new Error('Usage: node summarize-benchmark.js --input games.jsonl --output report.json');
const raw = readFileSync(input);
const rows = raw.toString('utf8').trim().split('\n').map(line => JSON.parse(line));
const runs = rows.filter(row => row.kind === 'run');
const legacyFailures = rows.filter(row => row.kind === 'game' &&
  /^(baseline|Pikafish) error:/.test(row.result?.reason || ''));
const aborted = [...rows.filter(row => row.kind === 'aborted'),
  ...legacyFailures.map(row => ({ game: row.game, reason: row.result.reason, plies: row.plies,
    moves: row.moves, legacyForfeit: true }))];
const games = rows.filter(row => row.kind === 'game' && !legacyFailures.includes(row));
if (runs.length !== 1 || (!games.length && !aborted.length))
  throw new Error('Expected one run header and at least one game or aborted game');
const bookPath = new URL('./data/master-opening-book.json', import.meta.url);
if (runs[0].settings.masterOpeningBookHash &&
    createHash('sha256').update(readFileSync(bookPath)).digest('hex') !== runs[0].settings.masterOpeningBookHash)
  throw new Error('The current master opening book differs from the benchmark run');
const book = runs[0].settings.masterOpeningBookHash ? loadMasterOpeningBook(bookPath) : null;
const summary = games.map(game => {
  let position = parseFen();
  const movesChinese = [];
  const bookDecisions = [];
  for (const [index, move] of game.moves.entries()) {
    const chinese = formatChineseMove(position, move);
    movesChinese.push(chinese);
    const engineTurn = position.side === game.baselineSide && index >= game.opening.length;
    if (engineTurn && book) {
      const candidates = masterOpeningCandidates(position, book);
      if (candidates.length) bookDecisions.push({ ply: index + 1, selected: chinese,
        withinBook: candidates.some(item => item.move === move), candidateCount: candidates.length });
    }
    position = playMove(position, move);
  }
  if (toFen(position) !== game.finalFen || movesChinese.length !== game.plies)
    throw new Error(`Game ${game.game} does not replay to its recorded final position`);
  return { game: game.game, engineSide: game.baselineSide, result: game.result,
    engineScore: game.baselineScore, plies: game.plies, openingChinese: movesChinese.slice(0, game.opening.length),
    movesChinese, bookDecisions };
});
const report = { source: input, sourceSha256: createHash('sha256').update(raw).digest('hex'),
  settings: runs[0].settings, completedGames: games.length,
  abortedGames: aborted, enginePoints: summary.reduce((sum, game) => sum + game.engineScore, 0), games: summary };
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ completedGames: games.length, enginePoints: report.enginePoints,
  abortedGames: aborted.length,
  bookDecisions: summary.reduce((sum, game) => sum + game.bookDecisions.length, 0), output }));
