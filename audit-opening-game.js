import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { formatChineseMove } from './src/chinese-notation.js';
import { loadMasterOpeningBook, masterOpeningCandidates } from './src/opening-book.js';
import { PikafishTeacher } from './src/pikafish-teacher.js';
import { parseFen, playMove, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = option('--input');
const output = option('--output');
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const gameNumber = Number(option('--game', 1));
const moveTime = Number(option('--movetime', 5000));
if (!input || !output || !binary || !Number.isInteger(gameNumber) || gameNumber < 1 ||
    !Number.isInteger(moveTime) || moveTime < 100)
  throw new Error('Usage: node audit-opening-game.js --input games.jsonl --game N --pikafish /path/to/Pikafish --movetime 5000 --output report.json');

const raw = readFileSync(input);
const rows = raw.toString('utf8').trim().split('\n').map(line => JSON.parse(line));
const run = rows.find(row => row.kind === 'run');
const game = rows.find(row => row.kind === 'game' && row.game === gameNumber);
if (!run || !game || /^(baseline|Pikafish) error:/.test(game.result?.reason || ''))
  throw new Error('Selected game is missing or did not finish normally');
const bookPath = new URL('./data/master-opening-book.json', import.meta.url);
const bookRaw = readFileSync(bookPath);
if (createHash('sha256').update(bookRaw).digest('hex') !== run.settings.masterOpeningBookHash)
  throw new Error('Master opening book differs from the played game');
const book = loadMasterOpeningBook(bookPath);
const teacher = new PikafishTeacher(binary);
const positions = [];
let position = parseFen();
try {
  await teacher.ready();
  for (const [index, move] of game.moves.entries()) {
    if (index < book.maxPlies && index >= game.opening.length && position.side === game.baselineSide) {
      const candidates = masterOpeningCandidates(position, book);
      if (candidates.length) {
        const analysis = await teacher.analyse(toFen(position), moveTime, game.moves.slice(0, index));
        const selected = analysis.candidates.find(item => item.move === move);
        const first = analysis.candidates[0];
        positions.push({ ply: index + 1, fen: toFen(position), played: move,
          playedChinese: formatChineseMove(position, move), bookCandidates: candidates.map(item => ({
            move: item.move, chinese: formatChineseMove(position, item.move),
            masterGames: item.masterGames, screenHorseGames: item.screenHorseGames })),
          teacherBest: analysis.best, teacherBestChinese: formatChineseMove(position, analysis.best),
          teacherDepth: analysis.depth, selectedRank: selected?.rank || null,
          scoreGapCp: first?.scoreType === 'cp' && selected?.scoreType === 'cp' ? first.score - selected.score : null,
          teacherCandidates: analysis.candidates.map(item => ({ ...item,
            chinese: formatChineseMove(position, item.move) })) });
      }
    }
    position = playMove(position, move);
  }
} finally { teacher.close(); }
if (toFen(position) !== game.finalFen) throw new Error('Game replay did not match its recorded final position');
const report = { source: input, sourceSha256: createHash('sha256').update(raw).digest('hex'),
  game: gameNumber, engineSide: game.baselineSide, moveTimeMs: moveTime,
  teacherBinarySha256: createHash('sha256').update(readFileSync(binary)).digest('hex'),
  bookSha256: run.settings.masterOpeningBookHash, positions,
  limitations: 'Finite-time MultiPV evaluations diagnose candidate book moves; they do not prove the best opening or playing strength.' };
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ auditedBookMoves: positions.length,
  teacherBestMatches: positions.filter(item => item.played === item.teacherBest).length, output }));
