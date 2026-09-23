import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { formatChineseMove } from './src/chinese-notation.js';
import { loadMasterOpeningBook, masterOpeningCandidates } from './src/opening-book.js';
import { PikafishTeacher } from './src/pikafish-teacher.js';
import { parseFen, playMove, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const binary = option('--pikafish', process.env.PIKAFISH_PATH);
const bookFile = option('--book', 'data/master-opening-book.json');
const output = option('--output', 'reports/opening-response-check.json');
const moveTime = Number(option('--movetime', 5000));
if (!binary || !Number.isInteger(moveTime) || moveTime < 100) {
  console.error('Usage: node check-opening-responses.js --pikafish /path/to/Pikafish [--movetime 5000]');
  process.exit(2);
}
const sha256 = filename => createHash('sha256').update(readFileSync(filename)).digest('hex');
const book = loadMasterOpeningBook(bookFile), teacher = new PikafishTeacher(binary);
const openings = [['b2e2'], ['b2e2', 'b9c7', 'b0c2', 'a9b9', 'a0b0']];
const positions = [];
try {
  await teacher.ready();
  for (const moves of openings) {
    teacher.send('ucinewgame');
    let position = parseFen();
    for (const move of moves) position = playMove(position, move);
    const fen = toFen(position), bookMoves = masterOpeningCandidates(position, book);
    const analysis = await teacher.analyse(fen, moveTime, moves);
    positions.push({ openingMoves: moves, fen, teacherBest: analysis.best,
      teacherBestChinese: formatChineseMove(position, analysis.best), teacherDepth: analysis.depth,
      teacherCandidates: analysis.candidates.map(item => ({ ...item,
        chinese: formatChineseMove(position, item.move) })),
      bookMoves: bookMoves.map(item => ({ move: item.move, chinese: formatChineseMove(position, item.move),
        masterGames: item.masterGames, screenHorseGames: item.screenHorseGames })) });
  }
} finally { teacher.close(); }
const report = { purpose: 'Independent finite-time check of key central-cannon book responses; Pikafish is not used during actual play',
  masterOpeningBookSha256: sha256(bookFile), teacherBinarySha256: sha256(binary), moveTimeMs: moveTime,
  positions, limitations: 'Two queried positions and five-second MultiPV scores do not prove optimality or match strength.' };
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ positions: positions.length, bookBestMatchesTeacher: positions.map(item =>
  item.bookMoves.some(candidate => candidate.move === item.teacherBest)), output }));
