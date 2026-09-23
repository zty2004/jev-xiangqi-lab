import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

const [teacherFile, gameFile, ...excludeFiles] = process.argv.slice(2);
if (!teacherFile || !gameFile) {
  console.error('Usage: node verify-ccpd-teacher.js data/teacher-ccpd.jsonl data/ccpd-computer-games.jsonl [prior-teacher.jsonl ...]');
  process.exit(2);
}
function sha256(filename) { return createHash('sha256').update(readFileSync(filename)).digest('hex'); }
function rows(filename) { return readFileSync(filename, 'utf8').trim().split('\n').map(JSON.parse); }
const games = new Map(rows(gameFile).filter(row => row.kind === 'game').map(row => [row.game, row]));
const teacherRows = rows(teacherFile);
const metadata = teacherRows.shift();
if (metadata?.kind !== 'meta' || metadata.sourceSha256 !== sha256(gameFile) || metadata.positions !== teacherRows.length)
  throw new Error('Teacher metadata or row count mismatch');
if (metadata.excludeSources.length !== excludeFiles.length ||
    metadata.excludeSources.some((source, index) => source.sha256 !== sha256(excludeFiles[index])))
  throw new Error('Excluded teacher source hash mismatch');
const excluded = new Set();
for (const filename of excludeFiles) {
  for (const row of rows(filename)) if (row.kind === 'position') excluded.add(row.fen.split(' ').slice(0, 2).join(' '));
}
const labelled = new Map();
for (const row of teacherRows) {
  if (row.kind !== 'position' || !games.has(row.game) || !Number.isInteger(row.ply) || row.ply < 8)
    throw new Error('Invalid teacher row identity');
  const game = games.get(row.game);
  if (row.ply >= game.moves.length || row.sourceCategory !== game.sourceCategory ||
      row.sourceFile !== game.sourceFile || row.sourceSha256 !== game.sourceSha256 ||
      row.played !== game.moves[row.ply]) throw new Error(`Source game mismatch ${row.game}:${row.ply}`);
  const key = `${row.game}:${row.ply}`;
  if (labelled.has(key)) throw new Error(`Duplicate source position ${key}`);
  labelled.set(key, row);
}
const seen = new Set();
let candidates = 0, bestOutsideCandidates = 0, checked = 0;
for (const game of games.values()) {
  let position = parseFen(game.startFen);
  const history = [positionKey(position)];
  for (let ply = 0; ply < game.moves.length; ply++) {
    const row = labelled.get(`${game.game}:${ply}`);
    if (row) {
      const legal = legalMoves(position).map(moveName);
      const unique = new Set(row.candidates.map(item => item.move));
      if (gameResult(position, history) || row.fen !== toFen(position) ||
          JSON.stringify(row.legal) !== JSON.stringify(legal) ||
          !legal.includes(row.best) || unique.size !== row.candidates.length ||
          row.candidates.some(item => !legal.includes(item.move) || !Number.isFinite(item.score)))
        throw new Error(`Invalid analysis at ${game.game}:${ply}`);
      const boardKey = positionKey(position);
      if (excluded.has(boardKey) || seen.has(boardKey)) throw new Error(`Teacher position overlap at ${game.game}:${ply}`);
      seen.add(boardKey);
      candidates += row.candidates.length;
      bestOutsideCandidates += Number(!unique.has(row.best));
      checked++;
    }
    const move = legalMoves(position).find(item => moveName(item) === game.moves[ply]);
    if (!move) throw new Error(`Illegal source move at ${game.game}:${ply}`);
    position = makeMove(position, move);
    history.push(positionKey(position));
  }
}
if (checked !== teacherRows.length) throw new Error(`Verified ${checked} of ${teacherRows.length} rows`);
console.log(JSON.stringify({ positions: checked, games: new Set(teacherRows.map(row => row.game)).size,
  uniquePositions: seen.size, meanCandidates: candidates / checked, bestOutsideCandidates,
  sourceSha256: metadata.sourceSha256, teacherBinarySha256: metadata.teacherBinarySha256 }));
