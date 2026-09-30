import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chineseMove } from './src/chinese-notation.js';
import { decodePgn, pgnHeader, pgnMoves } from './src/pgn.js';
import { gameResult, parseFen, playMove, positionKey, toFen, START_FEN } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = option('--input');
const sourceCommit = option('--source-commit');
const scope = option('--scope', 'computer');
const output = option('--output', 'data/ccpd-computer-games.jsonl');
const reportPath = option('--report', 'reports/ccpd-computer-import.json');
const comparisonFiles = args.flatMap((value, index) => value === '--compare-teacher' ? [args[index + 1]] : []);
if (!input || !sourceCommit || !['computer', 'all'].includes(scope)) {
  console.error('Usage: node ccpd-games.js --input /path/to/CCPD/Dataset/對局/電腦對局 --source-commit SHA [--scope computer|all] [--output data/ccpd-computer-games.jsonl]');
  process.exit(2);
}

const resultSide = value => value === '1-0' ? 'red' : value === '0-1' ? 'black' : value === '1/2-1/2' ? null : undefined;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
async function pgnFiles(directory, prefix = '') {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...await pgnFiles(path.join(directory, entry.name), relative));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith('.pgn')) files.push(relative);
  }
  return files;
}
function sourceCategory(sourceFile) {
  if (sourceFile.includes('電腦對局競賽/')) return '電腦對局競賽';
  if (sourceFile.includes('人機賽/')) return '人機賽';
  if (sourceFile.includes('大師對局/')) return '大師對局';
  return sourceFile.split('/')[0];
}
const files = scope === 'computer'
  ? (await Promise.all(['電腦對局競賽', '人機賽'].map(async category =>
      (await pgnFiles(path.join(input, category))).map(filename => `${category}/${filename}`)))).flat()
  : await pgnFiles(input);
const categories = [...new Set(files.map(sourceCategory))].sort();
const sourceHash = createHash('sha256'), games = [], failures = [], duplicates = [];
const seen = new Map(), counts = {};
const positionKeys = new Set(), phasePlies = { opening: 0, middle: 0, late: 0 };
for (const category of categories) counts[category] = { files: 0, imported: 0, failed: 0, duplicates: 0, plies: 0 };
for (const sourceFile of files) {
    const category = sourceCategory(sourceFile), bytes = await readFile(path.join(input, sourceFile));
    counts[category].files++;
    sourceHash.update(sourceFile); sourceHash.update(bytes);
    try {
      const pgn = decodePgn(bytes);
      const declared = pgnHeader(pgn, 'Result');
      const bodyResult = pgn.match(/(1-0|0-1|1\/2-1\/2|\*)\s*$/)?.[1];
      if (!['1-0', '0-1', '1/2-1/2', '*'].includes(declared) ||
          (bodyResult && bodyResult !== declared) || (declared !== '*' && bodyResult !== declared))
        throw new Error(`missing or inconsistent PGN result: header=${declared}, body=${bodyResult || ''}`);
      const startFen = pgnHeader(pgn, 'FEN') || START_FEN;
      let position = parseFen(startFen);
      const history = [positionKey(position)], moves = [];
      for (const notation of pgnMoves(pgn, true)) {
        const move = chineseMove(position, notation);
        position = playMove(position, move);
        moves.push(move);
        history.push(positionKey(position));
      }
      if (moves.length < 2) throw new Error('fewer than two legal moves');
      const key = `${startFen.split(' ').slice(0, 2).join(' ')} ${moves.join(' ')}`;
      if (seen.has(key)) {
        duplicates.push({ file: sourceFile, duplicateOf: seen.get(key) });
        counts[category].duplicates++;
        continue;
      }
      seen.set(key, sourceFile);
      for (let index = 1; index < history.length; index++) {
        positionKeys.add(history[index]);
        phasePlies[index <= 16 ? 'opening' : index <= 60 ? 'middle' : 'late']++;
      }
      const winner = resultSide(declared), censored = declared === '*';
      const adjudicated = gameResult(position, history);
      games.push({ kind: 'game', game: games.length, startFen, opening: moves.slice(0, 8), moves,
        finalFen: toFen(position), result: censored ? { winner: null, reason: 'CCPD PGN result unknown', censored: true }
          : { winner, reason: `CCPD PGN result ${declared}` },
        rulesTerminal: adjudicated, sourceCategory: category, sourceFile,
        sourceSha256: sha256(bytes), red: pgnHeader(pgn, 'Red'), black: pgnHeader(pgn, 'Black'),
        event: pgnHeader(pgn, 'Event'), ecco: pgnHeader(pgn, 'ECCO') });
      counts[category].imported++; counts[category].plies += moves.length;
    } catch (error) {
      failures.push({ file: sourceFile, reason: error.message });
      counts[category].failed++;
    }
}
const metadata = { kind: 'meta', format: 'iccs-game-v1', source: 'Chinese Chess Practical Dataset (CCPD)',
  sourceUrl: 'https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset', sourceCommit,
  license: 'CC BY 4.0', scope, sourceTreeSha256: sourceHash.digest('hex'), categories,
  files: Object.values(counts).reduce((n, x) => n + x.files, 0) };
const comparisonKeys = new Set(), comparisonTeacherSha256 = [];
for (const filename of comparisonFiles) {
  const bytes = await readFile(filename);
  comparisonTeacherSha256.push({ file: filename, sha256: sha256(bytes) });
  for (const line of bytes.toString('utf8').trim().split('\n')) {
    const row = JSON.parse(line);
    if (row.kind === 'position') comparisonKeys.add(row.fen.split(' ').slice(0, 2).join(' '));
  }
}
const overlap = [...positionKeys].filter(key => comparisonKeys.has(key)).length;
const report = { ...metadata, kind: 'import-report', counts, importedGames: games.length,
  importedPlies: games.reduce((n, game) => n + game.moves.length, 0),
  uniquePositions: positionKeys.size, phasePlies,
  comparisonTeacherSha256, crossTeacherPositionOverlap: overlap,
  novelPositionsAgainstComparedTeachers: positionKeys.size - overlap,
  resultCounts: { red: games.filter(game => game.result.winner === 'red').length,
    black: games.filter(game => game.result.winner === 'black').length,
    draw: games.filter(game => !game.result.censored && game.result.winner === null).length,
    censored: games.filter(game => game.result.censored).length },
  rulesTerminalDisagreements: games.filter(game => !game.result.censored && game.rulesTerminal && game.rulesTerminal.winner !== game.result.winner).length,
  duplicates, failures };
await writeFile(output, `${[metadata, ...games].map(row => JSON.stringify(row)).join('\n')}\n`);
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ files: metadata.files, imported: games.length, failures: failures.length,
  duplicates: duplicates.length, plies: report.importedPlies, resultCounts: report.resultCounts }));
