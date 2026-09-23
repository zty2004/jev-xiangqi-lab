import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chineseMove } from './src/chinese-notation.js';
import { decodePgn, pgnHeader, pgnMoves } from './src/pgn.js';
import { parseFen, playMove, positionKey, START_FEN } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const input = option('--input');
const sourceCommit = option('--source-commit');
const output = option('--output', 'data/master-opening-book.json');
const reportPath = option('--report', 'reports/master-opening-book.json');
const maxPlies = Number(option('--maxplies', 24));
if (!input || !sourceCommit || !Number.isInteger(maxPlies) || maxPlies < 8 || maxPlies > 40) {
  console.error('Usage: node master-opening-book.js --input /path/to/CCPD/master/by-opening --source-commit SHA [--maxplies 24]');
  process.exit(2);
}
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

async function filesUnder(directory, prefix = '') {
  const files = [];
  for (const item of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    const relative = path.join(prefix, item.name);
    if (item.isDirectory()) files.push(...await filesUnder(path.join(directory, item.name), relative));
    else if (item.name.toLowerCase().endsWith('.pgn')) files.push(relative);
  }
  return files;
}

const files = await filesUnder(input);
const sourceHash = createHash('sha256'), seenGames = new Set(), positions = new Map();
const failures = [], duplicates = [], categories = {};
let imported = 0, recordedPlies = 0, screenHorseGames = 0;
for (const sourceFile of files) {
  const bytes = await readFile(path.join(input, sourceFile));
  sourceHash.update(sourceFile); sourceHash.update(bytes);
  try {
    const pgn = decodePgn(bytes), startFen = pgnHeader(pgn, 'FEN') || START_FEN;
    if (startFen.split(' ').slice(0, 2).join(' ') !== START_FEN.split(' ').slice(0, 2).join(' '))
      throw new Error('nonstandard start position');
    const tokens = pgnMoves(pgn);
    if (tokens.length < 8) throw new Error('fewer than eight moves');
    const identity = digest(`${pgnHeader(pgn, 'Date')}|${pgnHeader(pgn, 'Red')}|${pgnHeader(pgn, 'Black')}|${tokens.join(' ')}`);
    if (seenGames.has(identity)) { duplicates.push(sourceFile); continue; }
    const ecco = pgnHeader(pgn, 'ECCO');
    const line = [], nodes = [];
    let position = parseFen();
    for (const token of tokens.slice(0, maxPlies)) {
      const key = positionKey(position);
      const move = chineseMove(position, token);
      position = playMove(position, move);
      nodes.push(key); line.push(move);
    }
    if (line.length < 8) throw new Error('fewer than eight legal moves');
    seenGames.add(identity);
    const category = sourceFile.split(path.sep)[0];
    categories[category] = (categories[category] || 0) + 1;
    const isScreenHorse = /^C\d\d$/.test(ecco);
    screenHorseGames += Number(isScreenHorse);
    for (let ply = 0; ply < line.length; ply++) {
      const key = nodes[ply];
      if (!positions.has(key)) positions.set(key, new Map());
      const choices = positions.get(key), move = line[ply];
      if (!choices.has(move)) choices.set(move, { move, masterGames: 0, screenHorseGames: 0, sourceExamples: [] });
      const choice = choices.get(move);
      choice.masterGames++;
      choice.screenHorseGames += Number(isScreenHorse);
      if (choice.sourceExamples.length < 2) choice.sourceExamples.push(sourceFile);
    }
    imported++; recordedPlies += line.length;
  } catch (error) { failures.push({ sourceFile, reason: error.message }); }
  if ((imported + failures.length + duplicates.length) % 1000 === 0) console.log(`${imported} imported, ${failures.length} failed`);
}
const entries = [...positions.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([key, choices]) => ({
  position: key,
  moves: [...choices.values()].sort((a, b) => b.masterGames - a.masterGames || a.move.localeCompare(b.move))
}));
const metadata = { kind: 'xiangqi-master-opening-book', source: 'Chinese Chess Practical Dataset (CCPD) master games by opening',
  sourceUrl: 'https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset', sourceCommit,
  license: 'CC BY 4.0', sourceTreeSha256: sourceHash.digest('hex'),
  maxPlies, files: files.length, games: imported, positions: entries.length };
const book = { ...metadata, entries };
await writeFile(output, JSON.stringify(book) + '\n');
await writeFile(reportPath, JSON.stringify({ ...metadata, recordedPlies, screenHorseGames,
  categories, duplicates, failures, bookSha256: digest(JSON.stringify(book) + '\n') }, null, 2) + '\n');
console.log(JSON.stringify({ files: files.length, imported, duplicates: duplicates.length,
  failures: failures.length, positions: entries.length, screenHorseGames }));
