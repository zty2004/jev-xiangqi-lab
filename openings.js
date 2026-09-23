import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chineseMove } from './src/chinese-notation.js';
import { parseFen, playMove, START_FEN } from './src/xiangqi.js';

const args = process.argv.slice(2);
function option(name, fallback) { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; }
const input = option('--input');
const output = option('--output', 'data/openings.json');
const maxPlies = Number(option('--maxplies', 24));
if (!input || !Number.isInteger(maxPlies) || maxPlies < 1) {
  console.error('Usage: node openings.js --input /path/to/PGNs [--output data/openings.json] [--maxplies 24]');
  process.exit(2);
}

function header(pgn, key) { return pgn.match(new RegExp(`^\\[${key} "(.*?)"\\]`, 'm'))?.[1] || ''; }
function tokens(pgn) {
  const body = pgn.replace(/^\[.*\]$/gm, '').replace(/\{[^}]*\}/gs, '').replace(/\([^)]*\)/gs, '')
    .replace(/\d+\.(?:\.\.)?/g, ' ');
  return body.split(/\s+/).filter(value => /^[\p{Script=Han}０-９0-9]{4}$/u.test(value));
}

async function main() {
  const files = (await readdir(input)).filter(name => name.toLowerCase().endsWith('.pgn')).sort();
  const entries = [], failures = [], seen = new Set(), hash = createHash('sha256');
  for (const file of files) {
    const bytes = await readFile(path.join(input, file));
    hash.update(file); hash.update(bytes);
    let pgn;
    try { pgn = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
    catch { pgn = new TextDecoder('big5', { fatal: true }).decode(bytes); }
    try {
      const fen = header(pgn, 'FEN') || START_FEN;
      let position = parseFen(fen);
      if (fen.split(' ')[0] !== START_FEN.split(' ')[0]) throw new Error('nonstandard start position');
      const moves = [];
      for (const token of tokens(pgn).slice(0, maxPlies)) {
        const move = chineseMove(position, token);
        position = playMove(position, move);
        moves.push(move);
      }
      if (moves.length < 2) throw new Error('fewer than two moves');
      const key = moves.join(' ');
      if (seen.has(key)) continue;
      seen.add(key);
      entries.push({ name: header(pgn, 'Event'), ecco: header(pgn, 'ECCO'), moves, sourceFile: file });
    } catch (error) { failures.push({ file, reason: error.message }); }
  }
  const result = { kind: 'xiangqi-opening-book', source: 'Chinese Chess Practical Dataset (CCPD)',
    sourceUrl: 'https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset', license: 'CC BY 4.0',
    sourceHash: hash.digest('hex'), maxPlies, files: files.length, entries, failures };
  await writeFile(output, JSON.stringify(result, null, 2) + '\n');
  console.log(`Imported ${entries.length}/${files.length} PGNs (${failures.length} failed, ${files.length - entries.length - failures.length} duplicate). Saved ${output}`);
  if (failures.length) console.log(`First failures: ${JSON.stringify(failures.slice(0, 5))}`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
