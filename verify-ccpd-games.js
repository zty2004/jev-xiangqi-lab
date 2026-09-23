import { createReadStream } from 'node:fs';
import readline from 'node:readline';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

const filename = process.argv[2];
if (!filename) {
  console.error('Usage: node verify-ccpd-games.js data/ccpd-computer-games.jsonl');
  process.exit(2);
}

let metadata = null, games = 0, plies = 0, positions = new Set();
const identifiers = new Set(), traces = new Set(), results = { red: 0, black: 0, draw: 0, censored: 0 };
for await (const line of readline.createInterface({ input: createReadStream(filename), crlfDelay: Infinity })) {
  const row = JSON.parse(line);
  if (row.kind === 'meta') {
    if (metadata || games || row.license !== 'CC BY 4.0' || row.format !== 'iccs-game-v1') throw new Error('Invalid dataset metadata');
    metadata = row;
    continue;
  }
  if (!metadata || row.kind !== 'game' || identifiers.has(row.game) || !Array.isArray(row.moves) || row.moves.length < 2)
    throw new Error(`Invalid game record ${row.game}`);
  identifiers.add(row.game);
  const trace = `${row.startFen.split(' ').slice(0, 2).join(' ')} ${row.moves.join(' ')}`;
  if (traces.has(trace)) throw new Error(`Duplicate move trace ${row.game}`);
  traces.add(trace);
  if (!['電腦對局競賽', '人機賽'].includes(row.sourceCategory) ||
      !row.sourceFile.startsWith(`${row.sourceCategory}/`) || !/^[a-f0-9]{64}$/.test(row.sourceSha256))
    throw new Error(`Invalid source attribution in game ${row.game}`);
  if (row.opening.join(' ') !== row.moves.slice(0, 8).join(' ')) throw new Error(`Invalid opening in game ${row.game}`);
  let position = parseFen(row.startFen);
  const history = [positionKey(position)];
  for (const notation of row.moves) {
    const move = legalMoves(position).find(item => moveName(item) === notation);
    if (!move) throw new Error(`Illegal move ${notation} in game ${row.game}`);
    position = makeMove(position, move);
    history.push(positionKey(position));
    positions.add(positionKey(position));
    plies++;
  }
  if (toFen(position) !== row.finalFen || JSON.stringify(gameResult(position, history)) !== JSON.stringify(row.rulesTerminal))
    throw new Error(`Final position mismatch in game ${row.game}`);
  if (row.result.censored) {
    if (row.result.winner !== null) throw new Error(`Invalid censored result in game ${row.game}`);
    results.censored++;
  } else if (row.result.winner === null) results.draw++;
  else if (row.result.winner === 'red' || row.result.winner === 'black') results[row.result.winner]++;
  else throw new Error(`Invalid result in game ${row.game}`);
  games++;
}
if (!metadata || games < 1) throw new Error('No games found');
console.log(JSON.stringify({ games, plies, uniquePositions: positions.size, results,
  sourceTreeSha256: metadata.sourceTreeSha256, sourceCommit: metadata.sourceCommit }));
