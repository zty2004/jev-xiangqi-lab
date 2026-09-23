import readline from 'node:readline';
import { chooseMove } from './src/engine.js';
import { START_FEN, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';
import { LocalChoice } from './src/local-choice.js';

let position = parseFen();
let history = [positionKey(position)];
const localChoice = process.env.CHOICE_MODEL ? new LocalChoice(process.env.CHOICE_MODEL) : null;
const io = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

function setPosition(command) {
  const tokens = command.trim().split(/\s+/);
  const moveAt = tokens.indexOf('moves');
  const base = tokens[1] === 'startpos' ? START_FEN : tokens[1] === 'fen' ? tokens.slice(2, moveAt < 0 ? undefined : moveAt).join(' ') : null;
  if (!base) return;
  position = parseFen(base);
  history = [positionKey(position)];
  if (moveAt < 0) return;
  for (const notation of tokens.slice(moveAt + 1)) {
    const move = legalMoves(position).find(item => moveName(item) === notation);
    if (!move) throw new Error(`Illegal position move: ${notation}`);
    position = makeMove(position, move);
    history.push(positionKey(position));
  }
}

function perft(pos, depth) {
  if (depth <= 0) return 1;
  return legalMoves(pos).reduce((total, move) => total + perft(makeMove(pos, move), depth - 1), 0);
}

io.on('line', async line => {
  try {
    if (line === 'uci') {
      console.log(`id name Jev Xiangqi Lab ${localChoice ? 'local choice' : 'baseline'}`);
      console.log('id author Codex');
      console.log('option name Move Overhead type spin default 100 min 0 max 1000');
      console.log('uciok');
    } else if (line === 'isready') { await localChoice?.ready(); console.log('readyok'); }
    else if (line === 'ucinewgame') { position = parseFen(); history = [positionKey(position)]; }
    else if (line.startsWith('position ')) setPosition(line);
    else if (line.startsWith('go ')) {
      const tokens = line.split(/\s+/);
      const timeIndex = tokens.indexOf('movetime'), depthIndex = tokens.indexOf('depth'), perftIndex = tokens.indexOf('perft');
      if (perftIndex >= 0) console.log(`Nodes searched: ${perft(position, Math.min(6, Number(tokens[perftIndex + 1]) || 1))}`);
      else {
        const requestedMs = timeIndex >= 0 ? Number(tokens[timeIndex + 1]) : 5000;
        const started = performance.now();
        const ranking = localChoice ? await localChoice.rank(toFen(position), legalMoves(position).map(moveName), Math.max(100, requestedMs)) : null;
        const priors = ranking ? new Map(ranking.choices.map(item => [item.move, item.probability])) : null;
        const result = chooseMove(position, { timeMs: Math.max(10, requestedMs - (performance.now() - started)),
          maxDepth: depthIndex >= 0 ? Number(tokens[depthIndex + 1]) : 12, history: history.slice(0, -1), priors });
        const selected = result.move ? moveName(result.move) : '0000';
        if (ranking) console.log(`info string local choice ranked ${ranking.choices.length} legal moves, selected ${selected} probability ${(priors.get(selected) || 0).toFixed(4)}`);
        console.log(`info depth ${result.depth} score cp ${result.score} nodes ${result.nodes} time ${result.timeMs} pv ${result.pv.join(' ')}`);
        console.log(`bestmove ${selected}`);
      }
    } else if (line === 'd') console.log(`Fen: ${toFen(position)}`);
    else if (line === 'quit') { localChoice?.close(); io.close(); }
  } catch (error) {
    console.log(`info string error ${error.message}`);
    if (line.startsWith('go ')) console.log('bestmove 0000');
  }
});
