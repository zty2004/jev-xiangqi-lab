import { createHash } from 'node:crypto';
import { createWriteStream, readFileSync } from 'node:fs';
import { once } from 'node:events';
import path from 'node:path';
import { chooseMove } from './src/engine.js';
import { formatChineseLine } from './src/chinese-notation.js';
import { LocalChoice } from './src/local-choice.js';
import { currentChoiceModel } from './src/model-selection.js';
import { loadNnueModel, NnueEvaluator } from './src/nnue-evaluator.js';
import { loadOpeningLines } from './src/opening-book.js';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

const args = process.argv.slice(2);
const option = (name, fallback) => { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1]; };
const games = Number(option('--games', 8));
const moveTime = Number(option('--movetime', 100));
const maxPlies = Number(option('--maxplies', 120));
const openingPlies = Number(option('--opening-plies', 8));
const openingsFile = option('--openings', 'data/openings.json');
const candidateFile = option('--candidate');
const output = path.resolve(option('--output', 'evaluator-match.jsonl'));
if (!candidateFile || !Number.isInteger(games) || games < 2 || games % 2 || !Number.isInteger(moveTime) || moveTime < 10)
  throw new Error('Usage: node evaluator-match.js --candidate models/value.json [--games 8 --movetime 100]');

const sha256 = filename => createHash('sha256').update(readFileSync(filename)).digest('hex');
const advance = (position, notation) => {
  const move = legalMoves(position).find(item => moveName(item) === notation);
  if (!move) throw new Error(`Illegal match move ${notation}`);
  return makeMove(position, move);
};

async function main() {
  const choicePath = currentChoiceModel();
  const choice = new LocalChoice(choicePath);
  const candidate = new NnueEvaluator(loadNnueModel(candidateFile));
  const openings = loadOpeningLines(openingsFile, openingPlies);
  const writer = createWriteStream(output, { flags: 'w' });
  writer.write(JSON.stringify({ kind: 'meta', schema: 'value-evaluator-match-v1', games, moveTime, maxPlies,
    openingPlies, openingsFile, openingsSha256: sha256(openingsFile), choiceModel: path.basename(choicePath),
    choiceModelSha256: sha256(choicePath), candidate: path.basename(candidateFile),
    candidateSha256: sha256(candidateFile), generatedAt: new Date().toISOString() }) + '\n');
  let candidateScore = 0, completed = 0;
  try {
    await choice.ready();
    for (let game = 0; game < games; game++) {
      const candidateSide = game % 2 === 0 ? 'red' : 'black';
      const opening = openings[Math.floor(game / 2) % openings.length];
      let position = parseFen(), history = [positionKey(position)], moves = [];
      for (const notation of opening) {
        moves.push(notation); position = advance(position, notation); history.push(positionKey(position));
      }
      const search = [];
      while (moves.length < maxPlies && !gameResult(position, history)) {
        const legal = legalMoves(position).map(moveName);
        const ranking = await choice.rank(toFen(position), legal, Math.max(10000, moveTime * 10), history);
        const priors = new Map(ranking.choices.map(item => [item.move, item.probability]));
        const result = chooseMove(position, { timeMs: moveTime, history: history.slice(0, -1), priors,
          evaluator: position.side === candidateSide ? candidate : null });
        const notation = moveName(result.move);
        search.push({ ply: moves.length, side: position.side, evaluator: position.side === candidateSide ? 'candidate' : 'classical',
          move: notation, phase: result.phase, depth: result.depth, nodes: result.nodes, score: result.score });
        moves.push(notation); position = advance(position, notation); history.push(positionKey(position));
      }
      const result = gameResult(position, history) || { winner: null, reason: 'maxPlies', censored: true };
      if (!result.censored) {
        completed++;
        candidateScore += result.winner === null ? 0.5 : Number(result.winner === candidateSide);
      }
      writer.write(JSON.stringify({ kind: 'game', game, candidateSide, opening, moves,
        chineseMoves: formatChineseLine(parseFen(), moves), search, finalFen: toFen(position), result }) + '\n');
      console.log(`game=${game + 1}/${games} candidate=${candidateSide} plies=${moves.length} result=${result.censored ? 'censored' : result.winner || 'draw'} score=${candidateScore}/${completed}`);
    }
  } finally {
    choice.close(); writer.end(); await once(writer, 'finish');
  }
  console.log(JSON.stringify({ games, completed, candidateScore, scoreRate: completed ? candidateScore / completed : null, output }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
