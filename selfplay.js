import { createHash } from 'node:crypto';
import { createWriteStream, readFileSync } from 'node:fs';
import { once } from 'node:events';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chooseMove } from './src/engine.js';
import { formatChineseLine } from './src/chinese-notation.js';
import { LocalChoice } from './src/local-choice.js';
import { currentChoiceModel, currentNnueModel } from './src/model-selection.js';
import { loadNnueModel, NnueEvaluator } from './src/nnue-evaluator.js';
import { loadOpeningLines } from './src/opening-book.js';
import { gameResult, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

function options(argv) {
  const value = (name, fallback) => { const index = argv.indexOf(name); return index < 0 ? fallback : argv[index + 1]; };
  const depth = argv.includes('--depth') ? Number(value('--depth')) : null;
  return { games: Number(value('--games', 2)), moveTime: Number(value('--movetime', 100)),
    depth,
    maxPlies: Number(value('--maxplies', 160)), seed: Number(value('--seed', 20260924)),
    openingPlies: Number(value('--opening-plies', 8)), explorePlies: Number(value('--explore-plies', 40)),
    temperature: Number(value('--temperature', 80)), openings: value('--openings', 'data/openings.json'),
    output: path.resolve(value('--output', 'data/selfplay-games.jsonl')),
    positionsOutput: path.resolve(value('--positions-output', 'data/selfplay-positions.jsonl')) };
}

export function randomGenerator(value) {
  let state = value >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

export function scorePolicy(candidates, temperature = 80) {
  if (!candidates.length) return [];
  if (!(temperature > 0)) return candidates.map((item, index) => ({ move: item.move, probability: Number(index === 0) }));
  const best = Math.max(...candidates.map(item => item.score));
  const weights = candidates.map(item => Math.exp(Math.max(-20, (item.score - best) / temperature)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return candidates.map((item, index) => ({ move: item.move, probability: weights[index] / total }));
}

export function samplePolicy(policy, random) {
  let cursor = random();
  for (const item of policy) {
    cursor -= item.probability;
    if (cursor <= 0) return item.move;
  }
  return policy.at(-1)?.move;
}

function sha256(filename) { return createHash('sha256').update(readFileSync(filename)).digest('hex'); }

function advance(position, notation) {
  const move = legalMoves(position).find(item => moveName(item) === notation);
  if (!move) throw new Error(`Illegal self-play move ${notation} in ${toFen(position)}`);
  return makeMove(position, move);
}

async function close(writer) {
  writer.end();
  await once(writer, 'finish');
}

export async function generateSelfPlay(config) {
  if (!Number.isInteger(config.games) || config.games < 1 || !Number.isInteger(config.moveTime) || config.moveTime < 10 ||
      !Number.isInteger(config.maxPlies) || config.maxPlies < 1 || !Number.isInteger(config.openingPlies) || config.openingPlies < 2 ||
      !Number.isInteger(config.explorePlies) || config.explorePlies < 0 || !(config.temperature > 0) ||
      (config.depth !== null && (!Number.isInteger(config.depth) || config.depth < 1 || config.depth > 64)))
    throw new Error('Invalid self-play options');
  const choicePath = currentChoiceModel();
  if (!choicePath) throw new Error('Self-play requires a promoted choice model');
  const valuePath = currentNnueModel();
  const choice = new LocalChoice(choicePath);
  const evaluator = valuePath ? new NnueEvaluator(loadNnueModel(valuePath)) : null;
  const openingLines = loadOpeningLines(config.openings, config.openingPlies);
  const random = randomGenerator(config.seed), order = [...openingLines];
  for (let index = order.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [order[index], order[other]] = [order[other], order[index]];
  }
  const gameWriter = createWriteStream(config.output, { flags: 'w' });
  const positionWriter = createWriteStream(config.positionsOutput, { flags: 'w' });
  const metadata = { schema: 'jev-selfplay-v1', games: config.games, moveTime: config.moveTime, depth: config.depth,
    maxPlies: config.maxPlies, seed: config.seed, openingBook: config.openings,
    openingPlies: config.openingPlies, openingBookHash: sha256(config.openings),
    choiceModel: path.basename(choicePath), choiceModelSha256: sha256(choicePath),
    valueModel: valuePath ? path.basename(valuePath) : null,
    valueModelSha256: valuePath ? sha256(valuePath) : null,
    explorePlies: config.explorePlies, temperatureCp: config.temperature,
    repetitionRule: 'single-side-perpetual-check-loss-v1', generatedAt: new Date().toISOString() };
  gameWriter.write(JSON.stringify({ kind: 'meta', format: 'iccs-game-v1', ...metadata }) + '\n');
  positionWriter.write(JSON.stringify({ kind: 'meta', format: 'jev-search-policy-v1', positions: null, ...metadata }) + '\n');
  let written = 0;
  try {
    await choice.ready();
    for (let game = 0; game < config.games; game++) {
      let position = parseFen(), history = [positionKey(position)];
      const opening = order[game % order.length], moves = [], plyRows = [];
      for (const notation of opening) {
        moves.push(notation);
        position = advance(position, notation);
        history.push(positionKey(position));
      }
      while (moves.length < config.maxPlies && !gameResult(position, history)) {
        const legal = legalMoves(position).map(moveName), fen = toFen(position);
        const ranking = await choice.rank(fen, legal, Math.max(10000, config.moveTime * 10), history);
        const priors = new Map(ranking.choices.map(item => [item.move, item.probability]));
        const analysis = chooseMove(position, { timeMs: config.moveTime, history: history.slice(0, -1),
          maxDepth: config.depth || undefined, priors, multiPv: 3, evaluator });
        if (!analysis.move || !analysis.candidates.length) throw new Error(`Search found no move in ${fen}`);
        const policy = scorePolicy(analysis.candidates, config.temperature);
        const exploring = moves.length < config.explorePlies;
        const played = exploring ? samplePolicy(policy, random) : analysis.candidates[0].move;
        const row = { kind: 'position', game, ply: moves.length, fen, legal,
          best: analysis.candidates[0].move, played, searchPolicy: policy,
          modelPolicy: ranking.choices, value: analysis.score, phase: analysis.phase,
          candidates: analysis.candidates, depth: analysis.depth, nodes: analysis.nodes,
          pruned: analysis.pruned, reduced: analysis.reduced };
        positionWriter.write(JSON.stringify(row) + '\n');
        plyRows.push(row);
        written++;
        moves.push(played);
        position = advance(position, played);
        history.push(positionKey(position));
      }
      const result = gameResult(position, history) || { winner: null, reason: 'maxPlies', censored: true };
      gameWriter.write(JSON.stringify({ kind: 'game', game, startFen: toFen(parseFen()), opening,
        moves, chineseMoves: formatChineseLine(parseFen(), moves), finalFen: toFen(position), result,
        generatedPositions: plyRows.length }) + '\n');
      console.log(`game=${game + 1}/${config.games} plies=${moves.length} result=${result.censored ? 'censored' : result.winner || 'draw'} reason=${result.reason}`);
    }
    return { games: config.games, positions: written, output: config.output, positionsOutput: config.positionsOutput };
  } finally {
    choice.close();
    await Promise.all([close(gameWriter), close(positionWriter)]);
  }
}

async function main() {
  const config = options(process.argv.slice(2));
  console.log(JSON.stringify(await generateSelfPlay(config)));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href)
  main().catch(error => { console.error(error); process.exitCode = 1; });
