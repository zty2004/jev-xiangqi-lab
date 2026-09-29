import { parentPort, workerData } from 'node:worker_threads';
import { chooseMove } from './engine.js';
import { loadNnueModel, NnueEvaluator } from './nnue-evaluator.js';
import { parseFen } from './xiangqi.js';

try {
  const position = parseFen(workerData.fen);
  const evaluator = workerData.nnueModel ? new NnueEvaluator(loadNnueModel(workerData.nnueModel)) : null;
  const priors = workerData.priors ? new Map(workerData.priors) : null;
  const analysis = chooseMove(position, {
    timeMs: workerData.timeMs,
    history: workerData.history,
    priors,
    multiPv: workerData.multiPv,
    allowedRootMoves: workerData.allowedRootMoves,
    evaluator
  });
  parentPort.postMessage(analysis);
} catch (error) {
  parentPort.postMessage({ error: error.message });
}
