import { parentPort, workerData } from 'node:worker_threads';
import { chooseMove } from './engine.js';
import { loadNnueModel, NnueEvaluator } from './nnue-evaluator.js';
import { parseFen } from './xiangqi.js';

try {
  const position = parseFen(workerData.fen);
  const evaluator = workerData.nnueModel ? new NnueEvaluator(loadNnueModel(workerData.nnueModel)) : null;
  const priors = workerData.priors ? new Map(workerData.priors) : null;
  let lastPublished = -Infinity;
  const publish = (analysis, final = false) => {
    const now = performance.now();
    if (!final && lastPublished !== -Infinity && now - lastPublished < 1000) return;
    lastPublished = now;
    parentPort.postMessage({ kind: final ? 'final' : 'progress', analysis });
  };
  const analysis = chooseMove(position, {
    timeMs: workerData.timeMs,
    history: workerData.history,
    priors,
    multiPv: workerData.multiPv,
    allowedRootMoves: workerData.allowedRootMoves,
    evaluator,
    onDepth: partial => publish(partial),
    onProgress: partial => publish(partial)
  });
  publish(analysis, true);
} catch (error) {
  parentPort.postMessage({ error: error.message });
}
