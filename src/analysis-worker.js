import { parentPort, workerData } from 'node:worker_threads';
import { chooseMove } from './engine.js';
import { loadNnueModel, NnueEvaluator } from './nnue-evaluator.js';
import { moveName, parseFen } from './xiangqi.js';

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
    preferredRootMoves: workerData.preferredRootMoves,
    evaluator,
    onDepth: partial => publish(partial),
    onProgress: partial => publish(partial)
  });
  // `chooseMove` keeps its final best move as an internal move object so
  // synchronous callers can reuse it directly.  Messages crossing the worker
  // boundary must use the coordinate notation used by the HTTP API instead.
  publish({ ...analysis, move: analysis.move ? moveName(analysis.move) : null }, true);
} catch (error) {
  parentPort.postMessage({ error: error.message });
}
