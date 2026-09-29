import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { chineseMove, formatChineseLine, formatChineseMove } from './src/chinese-notation.js';
import { LocalChoice } from './src/local-choice.js';
import { currentChoiceModel, currentNnueModel } from './src/model-selection.js';
import { loadMasterOpeningBook, masterOpeningCandidates } from './src/opening-book.js';
import { gameResult, isInCheck, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT) || 3000;
const choiceModel = currentChoiceModel();
const localChoice = choiceModel ? new LocalChoice(choiceModel) : null;
const nnueModel = currentNnueModel();
const masterBook = process.env.OPENING_BOOK === 'off' ? null :
  loadMasterOpeningBook(process.env.OPENING_BOOK || new URL('./data/master-opening-book.json', import.meta.url));
const assets = { '/': ['index.html', 'text/html'], '/app.js': ['app.js', 'text/javascript'], '/style.css': ['style.css', 'text/css'] };

function json(response, status, data) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(data));
}

async function body(request) {
  let raw = '';
  for await (const chunk of request) {
    raw += chunk;
    if (raw.length > 100_000) throw new Error('Request too large');
  }
  return JSON.parse(raw || '{}');
}

function view(position, history = []) {
  return { fen: toFen(position), side: position.side, board: position.board,
    legalMoves: legalMoves(position).map(moveName), inCheck: isInCheck(position),
    result: gameResult(position, history) };
}

function openingRootMoves(position) {
  const candidates = masterOpeningCandidates(position, masterBook);
  return { candidates, allowedRootMoves: candidates.length ? candidates.map(item => item.move) : null };
}

function startAnalysisWorker(position, history, timeMs, multiPv, allowedRootMoves, priors) {
  return new Worker(new URL('./src/analysis-worker.js', import.meta.url), { workerData: {
    fen: toFen(position), history: history.slice(0, -1), timeMs, multiPv, allowedRootMoves,
    priors: priors ? [...priors] : null, nnueModel
  } });
}

function analysisView(position, analysis, priors, count) {
  const redScore = score => position.side === 'red' ? score : -score;
  return { recommendations: analysis.candidates.slice(0, count).map(item => ({
    move: item.move, notation: formatChineseMove(position, item.move), score: redScore(item.score),
    pv: item.pv || [item.move], pvNotation: formatChineseLine(position, item.pv || [item.move]),
    ...(priors ? { probability: priors.get(item.move) || 0 } : {})
  })), phase: analysis.phase, depth: analysis.depth, nodes: analysis.nodes,
  pruned: analysis.pruned, reduced: analysis.reduced, timeMs: analysis.timeMs, score: redScore(analysis.score) };
}

async function analyzePosition(position, history, timeMs, multiPv = 1, allowedRootMoves = null) {
  const ranking = localChoice ? await localChoice.rank(toFen(position), legalMoves(position).map(moveName), Math.max(5000, timeMs), history) : null;
  const priors = ranking ? new Map(ranking.choices.map(item => [item.move, item.probability])) : null;
  const analysis = await new Promise((resolve, reject) => {
    const worker = startAnalysisWorker(position, history, timeMs, multiPv, allowedRootMoves, priors);
    worker.on('message', message => {
      if (message.error) reject(new Error(message.error));
      else if (message.kind === 'final') resolve(message.analysis);
    });
    worker.once('error', reject);
    worker.once('exit', code => { if (code !== 0) reject(new Error(`Analysis worker exited with ${code}`)); });
  });
  return { analysis, ranking, priors };
}

async function streamAnalysis(response, position, history, timeMs, count) {
  const ranking = localChoice ? await localChoice.rank(toFen(position), legalMoves(position).map(moveName), Math.max(5000, timeMs), history) : null;
  const priors = ranking ? new Map(ranking.choices.map(item => [item.move, item.probability])) : null;
  response.writeHead(200, { 'Content-Type': 'application/x-ndjson; charset=utf-8', 'Cache-Control': 'no-store' });
  const { allowedRootMoves } = openingRootMoves(position);
  const worker = startAnalysisWorker(position, history, timeMs, count, allowedRootMoves, priors);
  let closed = false, latest = null;
  const refresh = setInterval(() => {
    if (!closed && latest) response.write(JSON.stringify({ kind: 'progress', ...analysisView(position, latest, priors, count) }) + '\n');
  }, 1000);
  response.on('close', () => { closed = true; clearInterval(refresh); worker.terminate(); });
  await new Promise((resolve, reject) => {
    worker.on('message', message => {
      if (message.error) { reject(new Error(message.error)); return; }
      latest = message.analysis;
      if (!closed) response.write(JSON.stringify({ kind: message.kind, ...analysisView(position, latest, priors, count) }) + '\n');
      if (message.kind === 'final') resolve();
    });
    worker.once('error', reject);
    worker.once('exit', code => { if (code !== 0 && !closed) reject(new Error(`Analysis worker exited with ${code}`)); });
  });
  clearInterval(refresh);
  if (!closed) response.end();
}

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
    if (request.method === 'GET' && assets[url.pathname]) {
      const [filename, type] = assets[url.pathname];
      const data = await readFile(path.join(root, 'public', filename));
      response.writeHead(200, { 'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-store' });
      response.end(data);
    } else if (request.method === 'GET' && url.pathname === '/api/new') {
      const position = parseFen();
      json(response, 200, view(position, [positionKey(position)]));
    } else if (request.method === 'POST' && url.pathname === '/api/state') {
      const data = await body(request), position = parseFen(data.fen);
      json(response, 200, view(position, data.history || []));
    } else if (request.method === 'POST' && url.pathname === '/api/move') {
      const data = await body(request), position = parseFen(data.fen);
      let notation = data.move;
      if (typeof notation !== 'string') return json(response, 400, { error: '请提供中文棋谱走法' });
      if (!/^[a-i][0-9][a-i][0-9]$/.test(notation)) notation = chineseMove(position, notation);
      const move = legalMoves(position).find(candidate => moveName(candidate) === notation);
      if (!move) return json(response, 400, { error: '这一步不符合象棋规则' });
      const moveChinese = formatChineseMove(position, notation);
      const next = makeMove(position, move), history = [...(data.history || []), positionKey(next)];
      json(response, 200, { ...view(next, history), move: moveName(move), moveChinese, history });
    } else if (request.method === 'POST' && url.pathname === '/api/ai') {
      const data = await body(request), position = parseFen(data.fen), history = data.history || [];
      if (gameResult(position, history)) return json(response, 400, { error: '棋局已经结束' });
      const timeMs = Math.min(30_000, Math.max(100, Number(data.timeMs) || 1000));
      const { candidates: bookMoves, allowedRootMoves } = openingRootMoves(position);
      const { analysis, ranking, priors } = await analyzePosition(position, history, timeMs, 1,
        allowedRootMoves);
      if (!analysis.move) return json(response, 400, { error: '无合法走法' });
      const selectedNotation = analysis.move, moveChinese = formatChineseMove(position, selectedNotation);
      const choice = ranking ? { selected: selectedNotation, probability: priors.get(selectedNotation) || 0,
        rankedMoves: ranking.choices.length } : null;
      const bookEntry = bookMoves.find(item => item.move === selectedNotation);
      const opening = bookEntry ? { source: masterBook.source, sourceUrl: masterBook.sourceUrl,
        masterGames: bookEntry.masterGames, screenHorseGames: bookEntry.screenHorseGames,
        screenHorseRepertoire: bookEntry.screenHorseRepertoire, candidateCount: bookMoves.length,
        sourceExamples: bookEntry.sourceExamples } : null;
      const chosen = legalMoves(position).find(move => moveName(move) === selectedNotation);
      const next = makeMove(position, chosen), nextHistory = [...history, positionKey(next)];
      json(response, 200, { ...view(next, nextHistory), move: selectedNotation, moveChinese,
        analysis: { phase: analysis.phase, depth: analysis.depth, score: analysis.score, nodes: analysis.nodes,
          pruned: analysis.pruned, reduced: analysis.reduced, timeMs: analysis.timeMs,
          pv: analysis.pv, pvChinese: formatChineseLine(position, analysis.pv), choice, opening }, history: nextHistory });
    } else if (request.method === 'POST' && url.pathname === '/api/analyze') {
      const data = await body(request), position = parseFen(data.fen), history = data.history || [];
      if (gameResult(position, history)) return json(response, 200, { recommendations: [], depth: 0, nodes: 0, timeMs: 0 });
      const timeMs = Math.min(30_000, Math.max(100, Number(data.timeMs) || 1000));
      const count = Math.min(5, Math.max(1, Math.trunc(Number(data.count) || 3)));
      const { allowedRootMoves } = openingRootMoves(position);
      const { analysis, priors } = await analyzePosition(position, history, timeMs, count, allowedRootMoves);
      json(response, 200, analysisView(position, analysis, priors, count));
    } else if (request.method === 'POST' && url.pathname === '/api/analyze-stream') {
      const data = await body(request), position = parseFen(data.fen), history = data.history || [];
      if (gameResult(position, history)) return json(response, 200, { kind: 'final', recommendations: [], depth: 0, nodes: 0, timeMs: 0 });
      const timeMs = Math.min(30_000, Math.max(100, Number(data.timeMs) || 1000));
      const count = Math.min(5, Math.max(1, Math.trunc(Number(data.count) || 3)));
      await streamAnalysis(response, position, history, timeMs, count);
    } else json(response, 404, { error: 'Not found' });
  } catch (error) {
    json(response, 400, { error: error.message });
  }
});

server.listen(port, '127.0.0.1', () => console.log(`中国象棋已启动：http://127.0.0.1:${port}`));
