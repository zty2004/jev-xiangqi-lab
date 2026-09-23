import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chooseMove } from './src/engine.js';
import { LocalChoice } from './src/local-choice.js';
import { gameResult, isInCheck, legalMoves, makeMove, moveName, parseFen, positionKey, toFen } from './src/xiangqi.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT) || 3000;
const localChoice = process.env.CHOICE_MODEL ? new LocalChoice(process.env.CHOICE_MODEL) : null;
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
      const move = legalMoves(position).find(candidate => moveName(candidate) === data.move);
      if (!move) return json(response, 400, { error: '这一步不符合象棋规则' });
      const next = makeMove(position, move), history = [...(data.history || []), positionKey(next)];
      json(response, 200, { ...view(next, history), move: moveName(move), history });
    } else if (request.method === 'POST' && url.pathname === '/api/ai') {
      const data = await body(request), position = parseFen(data.fen), history = data.history || [];
      if (gameResult(position, history)) return json(response, 400, { error: '棋局已经结束' });
      const timeMs = Math.min(10_000, Math.max(100, Number(data.timeMs) || 1000));
      const started = performance.now();
      const ranking = localChoice ? await localChoice.rank(toFen(position), legalMoves(position).map(moveName), timeMs) : null;
      const priors = ranking ? new Map(ranking.choices.map(item => [item.move, item.probability])) : null;
      const analysis = chooseMove(position, { timeMs: Math.max(10, timeMs - (performance.now() - started)),
        history: history.slice(0, -1), priors });
      if (!analysis.move) return json(response, 400, { error: '无合法走法' });
      const chosen = analysis.move;
      const choice = ranking ? { selected: moveName(chosen), probability: priors.get(moveName(chosen)) || 0,
        rankedMoves: ranking.choices.length } : null;
      const next = makeMove(position, chosen), nextHistory = [...history, positionKey(next)];
      json(response, 200, { ...view(next, nextHistory), move: moveName(chosen), analysis: { depth: analysis.depth, score: analysis.score, nodes: analysis.nodes, timeMs: analysis.timeMs, choice }, history: nextHistory });
    } else json(response, 404, { error: 'Not found' });
  } catch (error) {
    json(response, 400, { error: error.message });
  }
});

server.listen(port, '127.0.0.1', () => console.log(`中国象棋已启动：http://127.0.0.1:${port}`));
