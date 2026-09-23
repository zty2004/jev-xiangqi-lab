const $ = id => document.getElementById(id);
const labels = { K: '帅', A: '仕', B: '相', N: '马', R: '车', C: '炮', P: '兵', k: '将', a: '士', b: '象', n: '马', r: '车', c: '炮', p: '卒' };
const files = 'abcdefghi';
let state = null, selected = null, busy = false, human = 'red', mode = 'play';
let snapshots = [], lastMove = null, analysis = null;

async function api(path, payload) {
  const response = await fetch(path, payload ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}

function squareName(index) { return files[index % 9] + (9 - Math.floor(index / 9)); }
function sideOf(piece) { return piece === '.' ? null : piece === piece.toUpperCase() ? 'red' : 'black'; }
function displayIndex(index) { return $('orientation').value === 'black' ? 89 - index : index; }
function point(name) {
  const index = (9 - Number(name[1])) * 9 + files.indexOf(name[0]);
  const shown = displayIndex(index);
  return { x: 40 + 90 * (shown % 9), y: 40 + 90 * Math.floor(shown / 9) };
}

function renderRecommendations() {
  const overlay = $('recommendation-overlay'), list = $('recommendations');
  overlay.replaceChildren(); list.replaceChildren();
  $('analysis-depth').textContent = analysis?.fen === state?.fen ? `深度 ${analysis.depth}` : '';
  if (mode !== 'teach' || analysis?.fen !== state?.fen) return;
  const colors = ['#a74934', '#246a79', '#6657a5', '#9a6b23', '#39714d'];
  const svg = (tag, attrs) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    return element;
  };
  for (const [index, item] of analysis.recommendations.entries()) {
    const source = point(item.move.slice(0, 2)), target = point(item.move.slice(2));
    const dx = target.x - source.x, dy = target.y - source.y, length = Math.hypot(dx, dy);
    if (length) {
      const ux = dx / length, uy = dy / length;
      const x1 = source.x + ux * 24, y1 = source.y + uy * 24;
      const x2 = target.x - ux * 26, y2 = target.y - uy * 26;
      overlay.append(svg('line', { x1, y1, x2, y2, stroke: colors[index], 'stroke-width': 8 - index,
        'stroke-linecap': 'round', opacity: 0.83 }));
      overlay.append(svg('polygon', { points: `${x2},${y2} ${x2 - ux * 16 - uy * 10},${y2 - uy * 16 + ux * 10} ${x2 - ux * 16 + uy * 10},${y2 - uy * 16 - ux * 10}`,
        fill: colors[index], opacity: 0.9 }));
    }
    const badgeX = Math.max(22, Math.min(778, target.x + 26));
    const badgeY = Math.max(22, Math.min(878, target.y - 27));
    overlay.append(svg('circle', { cx: badgeX, cy: badgeY, r: 18, fill: colors[index], stroke: '#fff8eb', 'stroke-width': 3 }));
    const text = svg('text', { x: badgeX, y: badgeY + 6, 'text-anchor': 'middle', fill: '#fff',
      'font-size': 19, 'font-weight': 800 });
    text.textContent = String(index + 1); overlay.append(text);
    const row = document.createElement('li'), title = document.createElement('strong'), detail = document.createElement('span');
    title.textContent = `${index + 1}. ${item.move}`;
    detail.textContent = `搜索评分 ${item.score >= 0 ? '+' : ''}${item.score}` +
      (item.probability === undefined ? '' : ` · 模型 ${(item.probability * 100).toFixed(1)}%`);
    row.append(title, detail); list.append(row);
  }
}

function render() {
  if (!state) return;
  const pieces = $('pieces'); pieces.replaceChildren();
  const flipped = $('orientation').value === 'black';
  $('river-left').textContent = flipped ? '汉 界' : '楚 河';
  $('river-right').textContent = flipped ? '楚 河' : '汉 界';
  const possible = selected === null ? [] : state.legalMoves.filter(move => move.startsWith(squareName(selected)));
  for (let index = 0; index < 90; index++) {
    const shown = displayIndex(index);
    const x = shown % 9, y = Math.floor(shown / 9), piece = state.board[index], name = squareName(index);
    const square = document.createElement('button');
    square.className = 'square'; square.style.left = `${5 + x * 90 / 8}%`; square.style.top = `${40 / 9 + y * 10}%`;
    square.setAttribute('aria-label', `${name} ${labels[piece] || '空位'}`);
    if (index === selected) square.classList.add('selected');
    if (lastMove && (name === lastMove.slice(0, 2) || name === lastMove.slice(2))) square.classList.add('last');
    if (possible.some(move => move.slice(2) === name)) { square.classList.add('target'); if (piece !== '.') square.classList.add('capture'); }
    if (piece !== '.') {
      const disk = document.createElement('span'); disk.className = `piece ${sideOf(piece)}`; disk.textContent = labels[piece]; square.append(disk);
    }
    square.onclick = () => click(index);
    pieces.append(square);
  }
  renderRecommendations();
  const turn = state.side === 'red' ? '红方' : '黑方';
  $('status').textContent = state.result ? `${state.result.winner ? (state.result.winner === 'red' ? '红方' : '黑方') + '获胜' : '和棋'}` :
    busy ? mode === 'teach' ? '正在分析推荐…' : '电脑正在思考…' : `${turn}走棋${state.inCheck ? ' · 被将军' : ''}`;
  $('detail').textContent = state.result ? state.result.reason : mode === 'teach' ? '双方都由你走棋；推荐仅供参考。' :
    state.side === human ? '选择棋子，再选择落点。' : `搜索引擎正在为${turn}选招。`;
  $('move-count').textContent = `${snapshots.length ? snapshots.length - 1 : 0} 步`;
  $('undo').disabled = busy || snapshots.length <= (mode === 'teach' ? 1 : human === 'black' ? 2 : 1);
  $('new').disabled = busy; $('mode').disabled = busy; $('side').disabled = busy;
  $('time').disabled = busy; $('recommendation-count').disabled = busy; $('load').disabled = busy;
  $('side-control').hidden = mode === 'teach';
  $('recommendation-control').hidden = mode !== 'teach';
  $('recommendations-card').hidden = mode !== 'teach';
  $('time-label').textContent = mode === 'teach' ? '推荐分析时间' : '电脑思考时间';
  const list = $('moves'); list.replaceChildren();
  for (let i = 1; i < snapshots.length; i++) {
    const li = document.createElement('li'), count = document.createElement('span');
    count.textContent = `${Math.ceil(i / 2)}.${i % 2 ? '红' : '黑'}`;
    li.append(count, document.createTextNode(snapshots[i].move || ''));
    list.append(li);
  }
  list.scrollTop = list.scrollHeight;
}

function save(next) { state = next; snapshots.push(structuredClone(next)); lastMove = next.move || null; selected = null; analysis = null; render(); }

async function click(index) {
  if (!state || busy || state.result || (mode === 'play' && state.side !== human)) return;
  const piece = state.board[index], name = squareName(index);
  if (selected !== null) {
    const notation = squareName(selected) + name;
    if (state.legalMoves.includes(notation)) {
      busy = true; render();
      try { save(await api('/api/move', { fen: state.fen, history: state.history, move: notation }));
        if (mode === 'teach') await analyzeCurrent(); else await computerTurn(); }
      catch (error) { alert(error.message); }
      finally { busy = false; render(); }
      return;
    }
  }
  selected = sideOf(piece) === (mode === 'teach' ? state.side : human) ? index : null;
  render();
}

async function computerTurn() {
  if (mode !== 'play' || !state || state.result || state.side === human) return;
  busy = true; render();
  try { save(await api('/api/ai', { fen: state.fen, history: state.history, timeMs: Number($('time').value) })); }
  catch (error) { alert(error.message); }
  finally { busy = false; render(); }
}

async function analyzeCurrent() {
  if (mode !== 'teach' || !state || state.result) { analysis = null; render(); return; }
  const fen = state.fen;
  busy = true; analysis = null; render();
  try {
    const result = await api('/api/analyze', { fen, history: state.history,
      timeMs: Number($('time').value), count: Number($('recommendation-count').value) });
    if (mode === 'teach' && state?.fen === fen) analysis = { fen, ...result };
  } catch (error) { alert(error.message); }
  finally { busy = false; render(); }
}

async function newGame() {
  if (busy) return;
  busy = true; render();
  try {
    mode = $('mode').value; human = $('side').value; state = await api('/api/new'); state.history = [state.fen.split(' ').slice(0, 2).join(' ')];
    snapshots = [structuredClone(state)]; selected = null; lastMove = null; analysis = null; busy = false; render();
    if (mode === 'teach') await analyzeCurrent(); else await computerTurn();
  } catch (error) { busy = false; alert(error.message); render(); }
}

async function loadFen() {
  if (busy) return;
  busy = true; render();
  try {
    const next = await api('/api/state', { fen: $('fen').value.trim() });
    next.history = [next.fen.split(' ').slice(0, 2).join(' ')];
    state = next; snapshots = [structuredClone(next)]; selected = null; lastMove = null; analysis = null; busy = false; render();
    if (mode === 'teach') await analyzeCurrent(); else await computerTurn();
  } catch (error) { busy = false; alert(error.message); render(); }
}

function undo() {
  if (busy || snapshots.length < 2) return;
  if (mode === 'teach') snapshots.pop();
  else do { snapshots.pop(); } while (snapshots.length > 1 && snapshots.at(-1).side !== human);
  state = structuredClone(snapshots.at(-1)); selected = null; lastMove = state.move || null; analysis = null; render();
  if (mode === 'teach') analyzeCurrent();
}

$('new').onclick = newGame; $('undo').onclick = undo; $('load').onclick = loadFen;
$('side').onchange = () => { $('orientation').value = $('side').value; newGame(); };
$('mode').onchange = newGame;
$('orientation').onchange = render;
$('recommendation-count').onchange = analyzeCurrent;
newGame();
