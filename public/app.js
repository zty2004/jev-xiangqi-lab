const $ = id => document.getElementById(id);
const labels = { K: '帅', A: '仕', B: '相', N: '马', R: '车', C: '炮', P: '兵', k: '将', a: '士', b: '象', n: '马', r: '车', c: '炮', p: '卒' };
const files = 'abcdefghi';
let state = null, selected = null, busy = false, human = 'red', snapshots = [], lastMove = null;

async function api(path, payload) {
  const response = await fetch(path, payload ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}

function squareName(index) { return files[index % 9] + (9 - Math.floor(index / 9)); }
function sideOf(piece) { return piece === '.' ? null : piece === piece.toUpperCase() ? 'red' : 'black'; }

function render() {
  if (!state) return;
  const pieces = $('pieces'); pieces.replaceChildren();
  const possible = selected === null ? [] : state.legalMoves.filter(move => move.startsWith(squareName(selected)));
  for (let index = 0; index < 90; index++) {
    const x = index % 9, y = Math.floor(index / 9), piece = state.board[index], name = squareName(index);
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
  const turn = state.side === 'red' ? '红方' : '黑方';
  $('status').textContent = state.result ? `${state.result.winner ? (state.result.winner === 'red' ? '红方' : '黑方') + '获胜' : '和棋'}` : busy ? '电脑正在思考…' : `${turn}走棋${state.inCheck ? ' · 被将军' : ''}`;
  $('detail').textContent = state.result ? state.result.reason : state.side === human ? '选择棋子，再选择落点。' : `搜索引擎正在为${turn}选招。`;
  $('move-count').textContent = `${snapshots.length ? snapshots.length - 1 : 0} 步`;
  $('undo').disabled = busy || snapshots.length <= (human === 'black' ? 2 : 1);
  $('side').disabled = busy; $('time').disabled = busy;
  const list = $('moves'); list.replaceChildren();
  for (let i = 1; i < snapshots.length; i++) {
    const li = document.createElement('li'), count = document.createElement('span');
    count.textContent = `${Math.ceil(i / 2)}.${i % 2 ? '红' : '黑'}`;
    li.append(count, document.createTextNode(snapshots[i].move || ''));
    list.append(li);
  }
  list.scrollTop = list.scrollHeight;
}

function save(next) { state = next; snapshots.push(structuredClone(next)); lastMove = next.move || null; selected = null; render(); }

async function click(index) {
  if (!state || busy || state.result || state.side !== human) return;
  const piece = state.board[index], name = squareName(index);
  if (selected !== null) {
    const notation = squareName(selected) + name;
    if (state.legalMoves.includes(notation)) {
      busy = true; render();
      try { save(await api('/api/move', { fen: state.fen, history: state.history, move: notation })); await computerTurn(); }
      catch (error) { alert(error.message); }
      finally { busy = false; render(); }
      return;
    }
  }
  selected = sideOf(piece) === human ? index : null;
  render();
}

async function computerTurn() {
  if (!state || state.result || state.side === human) return;
  busy = true; render();
  try { save(await api('/api/ai', { fen: state.fen, history: state.history, timeMs: Number($('time').value) })); }
  catch (error) { alert(error.message); }
  finally { busy = false; render(); }
}

async function newGame() {
  busy = true; render();
  try {
    human = $('side').value; state = await api('/api/new'); state.history = [state.fen.split(' ').slice(0, 2).join(' ')];
    snapshots = [structuredClone(state)]; selected = null; lastMove = null; busy = false; render(); await computerTurn();
  } catch (error) { busy = false; alert(error.message); render(); }
}

async function loadFen() {
  busy = true; render();
  try {
    const next = await api('/api/state', { fen: $('fen').value.trim() });
    next.history = [next.fen.split(' ').slice(0, 2).join(' ')];
    state = next; snapshots = [structuredClone(next)]; selected = null; lastMove = null; busy = false; render(); await computerTurn();
  } catch (error) { busy = false; alert(error.message); render(); }
}

function undo() {
  if (busy || snapshots.length < 2) return;
  do { snapshots.pop(); } while (snapshots.length > 1 && snapshots.at(-1).side !== human);
  state = structuredClone(snapshots.at(-1)); selected = null; lastMove = state.move || null; render();
}

$('new').onclick = newGame; $('undo').onclick = undo; $('load').onclick = loadFen;
$('side').onchange = newGame;
newGame();
