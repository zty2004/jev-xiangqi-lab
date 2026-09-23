import { legalMoves, moveName } from './xiangqi.js';

const digits = new Map([...Array.from('一二三四五六七八九').map((value, index) => [value, index + 1]),
  ...Array.from('１２３４５６７８９').map((value, index) => [value, index + 1]),
  ...Array.from('123456789').map((value, index) => [value, index + 1])]);
const pieces = { 車: 'r', 车: 'r', 馬: 'n', 马: 'n', 相: 'b', 象: 'b', 仕: 'a', 士: 'a', 帥: 'k', 帅: 'k', 將: 'k', 将: 'k', 炮: 'c', 砲: 'c', 兵: 'p', 卒: 'p' };
const actions = { 平: 'flat', 進: 'forward', 进: 'forward', 退: 'backward' };
const prefix = { 前: 0, 中: 1, 後: -1, 后: -1 };

function fileNumber(x, side) { return side === 'red' ? x + 1 : 9 - x; }

export function chineseMove(position, notation) {
  const chars = Array.from(notation.trim());
  if (chars.length !== 4) throw new Error(`Unsupported notation: ${notation}`);
  const prefixed = chars[0] in prefix;
  const type = pieces[prefixed ? chars[1] : chars[0]];
  const action = actions[chars[2]], target = digits.get(chars[3]);
  if (!type || !action || !target) throw new Error(`Unsupported notation: ${notation}`);
  const { side, board } = position;
  let candidates = legalMoves(position).filter(move => {
    if (move.piece.toLowerCase() !== type) return false;
    const fromX = move.from % 9, fromY = Math.floor(move.from / 9);
    const toX = move.to % 9, toY = Math.floor(move.to / 9);
    if (!prefixed && fileNumber(fromX, side) !== digits.get(chars[1])) return false;
    const progress = (toY - fromY) * (side === 'red' ? -1 : 1);
    if (action === 'flat' && progress !== 0) return false;
    if (action === 'forward' && progress <= 0) return false;
    if (action === 'backward' && progress >= 0) return false;
    const usesFile = action === 'flat' || ['n', 'b', 'a'].includes(type);
    return (usesFile ? fileNumber(toX, side) : Math.abs(toY - fromY)) === target;
  });
  if (prefixed) {
    candidates = candidates.filter(move => {
      const x = move.from % 9;
      const matching = board.map((piece, index) => ({ piece, index }))
        .filter(item => item.piece.toLowerCase() === type && item.piece !== '.' && (item.piece === item.piece.toUpperCase()) === (side === 'red') && item.index % 9 === x)
        .sort((a, b) => side === 'red' ? a.index - b.index : b.index - a.index);
      const rank = prefix[chars[0]] === -1 ? matching.length - 1 : prefix[chars[0]];
      return matching.length >= 2 && matching[rank]?.index === move.from;
    });
  }
  if (candidates.length !== 1) throw new Error(`${notation}: matched ${candidates.length} legal moves`);
  return moveName(candidates[0]);
}
