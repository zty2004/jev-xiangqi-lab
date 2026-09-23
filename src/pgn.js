export function decodePgn(bytes) {
  try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
  catch { return new TextDecoder('big5', { fatal: true }).decode(bytes); }
}

export function pgnHeader(pgn, key) {
  return pgn.match(new RegExp(`^\\[${key} "(.*?)"\\]`, 'm'))?.[1] || '';
}

export function pgnMoves(pgn, strict = false) {
  const body = pgn.replace(/^\[.*\]$/gm, '').replace(/\{[^}]*\}/gs, '').replace(/\([^)]*\)/gs, '')
    .replace(/\d+\.(?:\.\.)?/g, ' ');
  const moves = [];
  for (const token of body.split(/\s+/).filter(Boolean)) {
    if (/^[\p{Script=Han}０-９0-9]{4}$/u.test(token)) moves.push(token);
    else if (!['1-0', '0-1', '1/2-1/2', '*'].includes(token) && strict)
      throw new Error(`Unrecognized PGN token: ${token}`);
  }
  return moves;
}
