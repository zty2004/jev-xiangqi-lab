import { readFileSync } from 'node:fs';

const PIECES = 'KABNRCPkabnrcp';
const SQUARES = 90;
const KING_BUCKETS = 9;
export const NNUE_FEATURES = KING_BUCKETS * PIECES.length * SQUARES;

function ownKing(position, perspective) {
  const square = position.board.indexOf(perspective === 'red' ? 'K' : 'k');
  if (square < 0) return -1;
  return perspective === 'red' ? square : 89 - square;
}

function geometry(position, perspective) {
  const king = ownKing(position, perspective);
  if (king < 0) return null;
  const kingX = king % 9, kingY = Math.floor(king / 9);
  const mirror = kingX > 4;
  const canonicalX = mirror ? 8 - kingX : kingX;
  const bucket = (kingY - 7) * 3 + canonicalX - 3;
  if (bucket < 0 || bucket >= KING_BUCKETS) throw new Error('King outside its palace');
  return { bucket, mirror };
}

function normalizePiece(piece, perspective) {
  if (perspective === 'red') return piece;
  return piece === piece.toUpperCase() ? piece.toLowerCase() : piece.toUpperCase();
}

export function featureIndex(position, perspective, piece, square) {
  const view = geometry(position, perspective);
  if (!view) throw new Error(`Missing ${perspective} king`);
  const { bucket, mirror } = view;
  let normalizedSquare = perspective === 'red' ? square : 89 - square;
  if (mirror) normalizedSquare = Math.floor(normalizedSquare / 9) * 9 + 8 - normalizedSquare % 9;
  const pieceIndex = PIECES.indexOf(normalizePiece(piece, perspective));
  if (pieceIndex < 0) throw new Error(`Unknown piece: ${piece}`);
  return (bucket * PIECES.length + pieceIndex) * SQUARES + normalizedSquare;
}

function decodeFloat32(value, expected, name) {
  if (typeof value !== 'string') throw new Error(`Invalid NNUE ${name}`);
  const bytes = Buffer.from(value, 'base64');
  if (bytes.byteLength !== expected * 4) throw new Error(`Invalid NNUE ${name} length`);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

export function loadNnueModel(filename) {
  const raw = JSON.parse(readFileSync(filename, 'utf8'));
  if (raw.format !== 'jev-xiangqi-nnue-v1' || raw.features !== NNUE_FEATURES ||
      !Number.isInteger(raw.hidden) || raw.hidden < 1 || !Number.isInteger(raw.head) || raw.head < 1)
    throw new Error('Unsupported NNUE model');
  const input = raw.hidden * 4;
  return { ...raw,
    embedding: decodeFloat32(raw.embedding, raw.features * raw.hidden, 'embedding'),
    featureBias: decodeFloat32(raw.featureBias, raw.hidden, 'featureBias'),
    fc1Weight: decodeFloat32(raw.fc1Weight, raw.head * input, 'fc1Weight'),
    fc1Bias: decodeFloat32(raw.fc1Bias, raw.head, 'fc1Bias'),
    outputWeight: decodeFloat32(raw.outputWeight, raw.head * 2, 'outputWeight'),
  };
}

export class NnueEvaluator {
  constructor(model) {
    if (model.features !== NNUE_FEATURES) throw new Error('NNUE feature count mismatch');
    this.model = model;
  }

  refresh(position, perspective) {
    const { hidden, embedding, featureBias } = this.model;
    const accumulator = Float32Array.from(featureBias);
    for (let square = 0; square < SQUARES; square++) {
      const piece = position.board[square];
      if (piece === '.') continue;
      const offset = featureIndex(position, perspective, piece, square) * hidden;
      for (let unit = 0; unit < hidden; unit++) accumulator[unit] += embedding[offset + unit];
    }
    return accumulator;
  }

  createState(position) {
    return { red: this.refresh(position, 'red'), black: this.refresh(position, 'black'),
      redGeometry: geometry(position, 'red'), blackGeometry: geometry(position, 'black') };
  }

  updateState(position, move, next, state) {
    const result = {};
    for (const perspective of ['red', 'black']) {
      const previousGeometry = state[`${perspective}Geometry`];
      const nextGeometry = geometry(next, perspective);
      result[`${perspective}Geometry`] = nextGeometry || previousGeometry;
      if (!nextGeometry) {
        result[perspective] = Float32Array.from(state[perspective]);
        continue;
      }
      if (previousGeometry.bucket !== nextGeometry.bucket || previousGeometry.mirror !== nextGeometry.mirror) {
        result[perspective] = this.refresh(next, perspective);
        continue;
      }
      const accumulator = Float32Array.from(state[perspective]);
      this.addFeature(accumulator, featureIndex(position, perspective, move.piece, move.from), -1);
      this.addFeature(accumulator, featureIndex(next, perspective, move.piece, move.to), 1);
      if (move.captured !== '.') this.addFeature(accumulator, featureIndex(position, perspective, move.captured, move.to), -1);
      result[perspective] = accumulator;
    }
    return result;
  }

  addFeature(accumulator, feature, sign) {
    const { hidden, embedding } = this.model;
    const offset = feature * hidden;
    for (let unit = 0; unit < hidden; unit++) accumulator[unit] += sign * embedding[offset + unit];
  }

  evaluate(position, state = this.createState(position)) {
    const { hidden, head, fc1Weight, fc1Bias, outputWeight, outputBias = 0, outputScale = 1000 } = this.model;
    const own = state[position.side], other = state[position.side === 'red' ? 'black' : 'red'];
    const input = new Float32Array(hidden * 4);
    for (let unit = 0; unit < hidden; unit++) {
      const a = Math.max(0, Math.min(1, own[unit]));
      const b = Math.max(0, Math.min(1, other[unit]));
      input[unit] = a;
      input[hidden + unit] = a * a;
      input[hidden * 2 + unit] = b;
      input[hidden * 3 + unit] = b * b;
    }
    let value = Number(outputBias);
    for (let row = 0; row < head; row++) {
      let sum = fc1Bias[row];
      const offset = row * input.length;
      for (let column = 0; column < input.length; column++) sum += fc1Weight[offset + column] * input[column];
      const active = Math.max(0, Math.min(1, sum));
      value += outputWeight[row] * active + outputWeight[head + row] * active * active;
    }
    return Math.round(Math.tanh(value) * outputScale);
  }
}
