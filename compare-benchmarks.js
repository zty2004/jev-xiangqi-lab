import { readFileSync, writeFileSync } from 'node:fs';

const args = process.argv.slice(2);
function options(name) {
  const values = [];
  for (let index = 0; index < args.length; index++) if (args[index] === name) values.push(args[index + 1]);
  return values;
}
function option(name) { const index = args.indexOf(name); return index < 0 ? null : args[index + 1]; }
const inputs = options('--input');
const output = option('--output');
if (inputs.length < 2 || !output)
  throw new Error('Usage: node compare-benchmarks.js --input first.json --input second.json --output comparison.json');

const reports = inputs.map(filename => ({ filename, report: JSON.parse(readFileSync(filename, 'utf8')) }));
const reference = reports[0].report.settings;
for (const { filename, report } of reports) {
  for (const key of ['moveTime', 'maxPlies', 'masterOpeningBookHash', 'pikafishHash', 'baselineHash', 'rulesHash']) {
    if (report.settings[key] !== reference[key]) throw new Error(`${key} differs in ${filename}`);
  }
}
const grouped = new Map();
for (const { filename, report } of reports) {
  const hash = report.settings.choiceModelHash;
  const model = grouped.get(hash) || { reports: [], choiceModelHash: hash, gameRows: [], abortedGamesExcluded: 0 };
  model.reports.push(filename);
  model.gameRows.push(...report.games);
  model.abortedGamesExcluded += report.abortedGames?.length || 0;
  grouped.set(hash, model);
}
const models = [...grouped.values()].map(model => {
  const wins = model.gameRows.filter(game => game.engineScore === 1).length;
  const draws = model.gameRows.filter(game => game.engineScore === 0.5).length;
  const losses = model.gameRows.filter(game => game.engineScore === 0).length;
  return { reports: model.reports, choiceModelHash: model.choiceModelHash,
    games: model.gameRows.length, points: model.gameRows.reduce((sum, game) => sum + game.engineScore, 0),
    wins, draws, losses, abortedGamesExcluded: model.abortedGamesExcluded,
    meanPlies: model.gameRows.reduce((sum, game) => sum + game.plies, 0) / model.gameRows.length,
    sides: model.gameRows.map(game => game.engineSide) };
});
const comparison = { purpose: 'Same-engine paired smoke comparison against a fixed Pikafish binary',
  sharedSettings: { moveTimeMs: reference.moveTime, maxPlies: reference.maxPlies,
    masterOpeningBookHash: reference.masterOpeningBookHash, pikafishHash: reference.pikafishHash,
    engineHash: reference.baselineHash, rulesHash: reference.rulesHash }, models,
  limitations: 'Two games per model only validate the match pipeline and can reject a candidate that loses both games; they cannot estimate a stable Elo difference.' };
writeFileSync(output, JSON.stringify(comparison, null, 2) + '\n');
console.log(JSON.stringify({ models: models.map(model => ({ hash: model.choiceModelHash.slice(0, 12),
  record: `${model.wins}-${model.draws}-${model.losses}`, meanPlies: model.meanPlies })), output }));
