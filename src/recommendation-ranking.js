export function rankRecommendations(candidates = []) {
  return candidates.map((candidate, index) => ({ candidate, index }))
    .sort((left, right) => {
      const scoreDifference = Number(right.candidate.score) - Number(left.candidate.score);
      return scoreDifference || left.index - right.index;
    })
    .map(({ candidate }) => candidate);
}
