#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
production="${PRODUCTION_MODEL:-models/choice-middle-round10-pv8-128x8-gpu0.pt}"
candidate="${CANDIDATE_MODEL:-models/choice-middle-round13-tree46k-value-128x8-gpu0.pt}"
data="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
positions="${POSITIONS:-500}"
seed="${SPLIT_SEED:-20261008}"
sample_seed="${SAMPLE_SEED:-20261009}"

gpu_inventory="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
if ! grep -Fxq "$gpu_uuid" <<<"$gpu_inventory"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$production" "$candidate" "$data"; do
  if [[ ! -s "$file" ]]; then echo "Required input is missing: $file" >&2; exit 1; fi
done

for spec in "production $production" "candidate $candidate"; do
  read -r label model <<<"$spec"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -m jev_engine.benchmark_policy_search \
    --model "$model" --data "$data" --positions "$positions" \
    --simulations 128 --batch-size 16 --maximum-depth 32 --phase middlegame \
    --seed "$seed" --sample-seed "$sample_seed" --device cuda \
    > "reports/round13-search-gate-puct-${label}.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -m jev_engine.benchmark_selective_minimax \
    --model "$model" --data "$data" --positions "$positions" --sample-offset 0 \
    --depth 2 --internal-width 8 --root-prior-weight 0.2 --batch-size 512 \
    --phase middlegame --seed "$seed" --sample-seed "$sample_seed" --device cuda \
    > "reports/round13-search-gate-minimax-${label}.json"
done

python3 - <<'PY' > reports/round13-search-gate-summary.json
import json

def read(path):
    with open(path) as handle:
        return json.load(handle)

report = {"schema": "jev-round13-search-gate-v1", "searches": {}}
for search in ("puct", "minimax"):
    old = read(f"reports/round13-search-gate-{search}-production.json")
    new = read(f"reports/round13-search-gate-{search}-candidate.json")
    report["searches"][search] = {
        "positions": old["positions"],
        "directTop1": old["directTop1"],
        "productionSearchTop1": old["searchTop1"],
        "candidateSearchTop1": new["searchTop1"],
        "candidateMinusProductionTop1": new["searchTop1"] - old["searchTop1"],
        "productionSearchTop8": old["searchTop8"],
        "candidateSearchTop8": new["searchTop8"],
        "candidateMinusProductionTop8": new["searchTop8"] - old["searchTop8"],
        "productionCorrections": old["corrections"],
        "productionSpoils": old["spoils"],
        "candidateCorrections": new["corrections"],
        "candidateSpoils": new["spoils"],
    }
print(json.dumps(report, indent=2))
PY

cat reports/round13-search-gate-summary.json
