#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
model="${MODEL:-models/choice-middle-round12-tree-value-128x8-gpu0.pt}"
data="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
positions="${POSITIONS:-1000}"
offset="${SAMPLE_OFFSET:-200}"
sample_seed="${SAMPLE_SEED:-20261007}"
prefix="${REPORT_PREFIX:-reports/selective-minimax-round12-confirmation}"

gpu_inventory="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
if ! grep -Fxq "$gpu_uuid" <<<"$gpu_inventory"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$model" "$data"; do
  if [[ ! -s "$file" ]]; then echo "Required input is missing: $file" >&2; exit 1; fi
done

configurations=("2 2 0.2 fast" "2 8 0.2 wide" "3 2 0.4 deep")
reports=()
for configuration in "${configurations[@]}"; do
  read -r depth width prior label <<<"$configuration"
  report="${prefix}-${label}.json"
  reports+=("$report")
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -m jev_engine.benchmark_selective_minimax \
    --model "$model" --data "$data" --positions "$positions" --sample-offset "$offset" \
    --depth "$depth" --internal-width "$width" --root-prior-weight "$prior" \
    --batch-size 512 --phase middlegame --seed 20261002 \
    --sample-seed "$sample_seed" --device cuda > "$report"
done

python3 - "${reports[@]}" <<'PY' > "${prefix}-summary.json"
import json
import sys

configurations = []
for filename in sys.argv[1:]:
    with open(filename) as handle:
        item = json.load(handle)
    configurations.append({key: item[key] for key in (
        "depth", "internalWidth", "rootPriorWeight", "directTop1", "searchTop1",
        "top1Gain", "directTop8", "searchTop8", "top8Gain", "corrections", "spoils",
        "medianElapsedMs", "meanModelEvaluations")})
    configurations[-1]["report"] = filename
first = json.load(open(sys.argv[1]))
print(json.dumps({
    "schema": "jev-selective-minimax-confirmation-v1",
    "model": first["model"],
    "modelSha256": first["modelSha256"],
    "sample": {"positions": first["positions"], "offset": first["sampleOffset"],
               "seed": first["sampleSeed"]},
    "configurations": configurations,
}, indent=2))
PY

cat "${prefix}-summary.json"
