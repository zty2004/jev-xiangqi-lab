#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
model="${MODEL:-models/choice-middle-round12-tree-value-128x8-gpu0.pt}"
data="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
positions="${POSITIONS:-200}"
offset="${SAMPLE_OFFSET:-0}"
sample_seed="${SAMPLE_SEED:-20261007}"
prefix="${REPORT_PREFIX:-reports/selective-minimax-round12-tuning}"

gpu_inventory="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
if ! grep -Fxq "$gpu_uuid" <<<"$gpu_inventory"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$model" "$data"; do
  if [[ ! -s "$file" ]]; then echo "Required input is missing: $file" >&2; exit 1; fi
done

configurations=(
  "2 2 0.05" "2 2 0.1" "2 2 0.2" "2 2 0.4"
  "2 4 0.05" "2 4 0.1" "2 4 0.2" "2 4 0.4"
  "2 8 0.05" "2 8 0.1" "2 8 0.2" "2 8 0.4"
  "3 2 0.2" "3 2 0.4" "3 2 0.8"
  "3 4 0.2" "3 4 0.4" "3 4 0.8"
)
reports=()
for configuration in "${configurations[@]}"; do
  read -r depth width prior <<<"$configuration"
  report="${prefix}-d${depth}-w${width}-p${prior}.json"
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

rows = []
for filename in sys.argv[1:]:
    with open(filename) as handle:
        item = json.load(handle)
    rows.append({
        "report": filename,
        "depth": item["depth"],
        "internalWidth": item["internalWidth"],
        "rootPriorWeight": item["rootPriorWeight"],
        "directTop1": item["directTop1"],
        "searchTop1": item["searchTop1"],
        "top1Gain": item["top1Gain"],
        "searchTop8": item["searchTop8"],
        "top8Gain": item["top8Gain"],
        "corrections": item["corrections"],
        "spoils": item["spoils"],
        "medianElapsedMs": item["medianElapsedMs"],
        "meanModelEvaluations": item["meanModelEvaluations"],
    })
rows.sort(key=lambda item: (item["searchTop1"], item["searchTop8"],
                            -item["medianElapsedMs"]), reverse=True)
print(json.dumps({
    "schema": "jev-selective-minimax-sweep-v1",
    "sample": {
        "positions": json.load(open(sys.argv[1]))["positions"],
        "offset": json.load(open(sys.argv[1]))["sampleOffset"],
        "seed": json.load(open(sys.argv[1]))["sampleSeed"],
    },
    "configurations": rows,
}, indent=2))
PY

cat "${prefix}-summary.json"
