#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
name="choice-middle-round12-tree-value-128x8-gpu0"
init="${INIT_MODEL:-models/choice-middle-round10-pv8-128x8-gpu0.pt}"
middle150="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
tree="${TREE_LABELLED:-data/teacher-ccpd-middle-tree-r12-pilot-labelled.jsonl}"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$init" "$middle150" "$tree"; do
  if [[ ! -s "$file" ]]; then echo "Training input is missing: $file" >&2; exit 1; fi
done

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  --data "$middle150" --data "$tree" --source-weight 1:8 \
  --validation-source-index 1 --policy-target dominant --value-loss-weight 1 \
  --mirror-augmentation --selection-metric loss --policy-features v1 \
  --channels 128 --blocks 8 --init-model "$init" --train-value-only \
  --epochs 30 --patience 6 --lr-patience 2 --min-lr 0.00001 \
  --batch 512 --loader-workers 8 --lr 0.001 \
  --seed 20261006 --device cuda --output "models/${name}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_values.py \
  --model-a "$init" --model-b "models/${name}.pt" \
  --data "$middle150" --data "$tree" --test-source-index 1 \
  --phase middlegame --seed 20261006 --device cuda \
  > "reports/compare-${name}-vs-round10-tree.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_values.py \
  --model-a "$init" --model-b "models/${name}.pt" \
  --data "$middle150" --phase middlegame --seed 20261003 --device cuda \
  > "reports/compare-${name}-vs-round10-middle150-value.json"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 - "$init" "models/${name}.pt" <<'PY' \
  > "reports/${name}-policy-stability.json"
import hashlib
import json
import sys
import torch

old_path, new_path = sys.argv[1:]
old = torch.load(old_path, map_location="cpu", weights_only=True)
new = torch.load(new_path, map_location="cpu", weights_only=True)
policy_keys = [key for key in old["state_dict"] if not key.startswith("value.")]
changed = [key for key in policy_keys if not torch.equal(old["state_dict"][key], new["state_dict"][key])]
print(json.dumps({
    "schema": "jev-value-only-policy-stability-v1",
    "policyAndTrunkTensorCount": len(policy_keys),
    "changedPolicyAndTrunkTensors": changed,
    "temperatureA": old.get("temperature", 1.0),
    "temperatureB": new.get("temperature", 1.0),
    "bitStable": not changed and old.get("temperature", 1.0) == new.get("temperature", 1.0),
}, indent=2))
if changed or old.get("temperature", 1.0) != new.get("temperature", 1.0):
    raise SystemExit("value-only training changed policy behavior")
PY

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -m jev_engine.benchmark_policy_search \
  --model "models/${name}.pt" --data "$middle150" --positions 200 \
  --simulations 128 --batch-size 16 --maximum-depth 32 --phase middlegame \
  --seed 20261002 --sample-seed 20261004 --device cuda \
  > "reports/jev-all-node-puct-${name}-middle150-pilot.json"
