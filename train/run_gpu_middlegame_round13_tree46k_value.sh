#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
name="choice-middle-round13-tree46k-value-128x8-gpu0"
init="${INIT_MODEL:-models/choice-middle-round10-pv8-128x8-gpu0.pt}"
middle150="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
tree="${TREE_LABELLED:-data/teacher-ccpd-middle-tree-r13-labelled.jsonl}"
seed="${TRAIN_SEED:-20261008}"

gpu_inventory="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
if ! grep -Fxq "$gpu_uuid" <<<"$gpu_inventory"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$init" "$middle150" "$tree"; do
  if [[ ! -s "$file" ]]; then echo "Training input is missing: $file" >&2; exit 1; fi
done

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  --data "$middle150" --data "$tree" --source-weight 1:4 \
  --validation-source-index 1 --policy-target dominant --value-loss-weight 1 \
  --mirror-augmentation --selection-metric loss --policy-features v1 \
  --channels 128 --blocks 8 --init-model "$init" --train-value-only \
  --epochs 30 --patience 6 --lr-patience 2 --min-lr 0.00001 \
  --batch 512 --loader-workers 8 --lr 0.001 \
  --seed "$seed" --device cuda --output "models/${name}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_values.py \
  --model-a "$init" --model-b "models/${name}.pt" \
  --data "$middle150" --data "$tree" --test-source-index 1 \
  --phase middlegame --seed "$seed" --device cuda \
  > "reports/compare-${name}-vs-round10-tree.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_values.py \
  --model-a "$init" --model-b "models/${name}.pt" \
  --data "$middle150" --data "$tree" --test-source-index 0 \
  --phase middlegame --seed "$seed" --device cuda \
  > "reports/compare-${name}-vs-round10-middle150-value.json"

python3 - "$init" "models/${name}.pt" <<'PY' \
  > "reports/${name}-policy-stability.json"
import json
import sys
import torch

old_path, new_path = sys.argv[1:]
old = torch.load(old_path, map_location="cpu", weights_only=True)
new = torch.load(new_path, map_location="cpu", weights_only=True)
policy_keys = [key for key in old["state_dict"] if not key.startswith("value.")]
changed = [key for key in policy_keys if not torch.equal(old["state_dict"][key], new["state_dict"][key])]
stable = not changed and old.get("temperature", 1.0) == new.get("temperature", 1.0)
print(json.dumps({
    "schema": "jev-value-only-policy-stability-v1",
    "policyAndTrunkTensorCount": len(policy_keys),
    "changedPolicyAndTrunkTensors": changed,
    "temperatureA": old.get("temperature", 1.0),
    "temperatureB": new.get("temperature", 1.0),
    "bitStable": stable,
}, indent=2))
if not stable:
    raise SystemExit("value-only training changed policy behavior")
PY
