#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
node_bin="${NODE_BIN:-node}"
teacher="${MIDDLE_TEACHER:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
base="${BASE_MODEL:-models/choice-selfplay-round1-finetune-gpu0.pt}"
name="choice-middle-150k-128x8-planes-gpu0"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2; exit 1
fi
for file in "$teacher" "$base"; do
  if [[ ! -s "$file" ]]; then echo "Training input is missing: $file" >&2; exit 1; fi
done
"$node_bin" verify-data.js "$teacher"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  --data "$teacher" --policy-target dominant --value-loss-weight 0.1 \
  --mirror-augmentation --selection-metric top1 --policy-features planes \
  --channels 128 --blocks 8 --epochs 40 --patience 10 --lr-patience 2 \
  --min-lr 0.000001 --batch 256 --loader-workers 8 --lr 0.0002 \
  --seed 20261001 --device cuda --output "models/${name}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" --data "$teacher" --phase middlegame \
  --seed 20261001 --device cuda > "reports/${name}-fixednodes-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
  --model-a "$base" --model-b "models/${name}.pt" --data "$teacher" \
  --phase middlegame --seed 20261001 --device cuda \
  > "reports/compare-${name}-fixednodes-test.json"
