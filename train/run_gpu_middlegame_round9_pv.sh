#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
name="choice-middle-round9-pv-128x8-gpu0"
init="${INIT_MODEL:-models/choice-middle-150k-128x8-v1-gpu0.pt}"
middle150="${MIDDLE_150K:-data/teacher-ccpd-all-middle-150k-nodes250k.jsonl}"
pvdata="${PV_DATA:-data/teacher-ccpd-all-middle-20k-pvexpanded.jsonl}"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2; exit 1
fi
inputs=(
  "$middle150" "$pvdata" data/teacher-games-160.jsonl
  data/teacher-selfplay-middle-2400-pikafish.jsonl data/teacher-ccpd-middle-4000-pikafish.jsonl
  "$init"
)
for file in "${inputs[@]}"; do
  if [[ ! -s "$file" ]]; then echo "Training input is missing: $file" >&2; exit 1; fi
done

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  --data "$middle150" --data "$pvdata" --data data/teacher-games-160.jsonl \
  --data data/teacher-selfplay-middle-2400-pikafish.jsonl \
  --data data/teacher-ccpd-middle-4000-pikafish.jsonl \
  --source-weight 2:2 --source-weight 3:2 --source-weight 4:2 \
  --training-only-row-source pikafish-pv-continuation \
  --policy-target dominant --value-loss-weight 0.1 --mirror-augmentation \
  --selection-metric top1 --policy-features v1 --channels 128 --blocks 8 \
  --init-model "$init" --epochs 30 --patience 8 --lr-patience 2 \
  --min-lr 0.000001 --batch 256 --loader-workers 8 --lr 0.00002 \
  --seed 20261001 --device cuda --output "models/${name}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" --data "$middle150" --phase middlegame \
  --seed 20261001 --device cuda > "reports/${name}-middle150-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
  --model-a "$init" --model-b "models/${name}.pt" --data "$middle150" \
  --phase middlegame --seed 20261001 --device cuda \
  > "reports/compare-${name}-vs-round7-middle150.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
  --model-a models/choice-selfplay-round1-finetune-gpu0.pt \
  --model-b "models/${name}.pt" --data "$middle150" \
  --phase middlegame --seed 20261001 --device cuda \
  > "reports/compare-${name}-vs-production-middle150.json"

old_teacher=(
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --data data/teacher-selfplay-middle-2400-pikafish.jsonl
  --data data/teacher-ccpd-middle-4000-pikafish.jsonl
)
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" "${old_teacher[@]}" \
  --phase middlegame --seed 20260930 --device cuda \
  > "reports/${name}-prior-independent-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" --data data/teacher-ccpd-all-middle-20k-pv250k.jsonl \
  --phase middlegame --seed 20260930 --device cuda \
  > "reports/${name}-pv20k-root-test.json"
