#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
node_bin="${NODE_BIN:-node}"
teacher="${MIDDLE_TEACHER:-data/teacher-selfplay-middle-2400-pikafish.jsonl}"
base="${BASE_MODEL:-models/choice-selfplay-round1-finetune-gpu0.pt}"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$teacher" "$base"; do
  if [[ ! -s "$file" ]]; then
    echo "Training input is missing: $file" >&2
    exit 1
  fi
done
"$node_bin" verify-data.js "$teacher"

old_teacher=(
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --data data/teacher-ccpd-competition-1200.jsonl
)

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "$base" "${old_teacher[@]}" --data "$teacher" --test-source-index 3 \
  --phase middlegame --device cuda \
  > reports/choice-middle-baseline-new-teacher.json
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "$base" "${old_teacher[@]}" --data "$teacher" \
  --test-source-index 0 --test-source-index 1 --test-source-index 2 --phase middlegame --device cuda \
  > reports/choice-middle-baseline-independent.json

for spec in dominant:4 best:6; do
  target="${spec%%:*}"
  weight="${spec#*:}"
  name="choice-middle-${target}-w${weight}-gpu0"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
    "${old_teacher[@]}" --data "$teacher" --source-weight "3:${weight}" \
    --opening-data data/master-opening-positions.jsonl --opening-samples 1200 \
    --policy-target "$target" --mirror-augmentation \
    --channels 64 --blocks 4 --epochs 14 --patience 3 --batch 128 \
    --lr 0.00005 --seed 20260930 --device cuda --init-model "$base" \
    --output "models/${name}.pt"

  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" "${old_teacher[@]}" --data "$teacher" --test-source-index 3 \
    --phase middlegame --device cuda \
    > "reports/${name}-new-teacher.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" "${old_teacher[@]}" --data "$teacher" \
    --test-source-index 0 --test-source-index 1 --test-source-index 2 --phase middlegame --device cuda \
    > "reports/${name}-independent.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
    --model-a "$base" --model-b "models/${name}.pt" "${old_teacher[@]}" --data "$teacher" \
    --test-source-index 3 --phase middlegame --device cuda \
    > "reports/compare-${name}-new-teacher.json"
done
