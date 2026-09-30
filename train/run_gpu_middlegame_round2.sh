#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
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

teacher_args=(
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --data data/teacher-ccpd-competition-1200.jsonl
  --data "$teacher"
)

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  "${teacher_args[@]}" --source-weight 3:12 --validation-source-index 3 \
  --opening-data data/master-opening-positions.jsonl --opening-samples 800 \
  --policy-target best --mirror-augmentation --channels 64 --blocks 4 \
  --epochs 30 --patience 5 --batch 128 --lr 0.0002 --seed 20260930 \
  --device cuda --init-model "$base" \
  --output models/choice-middle-best-focused-64x4-gpu0.pt

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  "${teacher_args[@]}" --source-weight 3:6 --validation-source-index 3 \
  --opening-data data/master-opening-positions.jsonl --opening-samples 1200 \
  --policy-target best --mirror-augmentation --channels 128 --blocks 8 \
  --epochs 30 --patience 5 --batch 128 --lr 0.0005 --seed 20260930 \
  --device cuda --output models/choice-middle-best-focused-128x8-gpu0.pt

for name in choice-middle-best-focused-64x4-gpu0 choice-middle-best-focused-128x8-gpu0; do
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" "${teacher_args[@]}" --test-source-index 3 \
    --phase middlegame --seed 20260930 --device cuda \
    > "reports/${name}-new-teacher.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" \
    --data data/teacher-openings-5000.jsonl --data data/teacher-games-160.jsonl \
    --data data/teacher-ccpd-competition-1200.jsonl --data "$teacher" \
    --test-source-index 0 --test-source-index 1 --test-source-index 2 \
    --phase middlegame --seed 20260930 --device cuda > "reports/${name}-independent.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
    --model-a "$base" --model-b "models/${name}.pt" "${teacher_args[@]}" \
    --test-source-index 3 --phase middlegame --seed 20260930 --device cuda \
    > "reports/compare-${name}-new-teacher.json"
done
