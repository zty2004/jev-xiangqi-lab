#!/usr/bin/env bash
set -euo pipefail

gpu_uuid='GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24'
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU 0 UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi

common=(
  train/choice_model.py train
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --opening-data data/master-opening-positions.jsonl
  --opening-samples 1500
  --policy-target dominant
  --epochs 10
  --patience 3
  --batch 128
  --device cuda
  --seed 20260923
)

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u "${common[@]}" \
  --channels 64 --blocks 4 --lr 0.001 \
  --output models/choice-master-dominant-64x4-gpu0.pt

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u "${common[@]}" \
  --channels 64 --blocks 4 --lr 0.001 --mirror-augmentation \
  --output models/choice-master-dominant-mirror-64x4-gpu0.pt

for model in \
  choice-master-opening-gpu0 \
  choice-master-dominant-64x4-gpu0 \
  choice-master-dominant-mirror-64x4-gpu0
do
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${model}.pt" \
    --data data/teacher-openings-5000.jsonl \
    --data data/teacher-games-160.jsonl \
    --device cuda > "reports/${model}-teacher-test.json"
done

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_master_openings.py \
  --teacher data/teacher-openings-5000.jsonl \
  --teacher data/teacher-games-160.jsonl \
  --book data/master-opening-positions.jsonl \
  --model models/choice-master-opening-gpu0.pt \
  --model models/choice-master-dominant-64x4-gpu0.pt \
  --model models/choice-master-dominant-mirror-64x4-gpu0.pt \
  --exclude data/opening-positions.jsonl \
  --opening-samples 1500 --device cuda \
  --output reports/compare-master-round2-gpu0.json
