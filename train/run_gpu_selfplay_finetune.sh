#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
selfplay_data="${SELFPLAY_DATA:-data/selfplay-round1-positions.jsonl}"
base="models/choice-master-dominant-mirror-64x4-gpu0.pt"
model="choice-selfplay-round1-finetune-gpu0"
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$selfplay_data" "$base"; do
  if [[ ! -s "$file" ]]; then
    echo "Training input is missing: $file" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --data data/teacher-ccpd-competition-1200.jsonl \
  --data "$selfplay_data" --source-weight 3:0.10 \
  --opening-data data/master-opening-positions.jsonl --opening-samples 1800 \
  --policy-target dominant --mirror-augmentation \
  --channels 64 --blocks 4 --epochs 8 --patience 2 --batch 128 \
  --lr 0.0001 --seed 20260923 --device cuda --init-model "$base" \
  --output "models/${model}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${model}.pt" \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --data data/teacher-ccpd-competition-1200.jsonl \
  --device cuda > "reports/${model}-teacher-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${model}.pt" --data "$selfplay_data" --device cuda \
  > "reports/${model}-selfplay-test.json"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_master_openings.py \
  --teacher data/teacher-openings-5000.jsonl \
  --teacher data/teacher-games-160.jsonl \
  --teacher data/teacher-ccpd-competition-1200.jsonl \
  --book data/master-opening-positions.jsonl \
  --model "$base" --model "models/${model}.pt" \
  --exclude data/opening-positions.jsonl --opening-samples 1500 --device cuda \
  --output reports/compare-selfplay-round1-finetune-gpu0.json
