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
  --channels 96 --blocks 6 --lr 0.0005 \
  --output models/choice-master-dominant-96x6-gpu0.pt
