#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/nnue_value.py \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --data data/teacher-ccpd-competition-1200.jsonl \
  --output models/value-nnue-residual-128-gpu.json \
  --checkpoint models/value-nnue-residual-128-gpu.pt \
  --hidden 128 --head 32 --epochs 40 --patience 5 --batch 2048 \
  --lr 0.001 --seed 20260924 --device cuda --residual

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_value_evaluators.py \
  --model models/value-nnue-residual-128-gpu.pt \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --data data/teacher-ccpd-competition-1200.jsonl \
  --output reports/value-nnue-residual-128-gpu.json \
  --seed 20260924 --device cuda --residual
