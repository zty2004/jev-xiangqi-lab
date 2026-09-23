#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24
export OMP_NUM_THREADS=8
python3 -u train/choice_model.py train \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --data data/teacher-ccpd-competition-1200.jsonl \
  --opening-data data/opening-positions.jsonl --opening-samples 2000 \
  --output models/choice-ccpd1200-gpu0.pt \
  --policy-target dominant --epochs 12 --patience 3 \
  --batch 256 --channels 64 --blocks 4 \
  --seed 20260923 --device cuda
