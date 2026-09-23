#!/usr/bin/env bash
set -euo pipefail

train_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$train_root"

# This run is assigned to physical GPU 0 on dash-dev-191-root. The other card
# stays invisible to the Python process.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python3 -u train/choice_model.py train \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --opening-data data/opening-positions.jsonl \
  --opening-samples 2000 \
  --output models/choice-21k-gpu0.pt \
  --epochs 12 --batch 256 --channels 64 --blocks 4 \
  --seed 20260923 --device cuda
