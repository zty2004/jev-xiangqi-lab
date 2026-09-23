#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24
export OMP_NUM_THREADS=8
python3 -u train/choice_model.py train \
  --data data/teacher-openings-5000.jsonl \
  --data data/teacher-games-160.jsonl \
  --history-data data/history-labels-160.jsonl \
  --opening-data data/opening-positions.jsonl --opening-samples 2000 \
  --output models/choice-history-wdl-mixed-gpu0.pt \
  --value-head wdl --policy-target soft \
  --epochs 12 --patience 3 --batch 256 --lr 0.0005 \
  --channels 64 --blocks 4 --seed 20260923 --device cuda
