#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Bind to the selected physical card by UUID so another card remains invisible.
export CUDA_VISIBLE_DEVICES=GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24
export OMP_NUM_THREADS=8
python3 -u train/choice_model.py train \
  --data data/teacher-games-160.jsonl \
  --history-data data/history-labels-160.jsonl \
  --output models/choice-history-wdl-gpu0.pt \
  --value-head wdl --policy-target best \
  --epochs 12 --patience 3 --batch 256 --lr 0.0005 \
  --channels 64 --blocks 4 --seed 20260923 --device cuda
