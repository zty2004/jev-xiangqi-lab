#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
selfplay_data="${SELFPLAY_DATA:-data/selfplay-round1-positions.jsonl}"
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
if [[ ! -s "$selfplay_data" ]]; then
  echo "Self-play data is missing: $selfplay_data" >&2
  exit 1
fi

models=(choice-master-dominant-mirror-64x4-gpu0)
for spec in w10:0.10 w20:0.20; do
  label="${spec%%:*}"
  weight="${spec#*:}"
  model="choice-selfplay-round1-${label}-gpu0"
  models+=("$model")
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
    --data data/teacher-openings-5000.jsonl \
    --data data/teacher-games-160.jsonl \
    --data data/teacher-ccpd-competition-1200.jsonl \
    --data "$selfplay_data" \
    --source-weight "3:${weight}" \
    --opening-data data/master-opening-positions.jsonl \
    --opening-samples 1800 --policy-target dominant --mirror-augmentation \
    --channels 64 --blocks 4 --epochs 12 --patience 3 --batch 128 \
    --lr 0.001 --seed 20260923 --device cuda \
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
done

compare_args=()
for model in "${models[@]}"; do
  compare_args+=(--model "models/${model}.pt")
done
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_master_openings.py \
  --teacher data/teacher-openings-5000.jsonl \
  --teacher data/teacher-games-160.jsonl \
  --teacher data/teacher-ccpd-competition-1200.jsonl \
  --book data/master-opening-positions.jsonl \
  "${compare_args[@]}" \
  --exclude data/opening-positions.jsonl --opening-samples 1500 --device cuda \
  --output reports/compare-selfplay-round1-weight-sweep-gpu0.json
