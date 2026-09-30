#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
node_bin="${NODE_BIN:-node}"
selfplay_teacher="${SELFPLAY_TEACHER:-data/teacher-selfplay-middle-2400-pikafish.jsonl}"
ccpd_teacher="${CCPD_TEACHER:-data/teacher-ccpd-middle-4000-pikafish.jsonl}"
base="${BASE_MODEL:-models/choice-selfplay-round1-finetune-gpu0.pt}"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$selfplay_teacher" "$ccpd_teacher" "$base"; do
  if [[ ! -s "$file" ]]; then
    echo "Training input is missing: $file" >&2
    exit 1
  fi
done
"$node_bin" verify-data.js "$selfplay_teacher"
"$node_bin" verify-data.js "$ccpd_teacher"

teacher_args=(
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --data "$selfplay_teacher"
  --data "$ccpd_teacher"
)
name="choice-middle-diverse-128x8-gpu0"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
  "${teacher_args[@]}" --source-weight 2:3 --source-weight 3:8 \
  --validation-source-index 3 \
  --opening-data data/master-opening-positions.jsonl --opening-samples 1200 \
  --policy-target best --mirror-augmentation --selection-metric top1 \
  --channels 128 --blocks 8 --epochs 50 --patience 10 --batch 128 \
  --lr 0.0003 --seed 20260930 --device cuda \
  --output "models/${name}.pt"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" "${teacher_args[@]}" --test-source-index 3 \
  --phase middlegame --seed 20260930 --device cuda \
  > "reports/${name}-ccpd-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "models/${name}.pt" "${teacher_args[@]}" \
  --test-source-index 0 --test-source-index 1 --test-source-index 2 \
  --phase middlegame --seed 20260930 --device cuda \
  > "reports/${name}-independent-test.json"
CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
  --model-a "$base" --model-b "models/${name}.pt" "${teacher_args[@]}" \
  --test-source-index 3 --phase middlegame --seed 20260930 --device cuda \
  > "reports/compare-${name}-ccpd-test.json"
