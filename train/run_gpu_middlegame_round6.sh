#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${GPU_UUID:-GPU-44af9bac-81e9-4c9b-05f6-3d8207561c24}"
node_bin="${NODE_BIN:-node}"
teacher="${MIDDLE_TEACHER:-data/teacher-ccpd-all-middle-20k-nodes250k.jsonl}"
base="${BASE_MODEL:-models/choice-selfplay-round1-finetune-gpu0.pt}"

if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
  echo "Required GPU UUID is unavailable: $gpu_uuid" >&2
  exit 1
fi
for file in "$teacher" "$base"; do
  if [[ ! -s "$file" ]]; then
    echo "Training input is missing: $file" >&2
    exit 1
  fi
done
"$node_bin" verify-data.js "$teacher"

CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
  --model "$base" --data "$teacher" --phase middlegame --seed 20260930 --device cuda \
  > reports/choice-middle-fixednodes-baseline.json

train_candidate() {
  local name="$1" channels="$2" blocks="$3" features="$4" batch="$5" lr="$6"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 -u train/choice_model.py train \
    --data "$teacher" --policy-target best --value-loss-weight 0 \
    --mirror-augmentation --selection-metric top1 --policy-features "$features" \
    --channels "$channels" --blocks "$blocks" --epochs 80 --patience 12 \
    --batch "$batch" --lr "$lr" --seed 20260930 --device cuda \
    --output "models/${name}.pt"

  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" --data "$teacher" --phase middlegame \
    --seed 20260930 --device cuda > "reports/${name}-fixednodes-test.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/compare_choice_models.py \
    --model-a "$base" --model-b "models/${name}.pt" --data "$teacher" \
    --phase middlegame --seed 20260930 --device cuda \
    > "reports/compare-${name}-fixednodes-test.json"
}

train_candidate choice-middle-fixednodes-128x8-v1-gpu0 128 8 v1 256 0.0005
train_candidate choice-middle-fixednodes-192x10-v2-gpu0 192 10 v2 192 0.0003

old_teacher=(
  --data data/teacher-openings-5000.jsonl
  --data data/teacher-games-160.jsonl
  --data data/teacher-selfplay-middle-2400-pikafish.jsonl
  --data data/teacher-ccpd-middle-4000-pikafish.jsonl
)
for name in choice-middle-fixednodes-128x8-v1-gpu0 choice-middle-fixednodes-192x10-v2-gpu0; do
  CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 train/choice_model.py evaluate \
    --model "models/${name}.pt" "${old_teacher[@]}" \
    --phase middlegame --seed 20260930 --device cuda \
    > "reports/${name}-prior-independent-test.json"
done
