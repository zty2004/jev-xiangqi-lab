#!/usr/bin/env bash
set -euo pipefail

pikafish="${PIKAFISH_PATH:?set PIKAFISH_PATH to the native Pikafish binary}"
input="${PV_ROOT_DATA:-data/teacher-ccpd-all-middle-20k-pv250k.jsonl}"
output="${PV_EXPANDED_DATA:-data/teacher-ccpd-all-middle-20k-pvexpanded8.jsonl}"
shards="${PV_SHARDS:-4}"
weight="${PV_CONTINUATION_WEIGHT:-0.10}"

if [[ ! -x "$pikafish" || ! -s "$input" ]]; then
  echo "Pikafish or PV root data is unavailable" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "Output exists: $output" >&2
  exit 1
fi

shard_files=()
pids=()
for ((index = 0; index < shards; index++)); do
  shard="${output%.jsonl}.shard-${index}-of-${shards}.jsonl"
  if [[ -e "$shard" ]]; then
    echo "Shard exists: $shard" >&2
    exit 1
  fi
  shard_files+=("$shard")
  python3 -u train/expand_pv_training.py \
    --input "$input" --output "$shard" --pikafish "$pikafish" \
    --max-pv-plies 8 --continuation-weight "$weight" \
    --shard-index "$index" --shard-count "$shards" \
    > "${shard%.jsonl}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if ((failed)); then
  echo "At least one PV expansion shard failed" >&2
  exit 1
fi

combine_args=()
for shard in "${shard_files[@]}"; do combine_args+=(--input "$shard"); done
python3 train/combine_expanded_pv_shards.py "${combine_args[@]}" --output "$output"
sha256sum "$output"
