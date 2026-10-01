#!/usr/bin/env bash
set -euo pipefail

pikafish="${PIKAFISH_PATH:?set PIKAFISH_PATH to the native Pikafish binary}"
source_data="${PV_TEACHER:-data/teacher-ccpd-all-middle-20k-pv250k.jsonl}"
seed_data="${TREE_SEED:-data/teacher-ccpd-middle-tree-r12-pilot-seed.jsonl}"
labelled_data="${TREE_LABELLED:-data/teacher-ccpd-middle-tree-r12-pilot-labelled.jsonl}"
roots="${TREE_ROOTS:-300}"
candidates="${TREE_CANDIDATES:-4}"
plies="${TREE_PLIES:-4}"
nodes="${TEACHER_NODES:-100000}"
shards="${TEACHER_SHARDS:-32}"
seed="${TREE_RANDOM_SEED:-20261005}"

if [[ ! -x "$pikafish" || ! -s "$source_data" ]]; then
  echo "Pikafish or PV teacher data is unavailable" >&2
  exit 1
fi
if (( roots < 1 || candidates < 1 || plies < 1 || nodes < 1 || shards < 1 )); then
  echo "Tree relabel settings must be positive" >&2
  exit 1
fi

if [[ ! -s "$seed_data" ]]; then
  python3 -m train.sample_tree_distillation \
    --input "$source_data" --output "$seed_data" --phase middlegame \
    --roots "$roots" --candidates "$candidates" --plies "$plies" --seed "$seed"
fi

pids=()
shard_files=()
for ((index = 0; index < shards; index++)); do
  shard="${labelled_data%.jsonl}.shard-${index}-of-${shards}.jsonl"
  log="${shard%.jsonl}.log"
  shard_files+=("$shard")
  resume=()
  if [[ -s "$shard" ]]; then
    resume=(--resume)
  elif [[ -e "$shard" ]]; then
    rm -f "$shard"
  fi
  python3 -u train/relabel_pikafish_python.py \
    --input "$seed_data" --output "$shard" --pikafish "$pikafish" \
    --nodes "$nodes" --multipv 8 --hash 64 \
    --shard-index "$index" --shard-count "$shards" "${resume[@]}" \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one Pikafish relabel shard failed" >&2
  exit 1
fi

if [[ ! -s "$labelled_data" ]]; then
  combine_args=()
  for shard in "${shard_files[@]}"; do
    combine_args+=(--input "$shard")
  done
  python3 train/combine_relabel_shards.py "${combine_args[@]}" --output "$labelled_data"
fi

python3 - "$labelled_data" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
digest = hashlib.sha256()
positions = 0
with open(path, "rb") as handle:
    for line in handle:
        digest.update(line)
        if json.loads(line).get("kind") == "position":
            positions += 1
print(json.dumps({"path": path, "positions": positions, "sha256": digest.hexdigest()}))
PY
