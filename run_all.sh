#!/usr/bin/env bash
# Run the lowrank_selftrain method over FG10, split across 5 GPUs (2 datasets
# each), then merge the shards into one results row. Edit SHARDS / GPU count
# to taste. Requires the 'adapt' conda env (numpy+torch+tqdm, CUDA).
#
# If the feature caches are missing, set DATA_ROOT to the directory containing
# the 10 dataset folders and they will be extracted on the fly:
#   DATA_ROOT=/path/to/datasets ./run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
cd "$HERE"
mkdir -p logs

declare -A SHARDS=(
  [0]="fgvc_aircraft caltech101"
  [1]="stanford_cars dtd"
  [2]="eurosat oxford_flowers"
  [3]="food101 oxford_pets"
  [4]="sun397 ucf101"
)

ROOT_ARGS=()
if [ -n "${DATA_ROOT:-}" ]; then
  ROOT_ARGS=(--data_root "$DATA_ROOT")
  echo "[run_all] DATA_ROOT=$DATA_ROOT (auto-extract missing caches)"
fi

PIDS=()
for gpu in "${!SHARDS[@]}"; do
  echo "[run_all] gpu${gpu}: ${SHARDS[$gpu]}"
  $PY fg10_online.py --gpu "$gpu" --datasets ${SHARDS[$gpu]} --tag "shard${gpu}" \
      "${ROOT_ARGS[@]}" > "logs/shard${gpu}.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "[run_all] all shards done, aggregating..."
$PY aggregate.py
