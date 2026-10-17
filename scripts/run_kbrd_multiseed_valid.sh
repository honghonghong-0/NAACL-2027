#!/usr/bin/env bash
set -euo pipefail

ROOT=.
CRSLAB="$ROOT/baselines/crslab"
PY=python
GPU="${1:-1}"

for config in "$ROOT"/baselines/kbrd_configs/redial_ms_s*.yaml; do
  cp "$config" "$CRSLAB/config/crs/kbrd/"
done
cd "$CRSLAB"
declare -a pids
for seed in 0 1 2; do
  config="config/crs/kbrd/redial_ms_s${seed}.yaml"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_crslab.py -c "$config" -g "$GPU" -ss \
    > "../kbrd_ms_s${seed}.log" 2>&1 &
  pids[$seed]=$!
done

failed=0
for seed in 0 1 2; do
  if ! wait "${pids[$seed]}"; then
    echo "KBRD seed ${seed} failed; see baselines/kbrd_ms_s${seed}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

for seed in 0 1 2; do
  config="config/crs/kbrd/redial_ms_s${seed}.yaml"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" ../../scripts/crslab_infer.py \
    "$config" valid "../scores_kbrd_valid_s${seed}" \
    > "../kbrd_infer_valid_s${seed}.log" 2>&1 &
  pids[$seed]=$!
done
for seed in 0 1 2; do wait "${pids[$seed]}"; done

cd "$ROOT"
for seed in 0 1 2; do
  python scripts/eval_poola.py \
    valid "kbrd_s${seed}=baselines/scores_kbrd_valid_s${seed}.npz" \
    > "baselines/kbrd_poola_valid_s${seed}.log" 2>&1
done
