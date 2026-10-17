#!/usr/bin/env bash
set -euo pipefail

ROOT=.
PY=python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$ROOT"
for seed in 0 1 2; do
  "$PY" scripts/sasrec_poola.py \
    --seed "$seed" \
    --epochs 40 \
    --eval-every 5 \
    --learning-rate 1e-3 \
    --output "baselines/scores_sasrec_valid_s${seed}.npz"
  "$PY" scripts/eval_poola.py valid \
    --external-only \
    --output "baselines/results_sasrec_valid_s${seed}.json" \
    "sasrec_s${seed}=baselines/scores_sasrec_valid_s${seed}.npz" \
    > "baselines/eval_sasrec_valid_s${seed}.log"
done
