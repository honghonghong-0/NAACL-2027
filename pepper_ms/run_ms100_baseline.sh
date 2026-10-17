#!/bin/bash
cd ./pepper_ms
P=A-mem-paper-repro/.venv-paper/bin/python
for s in s1 s2keep-v1 s2shift-v1; do for i in 0 1 2 3 4; do
  E2E_CAP=0.80 E2E_CONDS=baseline E2E_INPUT=research-notes/pepper-ms-odkg-100/users-$s-shard$i.json $P launch_e2e_n.py ms100-$s-shard$i > logs_ms100/$s-shard$i.log 2>&1 &
done; done
wait; echo MS100_BASELINE_DONE
