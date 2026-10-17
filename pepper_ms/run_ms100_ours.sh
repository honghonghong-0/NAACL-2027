#!/bin/bash
cd ./pepper_ms
P=A-mem-paper-repro/.venv-paper/bin/python
for c in ours allmem; do for s in s2keep-v1 s2shift-v1; do for i in 0 1 2 3 4; do
  E2E_CAP=0.80 E2E_CONDS=$c E2E_INPUT=research-notes/pepper-ms-odkg-100/users-$s-shard$i.json nice -n 5 $P launch_e2e_ours.py ms100-$s-shard$i-$c > logs_ms100/$c-$s-shard$i.log 2>&1 &
done; done; done
wait; echo MS100_OURS_DONE
