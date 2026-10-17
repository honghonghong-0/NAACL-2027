cd ./pepper_ms
P=A-mem-paper-repro/.venv-paper/bin/python
sed -e "s#research-notes/pepper-ms-dev#research-notes/pepper-ms-odkg-100#" build_ms_inputs.py > build_ms_inputs_100.py
sed -e "s#pepper-ms-dev#pepper-ms-odkg-100#" build_ms_v1.py > build_ms_v1_100.py
$P build_ms_inputs_100.py 100 | head -3
$P build_ms_v1_100.py
# launcher for N users (no ==10 assert), output tag per shard
sed -e "s/assert len(users) == 10/assert len(users) >= 1/" -e "s/PREFLIGHT users=10/PREFLIGHT users=%d/" -e "s/TOTAL_CAP = 1.50/TOTAL_CAP = float(os.environ.get('E2E_CAP', '1.50'))/" launch_e2e_10x20.py > launch_e2e_n.py
grep -n "PREFLIGHT\|TOTAL_CAP =\|assert len(users)" launch_e2e_n.py
