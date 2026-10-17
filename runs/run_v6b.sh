export MEM_FILE=relational_memory_v6.json EXT_FILE=extract_v6/compat_v5.jsonl ALPHA_FILE=alpha_v6.json
until [ -f processed/item_chain/rho_valid_v6.npz ]; do sleep 30; done
export CACHE_FILE=rho_valid_v6b.npz
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 python"
$PY scripts/affordance_read_v2.py > baselines/log_chain_cache_v6b.txt 2>&1 || exit 1
for b in "cooc cooc" "kbrd_s0 baselines/scores_kbrd_valid_s0.npz" "sasrec_s0 baselines/scores_sasrec_valid_s0.npz" "realize_nochain baselines/scores_realize_v2_base+hist+utter_s0_valid.npz"; do $PY scripts/residual_fuse.py fuse $b; done > baselines/log_residual_fuse_v6b_valid.txt 2>&1
echo DONE >> baselines/log_residual_fuse_v6b_valid.txt
