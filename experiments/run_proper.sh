#!/bin/zsh
# Global rerun pool: all four job groups run concurrently through one
# fixed-width worker pool (fine-grained shards -> good load balance, short tail).
#   1. mega PMR rerun      (post-TriComp code; NUTS cells stay cached)  24 shards
#   2. pdb  PMR rerun      (8 seeds, gold-judged)                       12 shards
#   3. pdb  SOTA panel     (8 methods x 37 posteriors x 4 seeds)        24 shards
#   4. mega SOTA panel     (8 methods x 102 targets x 4 seeds)          36 shards
# Every cell is cached in its shard file, so this script is safely re-runnable.
cd /Users/bsoonjun/Documents/GitHub/dataglass/dgbe/tools/pmr-hmc-lab
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# XLA spawns its own intra-op pool per process (ignores the BLAS caps) -> load
# hit ~100 with 14 workers on first launch; cap it to one thread per worker
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=/Users/bsoonjun/Documents/GitHub/dataglass/dgbe/vilya/.venv/bin/python
P=${POOL_WIDTH:-14}
mkdir -p logs
: > proper_cmds.txt
for i in $(seq 0 11); do echo "MEGA_TARGETS_FILE=mega30.txt $PY mega_bench.py $i 12 >> logs/megare_s$i.log 2>&1" >> proper_cmds.txt; done
for i in $(seq 0 11); do echo "$PY pdb_rerun.py $i 12 >> logs/pdbre_s$i.log 2>&1" >> proper_cmds.txt; done
for i in $(seq 0 23); do echo "$PY panel_bench.py pdb $i 24 >> logs/pdbpanel_s$i.log 2>&1" >> proper_cmds.txt; done
for i in $(seq 0 11); do echo "PANEL_TARGETS_FILE=mega30.txt $PY panel_bench.py mega $i 12 >> logs/megapanel_s$i.log 2>&1" >> proper_cmds.txt; done
echo "pool start $(date) width=$P jobs=$(wc -l < proper_cmds.txt)"
xargs -P $P -I CMD nice -n 5 zsh -c CMD < proper_cmds.txt
echo "pool done $(date)"
