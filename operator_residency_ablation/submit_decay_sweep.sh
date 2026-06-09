#!/bin/bash
# Operator-residency ablation: submit decay ON/OFF over a rank sweep, paper-style.
# Run ON the the cluster login node, from the directory holding the two binaries:
#   decay.OR_on.out  (operator residency enabled)
#   decay.OR_off.out (--mpi-no-operator-resident; legacy gather/scatter)
#
#   source $Z/setenv.sh   (so LD_LIBRARY_PATH reaches the compute nodes)
#   MAX_JOBS=2 ./submit_decay_sweep.sh            # full sweep, <=2 jobs queued
#   CONFIGS="16 64" MAX_JOBS=2 ./submit_decay_sweep.sh   # one config (m=16,n=64)
#
# Constraints: -m <= 32 nodes, -n <= 4*m ranks (<=128). Shared q_share queue.
set -u

Z=$REPRO_ROOT
P=$Z/paper/ablperf

source "$Z/setenv.sh"
cd "$P"

COMMON="-q q_share -b -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128"
configs="${CONFIGS:-4 16;8 32;16 64;32 128}"
modes="${MODES:-on off}"
summary="$P/decay.sweep.log"
max_jobs="${MAX_JOBS:-2}"

qcount() { bjobs -u $USER 2>/dev/null | grep -c q_share; }

echo "[sweep-paperstyle] start $(date)" | tee -a "$summary"

old_ifs=$IFS
IFS=';'
for cfg in $configs; do
  IFS=$old_ifs
  set -- $cfg
  m=$1
  n=$2
  for mode in $modes; do
    case "$mode" in
      on) bin=./decay.OR_on.out ;;
      off) bin=./decay.OR_off.out ;;
      *) echo "[sweep-paperstyle] unknown mode=$mode" | tee -a "$summary"; continue ;;
    esac
    [ -x "$bin" ] || { echo "[sweep-paperstyle] missing $bin" | tee -a "$summary"; continue; }
    while [ "$(qcount)" -ge "$max_jobs" ]; do sleep 15; done
    log="decay.OR_${mode}.run_m${m}_n${n}.log"
    ( cd "$P" && bsub $COMMON -m "$m" -n "$n" -o "$log" "$bin" ) >/dev/null 2>&1
    echo "[$(date +%H:%M:%S)] submitted mode=$mode m$m n$n log=$log q=$(qcount)" | tee -a "$summary"
    sleep 5
  done
  IFS=';'
done
IFS=$old_ifs

while [ "$(qcount)" -gt 0 ]; do sleep 20; done
echo "[sweep-paperstyle] finish $(date)" | tee -a "$summary"
