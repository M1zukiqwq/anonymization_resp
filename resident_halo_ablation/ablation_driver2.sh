#!/bin/bash
# Resident-halo ablation v2: named-kernel ON/OFF variants (1024^2/500), ranks 16/32/64/128, <=2 in queue.
set -u
Z=$REPRO_ROOT
PAPER=$Z/paper; A=$PAPER/abl
source $Z/setenv.sh
RT=$Z/multi-node-test/runtime
CF="-std=c++17 -O2 -fsycl -I$RT -I/usr/sw/mpi/mpi_20230630_SEA/include"
LF="-L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread"
COMMON="-q q_share -b -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128"
PROG=$A/abl2.progress
qcount() { bjobs -u $USER 2>/dev/null | grep -c q_share; }

echo "ABL2 START $(date)" > "$PROG"
for v in on off; do
  d=$A/stencil_${v}n; mkdir -p "$d"
  if swsycl $CF -c "$A/stencil.halo_${v}.named.cpp" -o "$d/o.o" > "$d/c.log" 2>&1 && \
     ( cd "$d" && swsycl -fsycl o.o -o a.out $LF ) >> "$d/c.log" 2>&1; then
    echo "compiled $v" >> "$PROG"
  else
    echo "COMPILE FAIL $v: $(tail -2 "$d/c.log" | tr '\n' ' ')" >> "$PROG"
  fi
done
for v in on off; do
  for cfg in "4 16" "8 32" "16 64" "32 128"; do
    set -- $cfg; m=$1; n=$2; d=$A/stencil_${v}n
    [ -x "$d/a.out" ] || { echo "[skip] no a.out $v" >> "$PROG"; continue; }
    while [ "$(qcount)" -ge 2 ]; do sleep 15; done
    ( cd "$d" && bsub $COMMON -m "$m" -n "$n" -o "run_m${m}_n${n}.log" ./a.out ) >/dev/null 2>&1
    echo "[$(date +%H:%M:%S)] submitted $v m$m n$n" >> "$PROG"
  done
done
while [ "$(qcount)" -gt 0 ]; do sleep 20; done
echo "ABL2 DONE $(date)" >> "$PROG"
