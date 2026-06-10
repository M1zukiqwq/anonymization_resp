#!/bin/bash
# Resident-halo ablation, SIZE SWEEP: ON vs OFF (--mpi-no-resident-halo) at
# 1024^2/500, 2048^2/100, 4096^2/100, 8192^2/100, ranks 16/32/64/128, <=2 in queue.
#
# Full synchronization (OFF) moves O(N^2) per step (root-funnelled gather/scatter of the
# whole tensor) while the resident path (ON) exchanges only the O(N) boundary, so the
# OFF/ON slowdown GROWS with grid size:  geomean 70x -> 73x -> 229x -> 336x.
# The headline 63-76x (ablation_driver2.sh, 1024^2) is thus a CONSERVATIVE FLOOR; at the
# benchmark's 8192^2 size the optimization is worth ~340x.
#
# The ratio is step-count-independent (ON and OFF are both linear in steps), so the larger
# grids use fewer steps to keep the OFF wall-time tractable (8192^2/5000 OFF would be ~16 h).
# Builds every variant from the hand-named 1024^2/500 sources (src/stencil.halo_{on,off}.named.cpp).
set -u
Z=$REPRO_ROOT
PAPER=$Z/paper; A=$PAPER/abl
source $Z/setenv.sh
RT=$Z/multi-node-test/runtime
CF="-std=c++17 -O2 -fsycl -I$RT -I/usr/sw/mpi/mpi_20230630_SEA/include"
LF="-L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread"
COMMON="-q q_share -b -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128"
PROG=$A/sizesweep.progress
qcount() { bjobs -u $USER 2>/dev/null | grep -c q_share; }

echo "SIZESWEEP START $(date)" > "$PROG"
# build per-size ON/OFF binaries from the named 1024^2/500 sources
for sz in "1024 500" "2048 100" "4096 100" "8192 100"; do
  set -- $sz; N=$1; STEPS=$2
  for v in on off; do
    src=$A/stencil.halo_${v}.sweep${N}.cpp
    sed -e "s/const int NX = 1024;/const int NX = ${N};/" \
        -e "s/const int NY = 1024;/const int NY = ${N};/" \
        -e "s/const int TIME_STEPS = 500;/const int TIME_STEPS = ${STEPS};/" \
        "$A/stencil.halo_${v}.named.cpp" > "$src"
    d=$A/stencil_${v}_${N}; mkdir -p "$d"
    if swsycl $CF -c "$src" -o "$d/o.o" > "$d/c.log" 2>&1 && \
       ( cd "$d" && swsycl -fsycl o.o -o a.out $LF ) >> "$d/c.log" 2>&1; then
      echo "compiled $v N=$N steps=$STEPS" >> "$PROG"
    else
      echo "COMPILE FAIL $v N=$N: $(tail -2 "$d/c.log" | tr '\n' ' ')" >> "$PROG"
    fi
  done
done
# submit ON/OFF at each rank count, throttled to <=2 jobs
for N in 1024 2048 4096 8192; do
  for v in on off; do
    d=$A/stencil_${v}_${N}
    [ -x "$d/a.out" ] || { echo "[skip] no a.out $v $N" >> "$PROG"; continue; }
    for cfg in "4 16" "8 32" "16 64" "32 128"; do
      set -- $cfg; m=$1; n=$2
      while [ "$(qcount)" -ge 2 ]; do sleep 15; done
      ( cd "$d" && bsub $COMMON -m "$m" -n "$n" -o "run_m${m}_n${n}.log" ./a.out ) >/dev/null 2>&1
      echo "[$(date +%H:%M:%S)] submitted $v $N m$m n$n" >> "$PROG"
    done
  done
done
while [ "$(qcount)" -gt 0 ]; do sleep 20; done
echo "SIZESWEEP DONE $(date)" >> "$PROG"
