#!/bin/bash
# Full-grid correctness by RANK-INVARIANCE.
#
# For each stencil benchmark we inject (injector.py) a root-side FNV-1a fingerprint over
# the WHOLE materialized grid in canonical global order, plus sum/min/max, right after the
# materialize call. We then run the SAME binary at 1/16/64/128 ranks:
#   - 1 rank  performs NO inter-rank halo exchange  => the sequential reference;
#   - 16/64/128 ranks exercise the inferred halo exchange.
# If the fingerprint is bit-identical across all rank counts, the inferred communication
# reproduced every one of the N*N elements exactly (no value read from a partition it was
# not written under). A small 512^2/100 grid suffices (correctness is size-independent).
#
# Sources (cluster working tree):
#   stencil (2D direct neighbour-copy halo): the resident-halo ON variant,
#            ../resident_halo_ablation/src/stencil.halo_on.named.cpp  (placed at $A/ on the cluster)
#   wave    (2D read-cache buffer-swap halo): the standard `--mpi` translation of
#            example/waveEquation (harnessed test source under $M/src/)
# Result: stencil + wave both bit-identical across ranks (see correctness_fingerprints.txt).
set -u
Z=$REPRO_ROOT
PAPER=$Z/paper; A=$PAPER/abl; M=$Z/multi-node-test; B=$PAPER/correctness
source $Z/setenv.sh
RT=$M/runtime
CF="-std=c++17 -O2 -fsycl -I$RT -I/usr/sw/mpi/mpi_20230630_SEA/include"
LF="-L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread"
COMMON="-q q_share -b -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"; PROG=$B/correctness.progress
qcount() { bjobs -u $USER 2>/dev/null | grep -c q_share; }
echo "CORRECTNESS START $(date)" > "$PROG"

# tag | source | final materialized tensor | materialize-call substring (main, not the wrapper)
prep() {
  local tag=$1 src=$2 tv=$3 sub=$4
  local dst=$B/$tag.chk.cpp
  cp "$src" "$dst"
  sed -i -e 's/\(const\|constexpr\) int NX = [0-9]*;/\1 int NX = 512;/' \
         -e 's/\(const\|constexpr\) int NY = [0-9]*;/\1 int NY = 512;/' \
         -e 's/\(const\|constexpr\) int TIME_STEPS = [0-9]*;/\1 int TIME_STEPS = 100;/' "$dst"
  python "$HERE/injector.py" "$dst" "$sub" "$tv" 2>>"$PROG" || { echo "INJECT FAIL $tag" >> "$PROG"; return; }
  local d=$B/$tag; mkdir -p "$d"
  if swsycl $CF -c "$dst" -o "$d/o.o" > "$d/c.log" 2>&1 && ( cd "$d" && swsycl -fsycl o.o -o a.out $LF ) >> "$d/c.log" 2>&1; then
    echo "compiled $tag" >> "$PROG"
  else
    echo "COMPILE FAIL $tag: $(tail -3 "$d/c.log" | tr '\n' ' ')" >> "$PROG"
  fi
}

prep stencil "$A/stencil.halo_on.named.cpp" "matIn" "__dacpp_mpi_or_stencilShell_stencil_0_materialize(__dacpp_mpi_or_ctx_0"
prep wave    "$M/src/waveEquation.mpi.dac_sycl_buffer_test.cpp" "matCur" "__dacpp_mpi_or_waveEqShell_waveEq_0_materialize(__dacpp_mpi_or_ctx_0"

for tag in stencil wave; do
  d=$B/$tag; [ -x "$d/a.out" ] || { echo "[skip] $tag no a.out" >> "$PROG"; continue; }
  for cfg in "1 1" "4 16" "16 64" "32 128"; do
    set -- $cfg; m=$1; n=$2
    while [ "$(qcount)" -ge 2 ]; do sleep 15; done
    ( cd "$d" && bsub $COMMON -m "$m" -n "$n" -o "run_m${m}_n${n}.log" ./a.out ) >/dev/null 2>&1
    echo "[$(date +%H:%M:%S)] submitted $tag m$m n$n" >> "$PROG"
  done
done
while [ "$(qcount)" -gt 0 ]; do sleep 20; done
echo "CORRECTNESS DONE $(date) -- compare the [CHK] line across run_m*_n*.log per benchmark" >> "$PROG"
