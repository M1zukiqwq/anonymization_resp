#!/bin/bash
# Throttled DACPP MPI sweep: at most 2 of MY jobs in the shared queue at any time.
# Reuses $REPRO_ROOT/paper/bin/<variant>_<bench>/a.out and run_m<m>_n<n>.log naming.
Z=$REPRO_ROOT
PAPER=$Z/paper
source $Z/setenv.sh
COMMON="-q q_share -b -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128"
NAMES=(decay_chain DFT gradientSum liuliang mandel matMul mdp monteCarloOption stencil waveEquation)
PROG=$PAPER/throttle.progress

qcount() { bjobs -u $USER 2>/dev/null | grep -c q_share; }

submit_one() {
  local v=$1 bench=$2 m=$3 n=$4
  local d=$PAPER/bin/${v}_${bench}
  [ -x "$d/a.out" ] || { echo "[skip] no a.out ${v}_${bench}" >> "$PROG"; return; }
  # wait until I have at most 1 job in the queue, then submit (=> <=2 in flight)
  while [ "$(qcount)" -ge 2 ]; do sleep 15; done
  ( cd "$d" && bsub $COMMON -m "$m" -n "$n" -o "run_m${m}_n${n}.log" ./a.out ) >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] submitted ${v}_${bench} m$m n$n (q=$(qcount))" >> "$PROG"
}

echo "THROTTLE START $(date)" > "$PROG"
for cfg in "4 16" "8 32" "16 64" "32 128"; do
  set -- $cfg; m=$1; n=$2
  for bench in "${NAMES[@]}"; do
    submit_one dacpp "$bench" "$m" "$n"
    submit_one base  "$bench" "$m" "$n"
  done
done
# wait for the last jobs to finish
while [ "$(qcount)" -gt 0 ]; do sleep 20; done
echo "THROTTLE DONE $(date)" >> "$PROG"
