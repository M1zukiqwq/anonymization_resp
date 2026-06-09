#!/bin/bash
# Robust collector for zsp/paper/bin/<v>_<bench>/run_m<m>_n<n>.log
# Handles all three output formats: total_max= , e2e_max= , e2e_seconds=
# Emits CSV: variant,benchmark,m,n,ranks,total_s,comm_s,comp_s
PAPER=$REPRO_ROOT/paper
echo "variant,benchmark,m,n,ranks,total_s,comm_s,comp_s"
for d in "$PAPER"/bin/*/; do
  tag=$(basename "$d"); v=${tag%%_*}; bench=${tag#*_}
  case "$bench" in
    decay_chain) bench=decay;; monteCarloOption) bench=monteCarlo;; waveEquation) bench=wave;;
  esac
  for log in "$d"run_m*_n*.log; do
    [ -f "$log" ] || continue
    mn=$(basename "$log" .log)
    m=$(echo "$mn" | sed -E 's/run_m([0-9]+)_n.*/\1/')
    n=$(echo "$mn" | sed -E 's/.*_n([0-9]+)/\1/')
    line=$(grep -aE "total_max=|e2e_max=|e2e_seconds=" "$log" 2>/dev/null | tail -1)
    tot=NA; comm=""; comp=""; ranks="$n"
    if [ -n "$line" ]; then
      tot=$(echo "$line"  | grep -oE "total_max=[0-9.]+"    | head -1 | cut -d= -f2)
      [ -z "$tot" ] && tot=$(echo "$line" | grep -oE "e2e_max=[0-9.]+"     | head -1 | cut -d= -f2)
      [ -z "$tot" ] && tot=$(echo "$line" | grep -oE "e2e_seconds=[0-9.]+" | head -1 | cut -d= -f2)
      [ -z "$tot" ] && tot=NA
      comm=$(echo "$line" | grep -oE "communication_max=[0-9.]+" | head -1 | cut -d= -f2)
      comp=$(echo "$line" | grep -oE "computation_max=[0-9.]+"   | head -1 | cut -d= -f2)
      r=$(echo "$line"    | grep -oE "ranks=[0-9]+"             | head -1 | cut -d= -f2)
      [ -n "$r" ] && ranks="$r"
    fi
    echo "$v,$bench,$m,$n,$ranks,$tot,$comm,$comp"
  done
done
