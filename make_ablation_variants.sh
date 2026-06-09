#!/bin/bash
# Mac-side: translate a harnessed benchmark DSL source into ablation variants.
# (Translator runs on Mac; resulting *.cpp are scp'd to the cluster for compile+run.)
#
# Usage: make_ablation_variants.sh <harnessed.benchmark.mpi.dac.cpp> <outdir> [axis]
#   axis = or | halo | both (default both)
#   - or   : ON  vs --mpi-no-operator-resident   (operator-resident axis, #1)
#   - halo : ON  vs --mpi-no-resident-halo        (resident-halo axis, #2, stencil/wave)
#
# Produces <outdir>/<base>.on.cpp and <base>.no_<axis>.cpp ready for the cluster.
set -u
TR=../dacpp.sh                       # relative to clang/tools/translator
SRC=$1; OUT=$2; AXIS=${3:-both}
base=$(basename "$SRC" .dac.cpp)
mkdir -p "$OUT"

emit() { # <flagvar-suffix> <translator-flags>
  local tag=$1; shift
  "$TR" translate "$SRC" --mode=buffer --mpi "$@" >/dev/null 2>&1
  cp "${SRC%.dac.cpp}.dac_sycl_buffer.cpp" "$OUT/${base}.${tag}.cpp"
  echo "  wrote $OUT/${base}.${tag}.cpp ($(grep -c '__dacpp_mpi_or_' "$OUT/${base}.${tag}.cpp") or-wrappers)"
}

echo "[$base] ON (all optimizations)"; emit on
case "$AXIS" in
  or|both)   echo "[$base] no operator-resident";  emit no_or   --mpi-no-operator-resident ;;
esac
case "$AXIS" in
  halo|both) echo "[$base] no resident-halo";       emit no_halo --mpi-no-resident-halo ;;
esac
echo "done. scp $OUT/*.cpp to zsp/paper/abl_perf/ then compile+bsub on the cluster."
