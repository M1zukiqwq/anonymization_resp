# Operator-residency ablation (decay, ON vs OFF)

Reproduces the general operator-residency result in the paper's evaluation: keeping
an operator's data **resident** across loop iterations vs. the legacy per-iteration
gather/scatter path. Toggled by the single translator flag `--mpi-no-operator-resident`.

**Result (paired OFF/ON `e2e_max` ratios, decay):**

| ranks (m/n) | ON e2e_max | OFF e2e_max | OFF/ON |
|-------------|-----------:|------------:|-------:|
| 16 (4/16)   | 74.47 s    | 113.36 s    | 1.52×  |
| 32 (8/32)   | 74.30 s    | 89.74 s     | 1.21×  |
| 64 (16/64)  | 73.87 s    | 88.74 s     | 1.20×  |
| 128 (32/128)| 102.08 s   | 111.43 s    | 1.09×  |

Geometric mean **1.25×**. The gain shrinks with rank count (a 1-D tensor's
per-iteration round trip is a smaller share of runtime as each rank's slice
shrinks), and is far below the stencil-specific resident-halo result (63–76×,
see `../ablation_results.csv` / `../plot_fig4.py`). Absolute seconds drift on
the shared queue (the 128-rank ON time reflects an unlucky allocation), so we
report the paired same-window ratio. Raw numbers: `operator_residency.csv`.

## Files

| file | purpose |
|------|---------|
| `decay.OR_on.cpp`             | generated ON variant (operator residency enabled) |
| `decay.OR_off.cpp`            | generated OFF variant (`--mpi-no-operator-resident`; legacy gather/scatter) |
| `submit_decay_sweep.sh` | submit ON/OFF over the rank sweep (LSF `bsub`, ≤2 jobs queued) |
| `operator_residency.csv`      | the eight `e2e_max` data points + OFF/ON ratio |
| `plot_operator_residency.py`  | bar chart of OFF/ON speedup (`operator_residency.{pdf,png}`) |
| `overlay/*.patch`             | runtime-header fix that lets the OFF variant compile (see below) |

## Why the OFF variant needs a runtime overlay

The OFF variant (`--mpi-no-operator-resident`) lowers to the **legacy** gather/scatter
runtime, which includes `DataReconstructor1.h` and `StencilExchangeRuntime.h`. The
vendor SYCL compiler (`swsycl`, Intel-LLVM based) rejects it for two reasons:

1. Several `parallel_for(...)` launches carry **no kernel name** — `swsycl` does not
   support `-fsycl-unnamed-lambda` ("No kernel name provided ...").
2. Two different kernels share the name `class MyKernel4`. The duplicate makes the
   generated SYCL **integration header** emit conflicting `KernelInfoData`
   specializations ("'KernelInfoData' is not a class template"). This is the root
   cause that survives if you only name the unnamed kernels — you must also make the
   duplicate names unique.

The two patches in `overlay/` forward-declare unique, template-dependent kernel-name
tags and attach one per launch (a kernel name is metadata only; behavior is
unchanged). Apply them as an **overlay** — never edit the shared runtime in place.

## Reproduce on the cluster

Paths: `Z=$REPRO_ROOT`, `P=$Z/paper/ablperf`, `RT=$Z/multi-node-test/runtime`.

```bash
# 0. the exact generated sources are bundled here (decay.OR_on.cpp, decay.OR_off.cpp);
#    upload them to $P. To regenerate from the DSL on the host translator instead:
#      ./dacpp.sh translate decay.dac.cpp --mode=buffer --mpi                         # ON
#      ./dacpp.sh translate decay.dac.cpp --mode=buffer --mpi --mpi-no-operator-resident  # OFF

# 1. build the patched runtime overlay (copy runtime, apply the two patches)
cp -r "$RT" "$P/rt_named"
( cd "$P/rt_named"           && patch -p2 < /path/to/overlay/DataReconstructor1.h.patch )
( cd "$P/rt_named"           && patch -p2 < /path/to/overlay/StencilExchangeRuntime.h.patch )
#   -p2 strips the leading "a/runtime/"; adjust to match your layout.

# 2. compile both variants (overlay -I FIRST so the patched headers win)
source "$Z/setenv.sh"
cd "$P"
for v in on off; do
  swsycl -std=c++17 -O2 -fsycl -I"$P/rt_named" -I"$RT" \
    -I/usr/sw/mpi/mpi_20230630_SEA/include \
    -c decay.OR_${v}.cpp -o decay.OR_${v}.o
  swsycl -fsycl decay.OR_${v}.o -o decay.OR_${v}.out \
    -L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib \
    -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread
done

# 3. submit the sweep (<=2 jobs at a time on the shared queue)
MAX_JOBS=2 ./submit_decay_sweep.sh
#   or one config:  CONFIGS="16 64" MAX_JOBS=2 ./submit_decay_sweep.sh

# 4. extract results (use the last run per log; m16_n64 was rerun)
for n in 16 32 64 128; do
  case $n in 16) m=4;; 32) m=8;; 64) m=16;; 128) m=32;; esac
  for mode in on off; do
    f=decay.OR_${mode}.run_m${m}_n${n}.log
    echo "$mode m=$m n=$n"; grep "\[MPI TEST\]" "$f" | tail -1
  done
done

# 5. plot
python3 plot_operator_residency.py
```
