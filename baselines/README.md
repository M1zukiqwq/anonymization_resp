# Tier-1 algorithm-aligned hand-written MPI+SYCL baselines

10 hand-written baselines, one per benchmark, each using the **exact same data
decomposition + communication pattern as the DACPP-generated code** (extracted
from `clang/tools/translator/example/<name>/<name>.dac_sycl_buffer.cpp`), but
written cleanly by hand. Purpose: a **fair** baseline that isolates
code-generation overhead — NOT a better algorithm, NOT a strawman. Expected
result ≈ DACPP within ~10–20% (codegen overhead), per the paper narrative
"automation without performance penalty".

All files are **self-contained**: only `<sycl/sycl.hpp>` (with `<CL/sycl.hpp>`
fallback, alias `dacpp_sycl`), `<mpi.h>`, `mpi_bench_timer.h`, and std headers.
No DACPP runtime headers. Timing via `BenchTimer` (mpi_bench_timer.h) prints the
same `[MPI TEST] ... total_max= ... communication_max= ... computation_max=`
format as the DACPP `_test` harness, so results drop straight into the analysis.

## Per-benchmark alignment + size flags (match DACPP `src/` sizes)
| benchmark | layout aligned to | comm pattern | size -D to match DACPP src |
|---|---|---|---|
| matMul | RowPartitionFullRow | Scatterv A rows + Bcast B + Gatherv C | `-DMATMUL_N=8192` |
| gradientSum | RowPartitionFullRow | Scatterv rows + per-row reduce + Gatherv | `-DGRADIENT_NUM_NEURONS=8192 -DGRADIENT_INPUT_SIZE=8192` |
| DFT | ReplicatedFullTensor | Bcast input + partition bins + Gatherv | `-DDFT_N=<src N>` |
| decay | Contiguous1D | Scatter once (resident) + Bcast scalar/step + Gather | `-DDECAY_NUM_ISOTOPES=1024576` |
| mandel | Contiguous1D | Scatter points + Gather flags | (set to src size) |
| monteCarloOption | RowBlock2D | Scatter row-block seeds + Gather prices | (set to src size) |
| liuliang | StencilWindow1D + halo(1) | Sendrecv 1-cell halo / step | `-DLIULIANG_WIDTH=1000000 -DLIULIANG_STEPS=5000` |
| mdp | StencilWindow1D + halo(1) | Sendrecv 1-cell halo / step | `-DMDP_N=1000000 -DMDP_T=20000` |
| stencil | StencilWindow2D + halo(1,1) | Sendrecv 1 ghost row / step | `-DSTENCIL_NX=.. -DSTENCIL_NY=..` |
| waveEquation | StencilWindow2D + halo(1,1), 3-buf | Sendrecv 1 ghost row / step | (set to src size) |

> ⚠️ Default sizes in some files are the small *example* sizes; for the paper
> comparison pass `-D` to match the exact DACPP `src/` problem size so baseline
> and DACPP run identical work. Confirm each `src/<name>...` size on the cluster.

## Build on the cluster (per baseline)
```bash
source $REPRO_ROOT/setenv.sh
swsycl -std=c++17 -O2 -fsycl -I. -I/usr/sw/mpi/mpi_20230630_SEA/include <-D...> \
  -c <name>.MPI_StandardSycl_aligned.cpp -o o.o
swsycl -fsycl o.o -o a.out -L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic \
  -L/usr/sw/lib -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread \
  -Wl,-rpath,$REPRO_ROOT/swsycl/lib/swcl/athread/lib
bsub -I -q q_share -b -m <m> -n <n> -cgsp 64 ... ./a.out
```

## ⚠️ Validation checklist (NOT yet done — no local SYCL compiler)
These are careful drafts written from the generated algorithm; they have **not**
been compiled or run. Before using in the paper:
1. **Compile** each with swsycl on the cluster; fix any API/syntax issues.
2. **Correctness**: confirm each produces the same answer as the DACPP version
   (compare the printed sanity value / checksum) at small size.
3. **Size match**: set `-D` so baseline and DACPP use identical problem size.
4. Then run the scaling sweep (ranks 4/32/128) and recompute the vs-baseline table.
