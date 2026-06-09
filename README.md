# Reproduction package — distributed MPI+SYCL code generation evaluation

This directory reproduces every quantitative result in the paper's evaluation:
the performance vs. expert hand-written code, strong scaling, the two optimization
ablations, and the code-size table. All experiments run on a production many-core
accelerator supercomputer; the host translator runs anywhere.

> The translator itself (`build/bin/translator/translator`) turns the declarative
> `*.dac.cpp` sources in `../example/` into MPI+SYCL. This package covers the
> *evaluation* of that generated code; the translator itself is a separate component.

## 1. Environment

**Host (translation).** Any platform with the built translator. Translate a benchmark:

```bash
./dacpp.sh translate example/<name>/<name>.dac.cpp --mode=buffer --mpi
# add --mpi-no-resident-halo / --mpi-no-operator-resident for the ablation OFF variants
```

**Cluster (compile + run).** A many-core accelerator supercomputer; x86 login node, many-core
compute nodes. Each node = 4 core groups; one MPI rank binds to each core group (1 management
core + 64 accelerator cores, `cgsp=64`). Max 32 nodes, `-n ≤ 4·m`, so ≤128 ranks.

Set `REPRO_ROOT` to your cluster scratch directory (the one holding `setenv.sh`); every script
below reads it.

```bash
export REPRO_ROOT=/path/to/your/cluster/scratch
source $REPRO_ROOT/setenv.sh        # puts the vendor SYCL compiler (swsycl) on PATH; do NOT pipe it

# compile + link (vendor SYCL = swsycl, Intel-LLVM based; MPI = 2023-06-30 build)
swsycl -std=c++17 -O2 -fsycl -I<runtime> -I/usr/sw/mpi/mpi_20230630_SEA/include -c src.cpp -o src.o
swsycl -fsycl *.o -o a.out -L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib \
       -lmpi -lpthread -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread

# submit (LSF). The generated baseline is built + launched IDENTICALLY.
bsub -q q_share -b -m <nodes> -n <ranks> -cgsp 64 \
     -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 -o run.log ./a.out
```

Caveats: keep ≤2 jobs in the shared `q_share` queue; `bsub` may not propagate
`LD_LIBRARY_PATH` to compute nodes (`libsycl.so.7` not found) — `source setenv.sh`
in the submitting shell. Program output is one of
`e2e_max=` / `total_max=` / `e2e_seconds=`; **`e2e_max` is the headline metric**.

## 2. Experiments → paper artifacts

| paper artifact | what | driver | collector | data | plotter |
|----------------|------|--------|-----------|------|---------|
| **Fig. (vs baseline)** | generated vs. expert MPI+SYCL, 16/32/64/128 ranks | `throttle.sh` | `collect_all.sh` | `sweep_clean.csv` | `plot_sweep.py` |
| **Fig. (strong scaling)** | speedup vs. 16 ranks | `throttle.sh` | `collect_all.sh` | `sweep_clean.csv` | `plot_sweep.py` |
| **Ablation: resident halo** | resident-halo vs. full-sync, 1024²/500 stencil (63–76×) | `resident_halo_ablation/` | — | `resident_halo_ablation/resident_halo.csv` | `resident_halo_ablation/plot_resident_halo.py` |
| **Ablation: operator residency** | decay ON/OFF (1.25× geomean) | `operator_residency_ablation/` | — | `operator_residency_ablation/operator_residency.csv` | `operator_residency_ablation/plot_operator_residency.py` |
| **Table (code size)** | DSL vs. generated vs. hand LOC | — | — | `loc_productivity.csv` | — |
| **Table (benchmark chars.)** | inferred layout/plan | from translator `--mpi` logs | — | — | — |

### 2a. Performance vs. baseline + strong scaling
`throttle.sh` reuses the compiled benchmark + algorithm-aligned baseline binaries in
`bin/<variant>_<bench>/` and submits the rank sweep (≤2 jobs); `collect_all.sh` parses
every result format (`e2e_max` / `total_max` / `e2e_seconds`, plus the comm/comp split)
into `sweep_clean.csv`; `plot_sweep.py` emits `figs/fig5_current_vs_baseline` and
`figs/fig1_strong_scaling` (and an unused `fig6_comm_breakdown`) and prints the speedup
summary.

All runs use the **full** problem sizes of the paper's benchmark table — wave
8192²/10000, stencil 8192²/5000, monteCarlo 4096²/8192 paths, matMul 8192³, DFT 2²⁰,
mandel 8192²/5000, etc. (verified against the translated sources). The baseline is a
legitimate hand-written implementation, not a strawman: e.g. wave uses non-blocking
`MPI_Isend/Irecv` neighbour halo exchange (one `Scatterv` in, one `Gatherv` out), not a
per-step full re-sync.

**Shared-queue noise (important).** Absolute runtimes drift with node allocation on the
shared queue — the *same* binary can vary by tens of percent between allocations (e.g. a
wave baseline measured ~37 s on one allocation and ~61 s on another). We therefore submit
each DACPP/baseline pair back-to-back (same window) and read the ratio, and we treat
per-benchmark margins within ~10% (the communication-bound kernels: wave, stencil, decay,
gradientSum, mdp) as **parity**, not robust wins. The dependable signals are the large /
consistent gains (matMul, monteCarlo, DFT) and the controlled ablations (§2b–2c).

### 2b. Resident-halo ablation (optimization b)
See **`resident_halo_ablation/README.md`** — the *same* 2-D stencil translated with and
without `--mpi-no-resident-halo` (kernels hand-named because swsycl rejects unnamed
lambdas), compiled and run at each rank count on a fixed 1024²/500 grid. ON is 1.2–1.4 s,
OFF ≈89 s; `plot_resident_halo.py` draws the 63–76× bar chart.

### 2c. Operator-residency ablation (general residency)
See **`operator_residency_ablation/README.md`** — includes the runtime-header overlay
patches that make the legacy (OFF) path build under the vendor SYCL, the submit
script, the data, and the plot.

### 2d. Code size
`loc_productivity.csv` lists, per benchmark, declarative-source / generated-MPI+SYCL /
hand-written-baseline line counts (declarative source is on average 2.2× shorter).

## 3. Layout

```
operator_residency_ablation/   self-contained operator-residency repro (sources + overlay patches + data + plot)
resident_halo_ablation/        self-contained resident-halo repro (sources + driver + data + plot)
throttle.sh                    vs-baseline + scaling sweep driver (≤2 jobs)
collect_all.sh                 result collector -> sweep_clean.csv
plot_sweep.py                  strong-scaling + vs-baseline plots
sweep_clean.csv                vs-baseline + scaling data
loc_productivity.csv           code-size data
baselines/                     algorithm-aligned hand-written MPI+SYCL baselines
figs/                          generated figures (copied into ../paper/figs/)
```

`ablation_src/` and `ablation_gen/` are a broader pool of translated ablation/opt
sources (a superset of what the two `*_ablation/` dirs carry, plus exploration variants
such as `stencil.full`, `wave.opt`); `make_ablation_variants.sh` hand-names their kernels;
`sweep_normalized.csv` is derived from `sweep_clean.csv` by `plot_sweep.py`;
`gap_analysis_vs_parent.md` and `ref/` are notes. These are kept for provenance and are
**not** on the canonical path above.

> Earlier-iteration data and plotters that predated the final full-size sweep
> (`iter*.csv`, `plot_current.py`, `plot_fair.py`, `plot.py`, and the `fig2*/fig3*`
> figures) have been **removed**: several wrote the same figure filenames as the
> canonical `plot_sweep.py` from stale numbers, which could overwrite the correct
> figures and contradict the paper. `plot_sweep.py` is the sole vs-baseline/scaling
> plotter; `sweep_clean.csv` is the single source of truth for those results.
