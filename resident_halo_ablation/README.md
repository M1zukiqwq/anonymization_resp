# Resident-halo ablation (2-D stencil, ON vs OFF)

Reproduces the paper's headline ablation: keeping the stencil halo **resident** and
updating only its edges, with depth-2 temporal blocking, vs. re-synchronizing the
whole tensor every step. Toggled by the single translator flag `--mpi-no-resident-halo`.
A fixed 1024²/500 grid isolates communication (the ratio is independent of step
count; a benchmark-size full-sync would take hours).

**Result (`e2e_max`, same source, one flag):**

| ranks (m/n) | ON (resident halo) | OFF (full sync) | OFF/ON |
|-------------|-------------------:|----------------:|-------:|
| 16 (4/16)   | 1.233 s            | 89.523 s        | 72.6×  |
| 32 (8/32)   | 1.182 s            | 89.320 s        | 75.5×  |
| 64 (16/64)  | 1.305 s            | 89.598 s        | 68.7×  |
| 128 (32/128)| 1.433 s            | 89.812 s        | 62.7×  |

**63–76×**, nearly constant across rank counts: full synchronization pays a global
round trip *every* step, the resident path only an edge exchange. The OFF binary is
exactly the lowering a correctness-only generator emits, so this is the cost of the
optimization against the only correct alternative, not a strawman. Raw numbers:
`resident_halo.csv`.

## Files

| file | purpose |
|------|---------|
| `ablation_driver2.sh`      | translate + hand-name kernels + compile + run ON/OFF at each rank (cluster) |
| `src/stencil.halo_on.named.cpp`  | generated ON variant, kernels hand-named (swsycl rejects unnamed lambdas) |
| `src/stencil.halo_off.named.cpp` | generated OFF variant (`--mpi-no-resident-halo`), hand-named |
| `resident_halo.csv`        | ON/OFF `e2e_max` per rank |
| `plot_resident_halo.py`    | OFF/ON slowdown bar chart (`resident_halo.{pdf,png}`) |

## Reproduce on the cluster

The `src/*.named.cpp` are the exact sources used (1024²/500). Kernels are hand-named
because the vendor `swsycl` rejects unnamed lambdas and does not support
`-fsycl-unnamed-lambda` (the generator's named-kernel path is part of optimization c;
these ablation variants were named with a regex pass over `h.parallel_for(`).

```bash
source $Z/setenv.sh
RT=$Z/multi-node-test/runtime
CF="-std=c++17 -O2 -fsycl -I$RT -I/usr/sw/mpi/mpi_20230630_SEA/include"
LF="-L/usr/sw/mpi/mpi_20230630_SEA/lib/single_dynamic -L/usr/sw/lib -lmpi -lpthread \
    -lswverbs -lswutils -lswtm -lm -ldl -lrt -lathread"
for v in on off; do
  swsycl $CF -c src/stencil.halo_${v}.named.cpp -o stencil_${v}.o
  swsycl -fsycl stencil_${v}.o -o stencil_${v}.out $LF
  for cfg in "4 16" "8 32" "16 64" "32 128"; do set -- $cfg
    bsub -q q_share -b -m $1 -n $2 -cgsp 64 -share_size 13000 -priv_size 16 \
         -host_stack 1024 -cache_size 128 -o run_${v}_m$1_n$2.log ./stencil_${v}.out
  done
done
# (keep <=2 jobs queued; ablation_driver2.sh wraps this with a throttle.)
python3 plot_resident_halo.py
```

`ablation_driver2.sh` is the cluster-side driver that does the above with a ≤2-job
throttle (paths are the cluster's `$Z/paper/abl`).
