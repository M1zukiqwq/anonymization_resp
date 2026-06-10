# Correctness: full-grid bit-exactness by rank-invariance

Verifies that the **inferred** halo communication preserves single-node semantics exactly —
not by sampling, but over every element of the final grid.

## Method

`injector.py` splices a root-side fingerprint right after the `materialize` call of a
generated MPI+SYCL program: an FNV-1a hash over the **entire** materialized grid in canonical
global (row-major) order, plus `sum`/`min`/`max`. We then run the *same* binary at several
rank counts:

- **1 rank** performs **no** inter-rank halo exchange — it is the sequential reference;
- **16 / 64 / 128 ranks** exercise the inferred rank-to-rank halo exchange.

Because a stencil has no cross-rank reduction, a value-preserving distribution must reproduce
the global grid **bit-for-bit** regardless of rank count. So a fingerprint that is identical
across all rank counts proves every element matches — any halo cell the inference got wrong
would change the hash. A small `512²/100` grid suffices (correctness is size-independent).

## Result (`correctness_fingerprints.txt`)

Both 2-D halo idioms are bit-identical across 1/16/64/128 ranks (262 144 elements each):

| benchmark | halo idiom | fingerprint (all rank counts) |
|-----------|-----------|-------------------------------|
| stencil   | direct neighbour-copy      | `fnv=55cfb88c3bf0cfdc` |
| wave      | read-cache buffer swap     | `fnv=6900a9c0d3871ced` |

## Reproduce (cluster)

```bash
export REPRO_ROOT=/path/to/your/cluster/scratch
bash run_correctness.sh          # injects, compiles, runs at 1/16/64/128 ranks (<=2 jobs queued)
# then compare the single [CHK] line across the per-rank logs; they must be identical:
grep -h '\[CHK\]' $REPRO_ROOT/paper/correctness/stencil/run_m*_n*.log | sort -u   # -> 1 line
grep -h '\[CHK\]' $REPRO_ROOT/paper/correctness/wave/run_m*_n*.log    | sort -u   # -> 1 line
```

## Files

| file | purpose |
|------|---------|
| `injector.py`                  | splices the whole-grid FNV-1a fingerprint after `materialize` (byte-safe latin-1 I/O) |
| `run_correctness.sh`           | inject + compile + run stencil & wave at 1/16/64/128 ranks |
| `correctness_fingerprints.txt` | the per-rank `[CHK]` lines (identical within each benchmark) |

Sources: stencil = the resident-halo ON variant in `../resident_halo_ablation/src/`; wave = the
standard `--mpi` translation of `example/waveEquation`.
