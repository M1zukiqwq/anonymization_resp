#!/usr/bin/env python3
"""Parse paper_sweep.txt (ranks 16/32/64/128) -> clean CSV + paper figures.

Input line format (one per run), produced by paper_sweep_par.sh / paper_sweep.sh:
    RESULT|<bench>:<variant>|ranks=<N>|<raw program output line(s)>
Timing is extracted from the raw text in priority order:
    total_max=<s>  |  e2e_max=<s>  |  e2e_seconds=<s>
Optional comm/comp: communication_max=<s>, computation_max=<s>.
"""
import os, re, csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
CSVIN = os.path.join(HERE, "sweep_clean.csv")          # from collect_all.sh on cluster
RANKS = [16, 32, 64, 128]
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

# canonical benchmark order + scalable (compute-bound) classification
BENCH = ["matMul","decay","gradientSum","DFT","mandel",
         "liuliang","mdp","monteCarlo","stencil","wave"]
SCALABLE = ["DFT","monteCarlo","wave","stencil","mandel"]

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load():
    """read sweep_clean.csv -> {(bench,variant): {ranks: (total,comm,comp)}}."""
    data = {}
    if not os.path.exists(CSVIN):
        sys.exit(f"missing {CSVIN} -- run collect_all.sh on the cluster and fetch it first")
    with open(CSVIN) as f:
        for row in csv.DictReader(f):
            variant = "gen" if row["variant"] == "dacpp" else row["variant"]
            bench = row["benchmark"]
            n = int(row["ranks"]) if row["ranks"].isdigit() else None
            tot = _f(row["total_s"])
            if n is None or tot is None:
                print(f"  [warn] no timing for {variant} {bench} m{row['m']} n{row['n']}")
                continue
            data.setdefault((bench, variant), {})[n] = (tot, _f(row["comm_s"]), _f(row["comp_s"]))
    return data

def write_csv(data):
    out = os.path.join(HERE, "sweep_normalized.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant","benchmark","ranks","total_s","comm_s","comp_s"])
        for (bench, variant), d in sorted(data.items()):
            for n in sorted(d):
                tot, comm, comp = d[n]
                w.writerow([variant, bench, n, tot,
                            "" if comm is None else comm,
                            "" if comp is None else comp])
    print("wrote", out)

def gen(data, b): return data.get((b, "gen"), {})
def base(data, b): return data.get((b, "base"), {})

def fig_vs_baseline(data):
    # heatmap: benchmarks x rank counts, each cell the speedup (annotated). A
    # shared speedup axis let matMul's ~12x dominate and squashed every other
    # benchmark; per-cell colour + annotation decouples them. Colour is a
    # diverging map centred at parity (1.0); matMul's large values clamp to vmax
    # but their exact numbers are printed.
    from matplotlib.colors import TwoSlopeNorm
    rows = []
    for b in BENCH:
        g, bs = gen(data,b), base(data,b)
        if not (g and bs):
            continue
        sp = {rk: bs[rk][0]/g[rk][0] for rk in RANKS
              if rk in g and rk in bs and g[rk][0] > 0}
        if len(sp) == len(RANKS):
            rows.append((b, sp))
    rows.sort(key=lambda r: np.exp(np.mean(np.log(list(r[1].values())))), reverse=True)  # matMul top
    labels = [b for b, _ in rows]
    M = np.array([[sp[rk] for rk in RANKS] for _, sp in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    norm = TwoSlopeNorm(vmin=0.9, vcenter=1.0, vmax=2.5)
    im = ax.imshow(M, aspect="auto", cmap="RdBu", norm=norm)  # red=slower, blue=faster
    ax.set_xticks(range(len(RANKS))); ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("MPI ranks")
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            c = norm(v)
            tc = "white" if (c > 0.80 or c < 0.20) else "black"
            ax.text(j, i, f"{v:.1f}" if v >= 10 else f"{v:.2f}",
                    ha="center", va="center", fontsize=8, color=tc)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cb.set_label("speedup over hand-written baseline\n($<$1 slower, $=$1 parity, $>$1 faster)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for e in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"fig5_current_vs_baseline.{e}"), dpi=150)
    plt.close(fig); print("wrote fig5_current_vs_baseline")

def fig_scaling(data):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    cols = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00"]
    for i, b in enumerate([b for b in SCALABLE if all(r in gen(data,b) for r in RANKS)]):
        g = gen(data,b)
        sp = [g[RANKS[0]][0]/g[r][0] for r in RANKS]
        ax.plot(RANKS, sp, marker="o", lw=2, color=cols[i % len(cols)], label=b)
    ideal = [r/RANKS[0] for r in RANKS]
    ax.plot(RANKS, ideal, "k--", lw=1.2, label="ideal")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    yt=[1,2,4,8]; ax.set_yticks(yt); ax.set_yticklabels(yt)
    ax.set_xlabel("MPI ranks"); ax.set_ylabel(f"speedup vs {RANKS[0]} ranks (T$_{{{RANKS[0]}}}$/T$_n$)")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    for e in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"fig1_strong_scaling.{e}"), dpi=150)
    plt.close(fig); print("wrote fig1_strong_scaling")

def fig_comm_breakdown(data):
    # gen-only: communication vs computation share of total_max at 128 ranks
    present = [b for b in BENCH if 128 in gen(data,b) and gen(data,b)[128][1] is not None]
    if not present:
        print("  [comm] no communication_max data; skipping breakdown fig"); return
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(present))
    comm = [gen(data,b)[128][1] for b in present]
    tot  = [gen(data,b)[128][0] for b in present]
    commfrac = [c/t if t else 0 for c,t in zip(comm,tot)]
    compfrac = [1-cf for cf in commfrac]
    ax.bar(x, compfrac, color=CB[2], label="computation")
    ax.bar(x, commfrac, bottom=compfrac, color=CB[1], label="communication")
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("fraction of e2e time (128 ranks)")
    pass  # no title (anonymized; caption describes)
    ax.legend(fontsize=8, ncol=2); ax.set_ylim(0,1)
    fig.tight_layout()
    for e in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"fig6_comm_breakdown.{e}"), dpi=150)
    plt.close(fig); print("wrote fig6_comm_breakdown")

def summary(data):
    print("\n=== SUMMARY (paste into eval.tex) ===")
    print(f"{'bench':12} " + " ".join(f"sp@{r}".rjust(8) for r in RANKS) + "   scaling(T16/T128)")
    for b in BENCH:
        g, bs = gen(data,b), base(data,b)
        row = f"{b:12} "
        for r in RANKS:
            if r in g and r in bs and g[r][0]>0:
                row += f"{bs[r][0]/g[r][0]:8.2f} "
            else:
                row += f"{'--':>8} "
        if all(r in g for r in (RANKS[0], RANKS[-1])):
            row += f"   {g[RANKS[0]][0]/g[RANKS[-1]][0]:6.1f}x"
        print(row)

if __name__ == "__main__":
    d = load()
    write_csv(d)
    fig_vs_baseline(d)
    fig_scaling(d)
    fig_comm_breakdown(d)
    summary(d)
