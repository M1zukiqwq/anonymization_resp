#!/usr/bin/env python3
"""Plot the operator-residency ablation (decay, ON vs OFF) as an OFF/ON speedup bar.

Reads operator_residency.csv and writes operator_residency.{pdf,png}. The paper
reports these as paired same-window ratios rather than absolute seconds, because
the shared queue makes absolute runtimes drift with node allocation (e.g. the
128-rank ON time reflects an unlucky allocation). Prints the geometric mean.
"""
import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CB = ["#0072B2", "#D55E00", "#009E73"]

ranks, ratio = [], []
with open(os.path.join(HERE, "operator_residency.csv")) as f:
    for row in csv.DictReader(f):
        ranks.append(int(row["ranks"]))
        ratio.append(float(row["off_over_on"]))

geomean = math.exp(sum(math.log(r) for r in ratio) / len(ratio))

def main():
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = list(range(len(ranks)))
    ax.bar(x, ratio, width=0.58, color=CB[0], edgecolor="black", lw=0.6, zorder=3)
    for xi, r in zip(x, ratio):
        ax.annotate(f"{r:.2f}×", (xi, r), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color=CB[1], lw=1.6, ls="--", zorder=2)
    ax.text(len(ranks) - 0.5, 1.02, "operator residency OFF $= 1\\times$",
            color=CB[1], fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ranks)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("speedup from operator\nresidency (OFF/ON, $\\times$)")
    ax.set_ylim(0, max(ratio) * 1.18)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"operator_residency.{e}"), dpi=150)
    print("ratios:", [round(r, 2) for r in ratio], "| geomean:", round(geomean, 3))

if __name__ == "__main__":
    main()
