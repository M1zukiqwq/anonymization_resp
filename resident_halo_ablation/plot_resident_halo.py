#!/usr/bin/env python3
"""Plot the resident-halo ablation (2-D stencil, 1024^2/500) as an OFF/ON speedup bar.

Reads resident_halo.csv (long form: variant,ranks,e2e_max_s) and writes
resident_halo.{pdf,png}. Two near-flat absolute-time lines decades apart read as
empty whitespace, so a single ratio bar puts the headline (63-76x) front and
center; absolute times (1.2-1.4 s ON vs ~89 s OFF) go in the caption/README.
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CB = ["#0072B2", "#D55E00", "#009E73"]

on, off = {}, {}
with open(os.path.join(HERE, "resident_halo.csv")) as f:
    for row in csv.DictReader(f):
        (on if row["variant"] == "on" else off)[int(row["ranks"])] = float(row["e2e_max_s"])
ranks = sorted(on)
ratio = [off[r] / on[r] for r in ranks]

def main():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = list(range(len(ranks)))
    ax.bar(x, ratio, width=0.6, color=CB[1], edgecolor="black", lw=0.6, zorder=3)
    for xi, r in zip(x, ratio):
        ax.annotate(f"{r:.0f}×", (xi, r), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color=CB[2], lw=1.8, ls="--", zorder=2)
    ax.text(len(ranks) - 0.55, max(ratio) * 0.05 + 2.5, "resident-halo (ON) $= 1\\times$",
            color=CB[2], fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ranks)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("slowdown from disabling\nresident-halo ($\\times$)")
    ax.set_ylim(0, max(ratio) * 1.18)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"resident_halo.{e}"), dpi=150)
    print("ratios:", [round(r, 1) for r in ratio])

if __name__ == "__main__":
    main()
