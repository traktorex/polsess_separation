"""Rysunek 5.1 — baseline quality vs parameter count (ch5, baselines section).

Simplified per author feedback 2026-08-02: no training-cost encoding, uniform
circular markers, one color per architecture, identity carried by direct labels
(no legend — every point is named in place, so a legend would only duplicate).

Scatter of validation SI-SDR ({SER,SE}, pilot corpus, 3 seeds where available)
against parameter count (log x). SI-SDR values embedded with provenance
comments: the Series-1 baseline runs were deleted from W&B (early Feb 2026), so
numbers are sourced from the verified evidence pack in the author's thesis
notes (re-pulled 2026-07-29) and thesis-log/01_baselines.md. Parameter counts
follow
docs/generated/model_manifest.md (NB Mamba-TasNet-L = 58.95M there; the older
59.6M in logs is stale).

Usage:
    python scripts/thesis_figures/fig_ch5_baseline_scatter.py [--out-dir DIR]
Writes rys_ch5_baseline_scatter.{png,pdf} (PNG 300 dpi, PDF vector).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch05"

# one color per architecture (fixed assignment — keep identical across ch5 figures)
COLOR = {
    "SPMamba": "#2a78d6",       # blue
    "SepFormer": "#eb6834",     # orange
    "MambaTasNet": "#1baf7a",   # aqua
    "DPRNN": "#008300",         # green
    "DPMamba": "#e87ba4",       # magenta
    "ConvTasNet": "#eda100",    # yellow
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# (arch, params, si_sdr, sigma) — sigma None = single seed
POINTS = [
    ("ConvTasNet", 8_708_672, 2.95, 0.29),       # 3.28/2.70/2.86; log-sourced
    ("DPRNN", 2_609_793, 3.03, 0.17),            # k=16; log-sourced
    ("DPRNN", 2_608_001, 4.57, 0.01),            # k=2; d9pw9f7x/mwxwnyvr (2 seeds)
    ("SepFormer", 25_679_361, 5.16, 0.14),       # posenc; kvyxo3t9/iqc12vqd/nvaol6l1
    ("SPMamba", 1_156_368, 5.56, 0.12),          # reduced 4L/192; log-sourced
    ("MambaTasNet", 2_210_176, 3.39, 0.12),      # XS, 3 seeds
    ("MambaTasNet", 7_926_528, 4.33, None),      # S
    ("MambaTasNet", 15_647_488, 4.61, None),     # M
    ("MambaTasNet", 58_951_168, 5.15, None),     # L (model_manifest count)
    ("DPMamba", 2_265_857, 3.52, None),          # XS
    ("DPMamba", 8_136_193, 3.84, None),          # S
    ("DPMamba", 15_869_441, 3.45, None),         # M
    ("DPMamba", 59_788_289, 4.40, None),         # L
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # size-ladder connectors, then the DPRNN window pair (same x, quality jump)
    for arch in ("MambaTasNet", "DPMamba"):
        pts = sorted((p, s) for a, p, s, _ in POINTS if a == arch)
        ax.plot(*zip(*pts), color=COLOR[arch], lw=1.2, alpha=0.45, zorder=1)
    ax.plot([2_608_001, 2_609_793], [4.57, 3.03], color=COLOR["DPRNN"], lw=1.0,
            ls=":", alpha=0.7, zorder=1)

    for arch, params, sdr, sigma in POINTS:
        ax.errorbar(params, sdr, yerr=sigma, fmt="none", ecolor=COLOR[arch],
                    elinewidth=1.0, capsize=2.5, zorder=2)
        ax.scatter(params, sdr, s=85, marker="o", facecolor=COLOR[arch],
                   edgecolor="white", linewidth=0.8, zorder=3)

    lab = dict(fontsize=8.4, color=INK)
    small = dict(fontsize=7.4, color=INK2)
    ax.annotate("SPMamba", (1_156_368, 5.56), (0, 11), "data", textcoords="offset points", ha="center", **lab)
    ax.annotate("SepFormer", (25_679_361, 5.16), (0, 10), "data", textcoords="offset points", ha="center", **lab)
    ax.annotate("Mamba-TasNet", (58_951_168, 5.15), (0, 10), "data", textcoords="offset points", ha="center", **lab)
    ax.annotate("DPMamba", (59_788_289, 4.40), (8, -3), "data", textcoords="offset points", ha="left", **lab)
    ax.annotate("DPRNN k=2", (2_608_001, 4.57), (-7, 7), "data", textcoords="offset points", ha="right", **lab)
    ax.annotate("k=16", (2_609_793, 3.03), (-8, -4), "data", textcoords="offset points", ha="right", **lab)
    ax.annotate("ConvTasNet", (8_708_672, 2.95), (14, -14), "data", textcoords="offset points", ha="left", **lab)
    # ladder size letters (Mamba-TasNet row; DPMamba mirrors the same sizes)
    ax.annotate("XS", (2_210_176, 3.39), (-8, -3), "data", textcoords="offset points", ha="right", **small)
    ax.annotate("S", (7_926_528, 4.33), (0, 8), "data", textcoords="offset points", ha="center", **small)
    ax.annotate("M", (15_647_488, 4.61), (0, 8), "data", textcoords="offset points", ha="center", **small)

    ax.set_xscale("log")
    ax.set_xlim(0.75e6, 1.4e8)
    ax.set_ylim(2.3, 6.1)
    ax.set_yticks([3, 4, 5, 6])       # integer ticks: no decimal separator to localise
    ax.set_yticks(np.arange(2.5, 6.1, 0.5), minor=True)
    ax.set_xticks([1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8])
    ax.set_xticklabels(["1 mln", "2 mln", "5 mln", "10 mln", "20 mln", "50 mln", "100 mln"])
    ax.set_xlabel("Liczba parametrów")
    ax.set_ylabel("SI-SDR [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_baseline_scatter.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_baseline_scatter.{{png,pdf}}")


if __name__ == "__main__":
    main()
