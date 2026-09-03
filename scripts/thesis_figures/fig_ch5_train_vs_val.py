"""Rysunek (ch5 §5.6, kandydat) — trajektorie (trening, walidacja) 16->32->64 tys.

Unifies fig_ch5_capacity_vs_data.py (val) and fig_ch5_traingap.py (gap) into
one chart: x = train SI-SDR at the best-val epoch (the "ceiling"), y = val
SI-SDR. Each model is a 3-point trajectory over 16/32/64 tys.; marker size
grows with data size, an arrowhead sits on the last segment. Faint diagonals
are iso-gap lines (val = train - g): vertical distance below the val=train
line IS the train-val gap. Reduced models move straight up (gap closure
against a fixed ceiling); full models drift up-and-right (ceiling still
rising). Same sources/harvest as the two sibling figures (2026-08-20);
MF2-matched@16k train/val from the surviving log of the clobbered dir.

Usage: python scripts/thesis_figures/fig_ch5_train_vs_val.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SF, SPM, MF2 = "#eb6834", "#2a78d6", "#7b52ab"
DASH = (0, (4, 2))

# (label, color, linestyle, [(train, val) @16k, @32k, @64k])
SERIES = [
    ("SepFormer", SF, "-",
     [(12.290, 6.638), (12.666, 8.005), (13.024, 8.765)]),
    ("SepFormer-reduced", SF, DASH,
     [(9.765, 5.391), (10.672, 6.501), (10.767, 7.044)]),
    ("MossFormer2-full", MF2, "-",
     [(14.035, 8.071), (14.124, 9.003), (14.171, 9.845)]),
    ("MossFormer2-matched", MF2, DASH,
     [(12.160, 7.150), (12.879, 8.296), (13.293, 8.948)]),
    ("SPMamba-full", SPM, "-",
     [(10.206, 5.983), (10.584, 7.029), (10.092, 6.793)]),
    ("SPMamba-reduced", SPM, DASH,
     [(10.163, 6.760), (10.094, 7.414), (10.611, 8.038)]),
]
SIZES = [18, 32, 50]  # 16k -> 64k
ISO_GAPS = [3, 4, 5, 6]


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
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    xlim, ylim = (9.3, 14.7), (5.0, 10.4)

    # iso-gap diagonals val = train - g
    for g in ISO_GAPS:
        xs = [max(xlim[0], ylim[0] + g), min(xlim[1], ylim[1] + g)]
        ax.plot(xs, [x - g for x in xs], color=GRID, lw=0.8, zorder=1)
        # g=3,4 labelled near the top edge; g=5,6 near the bottom edge,
        # where the lower-right region is empty (top-edge labels for those
        # collide with the MossFormer2-full trajectory)
        if g <= 4:
            lx = min(xlim[1] - 0.55, ylim[1] + g - 0.35)
        else:
            lx = ylim[0] + g + 0.55
        ax.annotate(f"luka {g} dB", (lx, lx - g), (0, 3),
                    textcoords="offset points", fontsize=7.0, color=INK2,
                    rotation=32, ha="center")

    for label, color, ls, pts in SERIES:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=1.6, ls=ls, zorder=2, label=label)
        for (x, y), sz in zip(pts, SIZES):
            ax.scatter([x], [y], s=sz, color=color, edgecolor="white",
                       linewidth=0.8, zorder=3)
        # arrowhead on the last segment
        ax.annotate("", xy=pts[2], xytext=pts[1],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                                    mutation_scale=11, shrinkB=4), zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("SI-SDR na zbiorze treningowym [dB]")
    ax.set_ylabel("SI-SDR na zbiorze walidacyjnym [dB]")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)
    ax.legend(loc="upper left", frameon=False, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_train_vs_val.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_train_vs_val.{{png,pdf}}")


if __name__ == "__main__":
    main()
