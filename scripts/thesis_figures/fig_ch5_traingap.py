"""Rysunek (ch5 §5.6, kandydat) — train-val gap (overfitting proxy) vs data size.

Twin of fig_ch5_capacity_vs_data.py: same six runs, same style, but y = the
train-val SI-SDR difference at each run's best-val epoch (the Tab 5.Y metric).
NB the gap contains a constant offset: train SI-SDR is the epoch mean over the
full 7-variant training mixture, val is {SER,SE} only — absolute values are
shifted, comparisons across runs/scales are valid (the chapter's 5.1.2 states
this once).

Sources: W&B histories (train_si_sdr at the best-val epoch), harvested
2026-08-20 — same runs as the capacity figure. Exception: MossFormer2-matched
@16k is the log-only run (weights lost); its pair Train 12.16 / Val 7.15 @
display-ep26 comes from the surviving training log at
checkpoints/mossformer2/SB/mossformer2-matched_16k_CLOBBERED_DO_NOT_EVAL/log
(INFO epoch-mean lines, NOT the per-batch tqdm postfix).

Values (train − val, 2 dp): SepFormer 5.65/4.66/4.26 · SepFormer-reduced
4.37/4.17/3.72 · MF2-full 5.96/5.12/4.33 · MF2-matched 5.01/4.58/4.35 ·
SPMamba-full 4.22/3.56/3.30 · SPMamba-reduced 3.13/2.97/2.54.

SPMamba-reduced = the BAZOWA (paper-lr) N2 runs `m5rqevw7`/`jno7mow0`/`lxmkkf7y`
(train@best 9.23/9.56/9.30 — W&B histories, harvested 2026-08-21), swapped in
2026-08-21 to mirror fig_ch5_capacity_vs_data.py (author ruling: like-for-like
partner of SPMamba-full). The tuned N1 gaps 3.40/2.68/2.57 are Tab 5.Y's
"po strojeniu (I)" row and stay there.

Usage: python scripts/thesis_figures/fig_ch5_traingap.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SF, SPM, MF2 = "#eb6834", "#2a78d6", "#7b52ab"  # fixed ch5 palette
DASH = (0, (4, 2))

SERIES = [
    ("SepFormer", SF, "-",
     [(16, 5.65, "5,7"), (32, 4.66, "4,7"), (64, 4.26, "4,3")]),
    ("SepFormer-reduced", SF, DASH,
     [(16, 4.37, "4,4"), (32, 4.17, "4,2"), (64, 3.72, "3,7")]),
    ("MossFormer2-full", MF2, "-",
     [(16, 5.96, "6,0"), (32, 5.12, "5,1"), (64, 4.33, "")]),
    ("MossFormer2-matched", MF2, DASH,
     [(16, 5.01, "5,0"), (32, 4.58, "4,6"), (64, 4.35, "4,3")]),
    ("SPMamba-full", SPM, "-",
     [(16, 4.22, "4,2"), (32, 3.56, "3,6"), (64, 3.30, "3,3")]),
    ("SPMamba-reduced", SPM, DASH,
     [(16, 3.13, "3,1"), (32, 2.97, "3,0"), (64, 2.54, "2,5")]),
]
ABOVE, BELOW = (0, 5), (0, -12)
# 64 tys. is congested: SF-full 4.26 / MF2-full 4.33 / MF2-matched 4.35 nearly
# coincide — spread their labels left/below/right.
SPECIAL = {
    ("SepFormer", 64): (0, -12),
    ("MossFormer2-matched", 32): (0, -12),
    ("SepFormer-reduced", 32): (0, -12),
    ("SepFormer", 16): (0, -12),          # 5,65 vs MF2-full 5,96
    ("SPMamba-full", 16): (0, -12),       # 4,22 sits 0.15 under SFred 4,37
}
# The two MF2 curves coincide at 64 tys. (4.33/4.35) — one shared "4,3" above
# the pair (drawn on -matched), the -full point unlabeled there.


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
    ax.set_xscale("log", base=2)

    for label, color, ls, pts in SERIES:
        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, zorder=2, label=label)
        ax.scatter(xs, ys, s=30, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3)
        for x, y, txt in pts:
            if not txt:
                continue
            dx, dy = SPECIAL.get((label, x), ABOVE)
            ax.annotate(txt, (x, y), (dx, dy), "data",
                        textcoords="offset points", ha="center",
                        fontsize=7.8, color=INK)

    ax.set_xlim(13.5, 78)
    ax.set_ylim(2.2, 6.6)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16 tys.", "32 tys.", "64 tys."])
    ax.minorticks_off()
    ax.set_xlabel("Rozmiar zbioru treningowego")
    ax.set_ylabel("SI-SDR trening − walidacja [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_traingap.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_traingap.{{png,pdf}}")


if __name__ == "__main__":
    main()
