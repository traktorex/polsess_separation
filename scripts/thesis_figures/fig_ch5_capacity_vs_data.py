"""Rysunek (ch5 §5.6) — capacity vs data: val SI-SDR across training-set sizes.

REGENERATED 2026-08-21 (SPMamba-reduced swapped to the bazowa N2 runs, see
below); previous regeneration 2026-08-20 after the SPMamba-full row and the
SepFormer-reduced trio completed (runs brief: thesis-log/09_ch5_scaling_runs.md). Six curves,
three full-vs-reduced pairs, ALL protocol-matched: flat 7-variant diet, seed
42, C_new_64 subsets, {SER,SE} 400-mix val SI-SDR. The 8-tys. pilot column of
the previous version is gone (with it the corpus-boundary caveat — moot now);
solid = full/large build, dashed = reduced/matched build, one color per
family. Previous version (Tabela 5.16 draft form, mixed schedules, dagger
marker) lives in git history.

Sources per point (single seed 42 unless noted):
  SepFormer-full      6.64 `1tvoj7dk` / 8.01 `gckob0q1` / 8.77 `bym7223m`
                      (s42 SINGLE seed, train 13.0239 @ep52 — SWAPPED IN
                      2026-08-21 so the caption's "pojedyncze ziarno" is
                      literally true for every point. The 3-seed mean 8.75
                      ± 0.11, mean(8.765/8.840/8.630) over bym7223m,z0omra18,
                      xmwu4tlw, stays in Tab 5.7 and is the number the prose
                      quotes there.)
  SepFormer-reduced   5.39 `ky8x5ukl` / 6.50 `k7zcc2yn` / 7.04 `8eyit3bu`
  MossFormer2-full    8.07 `fy0hz7on` / 9.00 `8xwepids` / 9.85 `f0pt0kly`
  MossFormer2-matched 7.15 (log-only, weights lost) / 8.30 `8mv4hg40` /
                      8.95 `q3hzv58q`
  SPMamba-full        5.98 `vcxw03o8` / 7.03 `jcwv8yyh` (checkpoint 7.0293 —
                      NOT the W&B summary, crashed-badge artifact) / 6.79
                      `qqaqglkn` (non-monotone 32k→64k step −0.24 dB:
                      within-noise per thesis-log/10 — anneal-artifact
                      reading refuted)
  SPMamba-reduced     6.10 `m5rqevw7` / 6.59 `jno7mow0` (checkpoint 6.5935) /
                      6.76 `lxmkkf7y` — the BAZOWA (paper-lr 1e-3) reduced runs,
                      N2 in the brief. SWAPPED IN 2026-08-21 (author ruling):
                      the like-for-like partner of SPMamba-full is the paper-recipe
                      reduced run (same lr 1e-3 / lr_factor 0.5; wd, clip, patience,
                      early-stop differ — all second-order per §5.3), NOT the tuned
                      N1 run (6.76 / 7.42 / 8.04 — lives in Tab 5.7 and is quoted
                      in prose). With the swap every curve in the figure is a
                      paper-recipe config. NB all three N2 runs were author-stopped
                      before ES15 (7/15, 10/15, 7/15 no-improve) — mildly
                      right-censored; any creep would move Δ further against full.

Point labels are explicit strings matched to the chapter tables (avoids
float-repr rounding drift, e.g. 9.845 -> "9,84").

Usage: python scripts/thesis_figures/fig_ch5_capacity_vs_data.py [--out-dir DIR]
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
    # (label, color, linestyle, [(size_k, value, printed_label), ...])
    ("SepFormer", SF, "-",
     [(16, 6.638, "6,64"), (32, 8.005, "8,01"), (64, 8.765, "8,77")]),
    ("SepFormer-reduced", SF, DASH,
     [(16, 5.391, "5,39"), (32, 6.501, "6,50"), (64, 7.044, "7,04")]),
    ("MossFormer2-full", MF2, "-",
     [(16, 8.071, "8,07"), (32, 9.003, "9,00"), (64, 9.845, "9,85")]),
    ("MossFormer2-matched", MF2, DASH,
     [(16, 7.15, "7,15"), (32, 8.296, "8,30"), (64, 8.948, "8,95")]),
    ("SPMamba-full", SPM, "-",
     [(16, 5.983, "5,98"), (32, 7.029, "7,03"), (64, 6.793, "6,79")]),
    ("SPMamba-reduced", SPM, DASH,
     [(16, 6.100, "6,10"), (32, 6.594, "6,59"), (64, 6.758, "6,76")]),
]
# Label placement: ABOVE/BELOW the marker, kept tight to the point.
ABOVE, BELOW = (0, 5), (0, -12)
# Points labelled BELOW (everything else defaults to ABOVE):
BELOW_POINTS = {
    ("SepFormer", 16), ("SepFormer-reduced", 16), ("SPMamba-full", 16),
    ("SepFormer", 32), ("SepFormer-reduced", 32),
    ("SepFormer", 64),
}
# Per-point (dx, dy) overrides in points for congested spots:
#  32 tys.: SPMamba-reduced 6,59 sits 0,09 above SepFormer-reduced 6,50 -> label
#           it ABOVE (SepFormer-reduced keeps BELOW).
#  64 tys.: SPMamba-full 6,79 and SPMamba-reduced 6,76 coincide, with
#           SepFormer-reduced 7,04 just above -> push both SPMamba labels to the
#           RIGHT of the markers (there is room: xlim runs to 78), full level,
#           reduced slightly below; 7,04 stays ABOVE.
#  16 tys.: SPMamba-full 5,98 BELOW, SPMamba-reduced 6,10 ABOVE (default).
SPECIAL = {
    ("SPMamba-full", 64): (22, -2),
    ("SPMamba-reduced", 64): (22, -13),
}


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
            dx, dy = SPECIAL.get(
                (label, x), BELOW if (label, x) in BELOW_POINTS else ABOVE)
            ax.annotate(txt, (x, y), (dx, dy), "data",
                        textcoords="offset points", ha="center",
                        fontsize=7.8, color=INK)

    ax.set_xlim(13.5, 78)
    ax.set_ylim(5.0, 10.5)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16 tys.", "32 tys.", "64 tys."])
    ax.minorticks_off()
    ax.set_xlabel("Rozmiar zbioru treningowego")
    ax.set_ylabel("SI-SDR [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)
    ax.legend(loc="upper left", frameon=False, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_capacity_vs_data.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_capacity_vs_data.{{png,pdf}}")


if __name__ == "__main__":
    main()
