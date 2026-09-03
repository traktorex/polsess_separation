"""Rysunek (ch5 §5.6.1) — dwupanelowy: poziom treningowego SI-SDR + luka train-val.

Two-panel replacement for fig_ch5_traingap.py (which stays on disk, single-panel,
as the revert path). Same six runs, same palette/linestyles as
fig_ch5_capacity_vs_data.py; stacked, shared x:

  (a) top    — train SI-SDR at each run's best-val epoch (the "ceiling" the
               model can reach on the data it has seen). Shows the tier claim
               of §5.6.1 ¶3: ~9-11 dB for 1-6M-param builds, ~12-13 dB for the
               26M ones, ~14 dB for the 56M MossFormer2-full — i.e. the level
               tracks parameter count, not architecture family, and it moves
               far less with data than the val curve of Rys 5.4 does.
  (b) bottom — train - val at the same epoch (identical content to the old
               single-panel figure).

Why stacked and not beside Rys 5.4: (a) and (b) sum to Rys 5.4's val curves,
so the three-way redundancy is only readable if the two *mechanism* panels sit
together and the *result* figure stays separate. Panels also read in the order
§5.6.1 ¶3 argues them (level first, then gap).

Sources: W&B histories, train_si_sdr and val_si_sdr at the best-val epoch,
re-harvested 2026-08-21 (all 17 runs re-verified against the two sibling
scripts — every value matched; `jcwv8yyh`'s history yields 7.0293, i.e. the
checkpoint value, NOT its bad crashed-badge summary):

  SepFormer          12.290/6.638 `1tvoj7dk` · 12.666/8.005 `gckob0q1` ·
                     13.024/8.765 `bym7223m` (s42 single seed, as in Rys 5.4)
  SepFormer-reduced   9.765/5.391 `ky8x5ukl` · 10.672/6.501 `k7zcc2yn` ·
                     10.767/7.044 `8eyit3bu`
  MossFormer2-full   14.035/8.071 `fy0hz7on` · 14.124/9.003 `8xwepids` ·
                     14.171/9.845 `f0pt0kly`
  MossFormer2-matched 12.160/7.150 (log-only, weights lost — epoch-mean INFO
                     lines of checkpoints/mossformer2/SB/
                     mossformer2-matched_16k_CLOBBERED_DO_NOT_EVAL/log, NOT the
                     per-batch tqdm postfix) · 12.879/8.296 `8mv4hg40` ·
                     13.293/8.948 `q3hzv58q`
  SPMamba-full       10.206/5.983 `vcxw03o8` · 10.584/7.029 `jcwv8yyh` ·
                     10.092/6.793 `qqaqglkn`
  SPMamba-reduced     9.234/6.100 `m5rqevw7` · 9.561/6.594 `jno7mow0` ·
                      9.298/6.758 `lxmkkf7y`  (the BAZOWA paper-lr N2 runs —
                     like-for-like partner of SPMamba-full; the tuned N1 gaps
                     3.40/2.68/2.57 belong to Tab 5.Y)

NB the gap carries a constant offset: train SI-SDR is the epoch mean over the
seven-variant training mixture, val is {SER,SE} only — absolute values are
shifted, comparisons across runs/scales are valid (§5.1.2 states this once).
Panel (a) inherits that offset, so its levels are likewise comparative.

Point labels: panel (b) labels every point (the prose quotes gap values and
their drops); panel (a) labels only the 16 and 64 tys. columns — the endpoints
carry the level-and-drift claim, and the 32 tys. column has two near-coincident
pairs (10.67/10.58, 12.88/12.67) that no offset scheme disambiguates cleanly.

`--label-sheet` additionally emits rys_ch5_trainlevel_gap_labels_32k.png — the
six 32 tys. panel-(a) labels the automatic placement leaves out, on a
transparent canvas in the figure's exact font/size/colour (DejaVu Sans 7.8 pt,
#0b0b0b, 300 dpi), plus a .txt with the value-to-curve mapping. Paste them into
the .xcf and position by hand; grey side notes name the curve and are NOT part
of the labels.

Usage: python scripts/thesis_figures/fig_ch5_trainlevel_gap.py [--out-dir DIR]
                                                               [--label-sheet]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SF, SPM, MF2 = "#eb6834", "#2a78d6", "#7b52ab"  # fixed ch5 palette
DASH = (0, (4, 2))

# (label, color, linestyle, [(size_k, train, val), ...])
SERIES = [
    ("SepFormer", SF, "-",
     [(16, 12.290, 6.638), (32, 12.666, 8.005), (64, 13.024, 8.765)]),
    ("SepFormer-reduced", SF, DASH,
     [(16, 9.765, 5.391), (32, 10.672, 6.501), (64, 10.767, 7.044)]),
    ("MossFormer2-full", MF2, "-",
     [(16, 14.035, 8.071), (32, 14.124, 9.003), (64, 14.171, 9.845)]),
    ("MossFormer2-matched", MF2, DASH,
     [(16, 12.160, 7.150), (32, 12.879, 8.296), (64, 13.293, 8.948)]),
    ("SPMamba-full", SPM, "-",
     [(16, 10.206, 5.983), (32, 10.584, 7.029), (64, 10.092, 6.793)]),
    ("SPMamba-reduced", SPM, DASH,
     [(16, 9.234, 6.100), (32, 9.561, 6.594), (64, 9.298, 6.758)]),
]

DPI = 300
ABOVE, BELOW = (0, 5), (0, -12)
# Crowded corners at 64 tys.: push the label to the RIGHT of the marker
# (xlim runs to 78, so there is room) instead of stacking above/below.
RIGHT = (21, -3)

# --- panel (a): train level -------------------------------------------------
# Labelled columns only; everything else defaults to ABOVE.
TRAIN_LABEL_SIZES = {16, 64}
TRAIN_SPECIAL = {
    ("MossFormer2-matched", 16): BELOW,   # 12,2 sits 0,13 under SepFormer 12,3
    ("SepFormer", 64): BELOW,             # 13,0 sits 0,27 under MF2-matched 13,3
    ("SepFormer-reduced", 16): BELOW,     # 9,8 vs SPMamba-full 10,2 above it
    ("SPMamba-full", 64): RIGHT,          # 10,1 boxed in by 10,8 above, 9,3 below
    ("SPMamba-reduced", 16): BELOW,       # 9,2 under SepFormer-reduced 9,8
}

# --- panel (b): gap ---------------------------------------------------------
# The two MF2 curves coincide at 64 tys. (4.33/4.35) — one shared "4,3" above
# the pair (drawn on -matched), the -full point unlabelled there.
GAP_SKIP = {("MossFormer2-full", 64)}
GAP_SPECIAL = {
    ("SepFormer", 64): RIGHT,          # 4,3 between the MF2 pair and 3,7
    ("MossFormer2-matched", 32): BELOW,
    ("SepFormer-reduced", 32): BELOW,
    ("SepFormer", 16): BELOW,          # 5,7 vs MF2-full 6,0
    ("SPMamba-full", 16): BELOW,       # 4,2 sits 0,15 under SF-reduced 4,4
}


def pl(value: float, decimals: int) -> str:
    """Polish decimal comma, fixed precision."""
    return f"{value:.{decimals}f}".replace(".", ",")


def draw(ax, series, value_of, *, decimals, special, label_sizes=None,
         skip=frozenset()):
    for label, color, ls, pts in series:
        xs = [x for x, _, _ in pts]
        ys = [value_of(t, v) for _, t, v in pts]
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, zorder=2, label=label)
        ax.scatter(xs, ys, s=30, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3)
        for x, y in zip(xs, ys):
            if (label, x) in skip:
                continue
            if label_sizes is not None and x not in label_sizes:
                continue
            dx, dy = special.get((label, x), ABOVE)
            ax.annotate(pl(y, decimals), (x, y), (dx, dy), "data",
                        textcoords="offset points", ha="center",
                        fontsize=7.8, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_xlim(13.5, 78)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16 tys.", "32 tys.", "64 tys."])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)


def label_sheet(out_dir: Path, fig, ax_top, size_k: int = 32) -> None:
    """Loose labels for one column of panel (a), for hand-placement in GIMP.

    Also dumps each marker's pixel position in the saved 300-dpi PNG, so the
    labels can be placed exactly rather than by eye. Offsets used elsewhere in
    the figure, converted to pixels: ABOVE = 21 px above the marker centre,
    BELOW = 50 px below it, horizontally centred on the marker.
    """
    rows = sorted(
        ((label, t) for label, _c, _ls, pts in SERIES
         for x, t, _v in pts if x == size_k),
        key=lambda r: -r[1])

    # Data -> pixel in the tight-bbox PNG written by main().
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = plt.rcParams["savefig.pad_inches"]
    px_of = {}
    for label, t in rows:
        xd, yd = ax_top.transData.transform((size_k, t))
        px_of[label] = (round((xd / fig.dpi - (bb.x0 - pad)) * DPI),
                        round(((bb.y1 + pad) - yd / fig.dpi) * DPI))

    fig = plt.figure(figsize=(2.4, 0.32 * len(rows) + 0.2))
    fig.patch.set_alpha(0.0)
    for i, (label, t) in enumerate(rows):
        y = 1.0 - (i + 0.5) / len(rows)
        fig.text(0.06, y, pl(t, 1), fontsize=7.8, color=INK,
                 ha="left", va="center")
        fig.text(0.42, y, f"← {label}", fontsize=6, color="#a8a7a0",
                 ha="left", va="center")

    png = out_dir / f"rys_ch5_trainlevel_gap_labels_{size_k}k.png"
    fig.savefig(png, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)

    txt = png.with_suffix(".txt")
    txt.write_text(
        f"Etykiety panelu (a) dla {size_k} tys. — do wklejenia ręcznie.\n"
        "Font: DejaVu Sans, 7.8 pt @ 300 dpi (~32 px em), kolor #0b0b0b.\n"
        "Kolejność malejąco; szare podpisy w PNG NIE należą do etykiet.\n\n"
        "Pozycje znaczników w rys_ch5_trainlevel_gap.png (300 dpi, "
        "origin = lewy górny róg):\n"
        "  etykieta NAD znacznikiem  = środek tekstu 21 px wyżej\n"
        "  etykieta POD znacznikiem  = środek tekstu 50 px niżej\n"
        "  w poziomie: wyśrodkowana na znaczniku\n"
        "UWAGA: SepFormer-reduced i SPMamba-full dzieli tylko ~12 px w pionie "
        "— to ta para wymaga ręcznego rozsunięcia.\n\n"
        + "".join(f"{pl(t, 1):>6}  x={px_of[label][0]:>5}  y={px_of[label][1]:>5}"
                  f"  {label}\n" for label, t in rows),
        encoding="utf-8")
    print(f"written: {png}\nwritten: {txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--label-sheet", action="store_true",
                    help="also emit the loose 32 tys. panel-(a) labels")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.6, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.14})

    draw(ax_top, SERIES, lambda t, v: t, decimals=1,
         special=TRAIN_SPECIAL, label_sizes=TRAIN_LABEL_SIZES)
    ax_top.set_ylim(8.6, 14.9)
    ax_top.set_yticks([9, 10, 11, 12, 13, 14])
    ax_top.set_ylabel("SI-SDR trening [dB]")

    draw(ax_bot, SERIES, lambda t, v: t - v, decimals=1,
         special=GAP_SPECIAL, skip=GAP_SKIP)
    ax_bot.set_ylim(2.2, 6.6)
    ax_bot.set_ylabel("SI-SDR trening − walidacja [dB]")
    ax_bot.set_xlabel("Rozmiar zbioru treningowego")
    ax_bot.legend(loc="upper right", frameon=False, fontsize=8.0)

    # (a)/(b) markers so the caption can address the panels.
    for ax, tag in ((ax_top, "(a)"), (ax_bot, "(b)")):
        ax.annotate(tag, (0.0, 1.0), (0, 8), "axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9, color=INK2)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_trainlevel_gap.{ext}",
                    dpi=DPI, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_trainlevel_gap.{{png,pdf}}")

    if args.label_sheet:
        label_sheet(args.out_dir, fig, ax_top)


if __name__ == "__main__":
    main()
