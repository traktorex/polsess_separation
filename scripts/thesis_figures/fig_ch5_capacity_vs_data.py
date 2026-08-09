"""Rysunek (ch5 §5.7) — capacity vs data: val SI-SDR across training-set sizes.

Figure form of Tabela 5.16 (draft 2026-08-05). One curve per architecture,
x = training-set size (log2: 8/16/32/64 tys. mixtures), y = val SI-SDR
({SER,SE}, the shared 400-mix validation set of phases 1-2).

Cell rule (matches the table): best configuration known for the architecture
at that scale as reported earlier in the chapter; multi-seed mean where seeds
exist. Sources per point:
  SepFormer   6.16 (tuned, 3-seed mean, Tab 5.5) / 6.655 `r634a0ua` /
              8.005 `gckob0q1` / 8.75 (baseline flat, 3-seed mean, Tab 5.7)
  SPMamba     5.94 (tuned, 2-seed mean, Tab 5.5) / 6.653 `knkhn29o` /
              7.425 `rz1v3gwe` / 7.72 `io29fz27` (progressive; flat `7ifvwekd`
              8.038 shown as hollow marker — the Tabela 5.16 dagger)
  MF2-matched 7.15 (log-only) / 8.296 `8mv4hg40` / 8.948 `q3hzv58q`
  MF2-full    8.071 `fy0hz7on` / 9.003 `8xwepids` / 9.845 `f0pt0kly`

The 8-tys. column is the pilot corpus generation (unique-mixture count);
16-64 tys. are subsets of the corrected C_new_64. The corpus boundary is
marked only by the x-tick label (author ruling 2026-08-06: no dashed
segments) — the caption carries the caveat.

Usage: python scripts/thesis_figures/fig_ch5_capacity_vs_data.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
MF2 = "#7b52ab"  # same fixed palette as the other ch5 figures
SERIES = [
    # (label, color, linestyle, [(size_k, value), ...])
    ("SepFormer", "#eb6834", "-",
     [(8, 6.16), (16, 6.655), (32, 8.005), (64, 8.75)]),
    ("SPMamba-reduced", "#2a78d6", "-",
     [(8, 5.94), (16, 6.653), (32, 7.425), (64, 7.72)]),
    ("MossFormer2-matched", MF2, (0, (4, 2)),
     [(16, 7.15), (32, 8.296), (64, 8.948)]),
    ("MossFormer2-full", MF2, "-",
     [(16, 8.071), (32, 9.003), (64, 9.845)]),
]
# per-point label offsets (x pt, y pt), keyed (label, size_k) where the default
# above-the-marker placement would collide (SepFormer/SPMamba tie at 16 tys.)
OFFSETS = {
    ("SepFormer", 16): (0, 8),
    ("SPMamba-reduced", 16): (0, -13),
    ("SepFormer", 8): (0, 8),
    ("SPMamba-reduced", 8): (0, -13),
    ("SepFormer", 32): (0, -13),
    ("SepFormer", 64): (0, -13),
    ("SPMamba-reduced", 64): (0, -13),
}
FLAT_SPMAMBA_64K = 8.038  # `7ifvwekd` — the table's dagger footnote


def fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


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
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, zorder=2, label=label)
        ax.scatter(xs, ys, s=30, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3)
        for x, y in pts:
            dx, dy = OFFSETS.get((label, x), (0, 8))
            ax.annotate(fmt(y), (x, y), (dx, dy), "data",
                        textcoords="offset points", ha="center",
                        fontsize=7.8, color=INK)

    # SPMamba 64k without the progressive schedule (Tabela 5.16 dagger)
    ax.scatter([64], [FLAT_SPMAMBA_64K], s=30, facecolor="white",
               edgecolor="#2a78d6", linewidth=1.2, zorder=3)
    ax.annotate(f"{fmt(FLAT_SPMAMBA_64K)} (bez harm.)", (64, FLAT_SPMAMBA_64K),
                (10, -3), "data", textcoords="offset points", ha="left",
                fontsize=7.4, color="#2a78d6")

    ax.set_xlim(6.9, 88)
    ax.set_ylim(5.4, 10.4)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels(["8 tys.\n(korpus pilotażowy)", "16 tys.", "32 tys.",
                        "64 tys."])
    ax.minorticks_off()
    ax.set_xlabel("Rozmiar zbioru treningowego")
    ax.set_ylabel("SI-SDR [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)
    ax.legend(loc="lower right", frameon=False, fontsize=8.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_capacity_vs_data.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_capacity_vs_data.{{png,pdf}}")


if __name__ == "__main__":
    main()
