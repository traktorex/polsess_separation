"""Rysunek (ch5 §5.5) — 64k convergence on a normalized clock: SepFormer vs SPMamba.

Val SI-SDR ({SER,SE}, 400-mix val) per epoch against ESTIMATED training time on
one reference GPU (RTX 4070): x = epoch x s/epoch, where s/epoch = the
benchmark's train-only s/1000-samples at the config batch (benchmark_training.csv,
2026-07-30 rerun) x 64. Raw wall-clock is unprintable — the actual runs sat on
heterogeneous GPUs (SepFormer scaling legs on rented cloud cards; hazard §7.7).

Runs (both FLAT 7-variant diet — schedule-matched; histories frozen 2026-08-02
in data/curves_64k_flat_pair.csv, guarding against W&B deletions):
  - SepFormer baseline s42  `bym7223m` (killed ep63, best 8.765 @ ep51)
  - SPMamba HPO config s42  `7ifvwekd` (crashed ep50, best 8.038 @ ep50 — lower bound)

SLOT: when the SPMamba-full 64k run lands (arc S3 / P2 — GPU-fault-blocked as of
2026-07-31), add its curve here (s/epoch = 577.80 x 64 s) — it either extends or
retires the "reduced SPMamba saturates" reading and sharpens the SepFormer-choice
justification this figure carries.

Usage: python scripts/thesis_figures/fig_ch5_convergence_64k.py [--out-dir DIR]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "curves_64k_flat_pair.csv"
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
# fixed per-architecture colors (same assignment as fig_ch5_baseline_scatter.py)
STYLE = {
    "sepformer_64k_flat": dict(color="#eb6834", label="SepFormer"),
    "spmamba_64k_flat": dict(color="#2a78d6", label="SPMamba"),
}
# benchmark train-only s/1000 samples at the config batch (bs 2 / bs 1) x 64
SEC_PER_EPOCH = {"sepformer_64k_flat": 80.75 * 64, "spmamba_64k_flat": 217.07 * 64}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["label"], []).append(
                (int(row["epoch"]), float(row["val_si_sdr"])))

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label, pts in curves.items():
        pts.sort()
        days = [e * SEC_PER_EPOCH[label] / 86400 for e, _ in pts]
        vals = [v for _, v in pts]
        st = STYLE[label]
        ax.plot(days, vals, color=st["color"], lw=1.8, zorder=2)
        bi = max(range(len(vals)), key=vals.__getitem__)
        ax.scatter(days[bi], vals[bi], s=42, color=st["color"],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{vals[bi]:.2f}".replace(".", ","),
                    (days[bi], vals[bi]), (0, 9), "data",
                    textcoords="offset points", ha="center",
                    fontsize=8.0, color=INK)
        ax.annotate(st["label"], (days[-1], vals[-1]), (7, -2), "data",
                    textcoords="offset points", ha="left",
                    fontsize=8.6, color=st["color"], fontweight="bold")

    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 9.6)
    ax.set_xticks(range(0, 10))
    ax.set_yticks(range(0, 10, 2))
    ax.set_xlabel("Szacowany czas treningu na RTX 4070 [dni]")
    ax.set_ylabel("SI-SDR [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_convergence_64k.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_convergence_64k.{{png,pdf}}")


if __name__ == "__main__":
    main()
