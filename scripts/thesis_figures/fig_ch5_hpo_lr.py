"""Rysunek (ch5 §5.3.2) — lr response shape across three HPO sweeps.

Three scatter panels, val SI-SDR (best_val_sisdr) vs learning rate (log x),
shared y so the magnitude of the lr effect reads directly:
  A  SepFormer stage2-8k  `qqjh7cvm` — wide search: flat region, then the
     instability cliff (every run above ~4e-4 lands below 0 dB and never
     recovers despite ReduceLROnPlateau halvings).
  B  SepFormer posenc-16k `0r4w3ep2` — search narrowed inside the stable
     region: flat top (top-5 span 0.27 dB ~= seed sigma).
  C  ConvTasNet stage2-8k `71wtfegp` — no cliff inside the searched range;
     the same flatness in a different family.
Filled = finished; hollow = terminated early (Hyperband kill / crash), so
hollow values are best-so-far lower bounds. The point of the figure is the
threshold character of lr, not per-architecture ranges (author 2026-08-08).

Data: run tables exported from W&B on 2026-08-08 by
thesis/thesis-log/sweep_plan/ch5_hpo_analysis/export_sweep_data.py.

Usage: venv/bin/python scripts/thesis_figures/fig_ch5_hpo_lr.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "thesis" / "thesis-log" / "sweep_plan" / "ch5_hpo_analysis"
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SEPFORMER, CONVTASNET = "#eb6834", "#eda100"
PANELS = [
    # (csv sweep name, panel title, color, x-ticks, x-tick labels)
    ("sepformer-stage2-8k", "SepFormer — zakres szeroki", SEPFORMER,
     [1e-4, 3e-4, 1e-3, 2e-3],
     ["$10^{-4}$", "$3{\\cdot}10^{-4}$", "$10^{-3}$", "$2{\\cdot}10^{-3}$"]),
    ("sepformer-hpo-posenc-16k", "SepFormer — zakres zawężony", SEPFORMER,
     [1e-4, 2e-4, 4e-4],
     ["$10^{-4}$", "$2{\\cdot}10^{-4}$", "$4{\\cdot}10^{-4}$"]),
    ("convtasnet-stage2-8k", "ConvTasNet", CONVTASNET,
     [4e-4, 7e-4, 1.2e-3],
     ["$4{\\cdot}10^{-4}$", "$7{\\cdot}10^{-4}$", "$1{,}2{\\cdot}10^{-3}$"]),
]


def load(sweep: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"runs__polsess-thesis-experiments__{sweep}.csv")
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    df["best_val_sisdr"] = pd.to_numeric(df["best_val_sisdr"], errors="coerce")
    return df.dropna(subset=["lr", "best_val_sisdr"])


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
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), sharey=True)

    for ax, (sweep, title, color, ticks, tick_labels) in zip(axes, PANELS):
        df = load(sweep)
        fin = df["state"] == "finished"
        ax.set_xscale("log")
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.scatter(df.loc[fin, "lr"], df.loc[fin, "best_val_sisdr"], s=24,
                   color=color, edgecolor="white", linewidth=0.6, zorder=3,
                   label="ukończone")
        ax.scatter(df.loc[~fin, "lr"], df.loc[~fin, "best_val_sisdr"], s=22,
                   facecolor="white", edgecolor=color, linewidth=1.0, zorder=2,
                   label="przerwane")
        ax.set_title(title, fontsize=9, color=INK, pad=6)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(length=3, color=AXIS)

    axes[0].set_ylabel("SI-SDR [dB]")
    axes[0].set_ylim(-5.0, 6.8)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7.8,
                   handletextpad=0.2, borderaxespad=0.1)
    fig.supxlabel("Współczynnik uczenia", fontsize=9, color=INK)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_hpo_lr_response.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_hpo_lr_response.{{png,pdf}}")


if __name__ == "__main__":
    main()
