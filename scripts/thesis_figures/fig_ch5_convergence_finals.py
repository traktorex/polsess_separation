"""Rysunek (ch5 §5.6) — 128k final trainings on a normalized clock.

Val SI-SDRi (all 8 variants, 1000-mix finale val set) per epoch against
ESTIMATED training time on one reference GPU (RTX 4070): x = epoch x s/epoch,
s/epoch = benchmark train-only s/1000 samples x 128 (benchmark_training.csv,
2026-07-30 rerun). Throughput row per run matches how the run processed samples:
SepFormer final ran bs 1 + grad-accum 2 -> the bs=1 row (97.15); MF2-matched ran
bs 2 -> trained row (96.40); MF2-full ran bs 1 -> trained row (238.98).
Raw wall-clock is unprintable (mixed author hardware).

Runs (histories frozen 2026-08-02 in data/curves_128k_finals.csv):
  - sepformer_128k_final_42   `8q3yeeoz` (killed ep73 post-convergence, best 16.00)
  - mossformer2_matched final `6xuv1tob` (early-stopped ep56, best 17.04 @ ep46)
  - mossformer2_full final    `cw1qdo1h` (seed 123; ended by author ep53, best 18.00 @ ep50)

Usage: python scripts/thesis_figures/fig_ch5_convergence_finals.py [--out-dir DIR]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "curves_128k_finals.csv"
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
MF2 = "#7b52ab"  # MossFormer2 joins the fixed per-architecture palette here
STYLE = {
    "sepformer_final": dict(color="#eb6834", ls="-", label="SepFormer"),
    "mf2_matched_final": dict(color=MF2, ls=(0, (4, 2)), label="MossFormer2-matched"),
    "mf2_full_final": dict(color=MF2, ls="-", label="MossFormer2-full"),
}
SEC_PER_EPOCH = {
    "sepformer_final": 97.15 * 128,   # bs=1 row (run used bs1 + accum 2)
    "mf2_matched_final": 96.40 * 128,  # trained row, bs 2
    "mf2_full_final": 238.98 * 128,    # trained row, bs 1
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["label"], []).append(
                (int(row["epoch"]), float(row["val_si_sdri"])))

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in ("sepformer_final", "mf2_matched_final", "mf2_full_final"):
        pts = sorted(curves[label])
        days = [e * SEC_PER_EPOCH[label] / 86400 for e, _ in pts]
        vals = [v for _, v in pts]
        st = STYLE[label]
        # legend rather than end-of-line labels: "MossFormer2-full" is long and its
        # curve ends at max x, so a direct label would spill past the axes and
        # widen the saved image (bbox_inches="tight" grows to include text).
        ax.plot(days, vals, color=st["color"], lw=1.8, ls=st["ls"], zorder=2,
                label=st["label"])
        bi = max(range(len(vals)), key=vals.__getitem__)
        ax.scatter(days[bi], vals[bi], s=42, color=st["color"],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{vals[bi]:.2f}".replace(".", ","),
                    (days[bi], vals[bi]), (0, 9), "data",
                    textcoords="offset points", ha="center",
                    fontsize=8.0, color=INK)

    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.8,
                    handlelength=2.6, borderaxespad=1.0)
    for txt in leg.get_texts():
        txt.set_color(INK)

    ax.set_xlim(0, 20.5)
    ax.set_ylim(7.5, 19.0)
    ax.set_xticks(range(0, 21, 2))
    ax.set_yticks(range(8, 19, 2))
    ax.set_xlabel("Szacowany czas treningu na RTX 4070 [dni]")
    ax.set_ylabel("SI-SDRi [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_convergence_finals.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_convergence_finals.{{png,pdf}}")


if __name__ == "__main__":
    main()
