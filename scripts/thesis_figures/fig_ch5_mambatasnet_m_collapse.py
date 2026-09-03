"""Rysunek (ch5 §5.3) — Mamba-TasNet-M training instability.

Train and validation SI-SDR per epoch for run `prur4hgw` (warm-silence-254, the
2026-07-20 bf16 re-validation of the M sweep winner grateful-sweep-14: lr 9.76e-4,
grad_clip 0.652, 15.6M params, PolSESS_C_both, seed 42). The run climbs healthily
to epoch 8 (train 7.78, val 3.28), then collapses at epoch 9 (val -2.71, train
5.15 -> 1.13 by ep 10) with ZERO NaN batches and at CONSTANT learning rate — the
first scheduler cut comes only at epoch 12, after the collapse. Recovery is slow
and truncated by early stopping at epoch 18 (patience counts from the pre-collapse
best, so a -6 dB excursion cannot be repaid in time).

Why this run: it is the clean case. The April fp16 twin (scarlet-wildflower-243,
grateful-sweep-14 seed 42) shows the same shape — healthy to ep 10, collapse ep 11,
truncated at ep 20 — so the collapse is config-intrinsic, NOT the fp16 AMP-dispatch
bug (2/2 across precisions, different hardware). See thesis-log/05_mamba_tasnet_hpo.md.

History frozen 2026-08-02 in data/curve_mambatasnet_m_collapse.csv.

Usage: python scripts/thesis_figures/fig_ch5_mambatasnet_m_collapse.py [--out-dir DIR]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "curve_mambatasnet_m_collapse.csv"
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch05"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
MT = "#1baf7a"   # Mamba-TasNet's fixed colour in the ch5 palette
TRAIN = "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ep, tr, va = [], [], []
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            tr.append(float(row["train_si_sdr"]))
            va.append(float(row["val_si_sdr"]))

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.plot(ep, tr, color=TRAIN, lw=1.8, zorder=2, label="zbiór treningowy")
    ax.plot(ep, va, color=MT, lw=1.8, ls=(0, (4, 2)), zorder=2,
            label="zbiór walidacyjny")

    # mark the collapse epoch on both curves
    ci = ep.index(9)
    ax.scatter([9, 9], [tr[ci], va[ci]], s=40, color=[TRAIN, MT],
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.annotate("załamanie treningu\n(epoka 9, stałe tempo uczenia)",
                (9, va[ci]), (14, -6), "data", textcoords="offset points",
                ha="left", va="top", fontsize=8.0, color=INK,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8,
                                shrinkA=3, shrinkB=4))

    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(-4.2, 9.0)
    ax.set_xticks(range(2, 19, 2))
    ax.set_yticks(range(-4, 9, 2))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("SI-SDR [dB]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)
    # upper right: empty after the collapse (train never regains its ep-8 peak),
    # and it keeps clear of the collapse annotation in the lower half
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.8,
                    handlelength=2.6, borderaxespad=0.8)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch5_mambatasnet_m_collapse.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch5_mambatasnet_m_collapse.{{png,pdf}}")


if __name__ == "__main__":
    main()
