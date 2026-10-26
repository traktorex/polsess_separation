"""Rysunek (ch6 §6.4.3) — jakość separatora a wynik ASR potoku: trzy osie SI-SDR(i).

PRZYWRÓCONY 2026-09-04 (uwaga promotora: §6.4.3 zbyt gęsty). Rozdział wraca do
JEDNEGO rysunku z trzema panelami SI-SDR(i) — po jednym na źródło pomiaru, w
kolejności prozy — a PESQ i STOI dostają jedno zdanie w tekście. Trójka rysunków
per-źródło z 2026-08-28 (`fig_ch6_{indomain,oracle,squim}_pesq_stoi.py`, każdy
SI-SDR(i) | PESQ | STOI jednego źródła) zostaje jako zapis tamtego układu; nie
regenerować do rozdziału bez ponownej decyzji.

Panele (kolejność = kolejność w prozie §6.4.3, od najsłabszego predyktora):
  (a) SI-SDRi w dziedzinie treningowej — zbiór testowy PolSESS_128k, all-8:
      separatory z PolSESS_64k leżą tu w paśmie 11,5–13,5 dB razem z dobrymi,
      a ich cpWER jest o ~10 pkt gorsze,
  (b) SI-SDRi na nagraniach niesymulowanych Z referencjami — CLARIN_oracle:
      sonda intruzyjna na kanałach debleed (6 nagrań × 15 okien 4 s, miks L+R,
      PIT SI-SDRi przy 8 kHz, `real_sep_probe.py`; specyfikacja zbioru:
      `thesis-log/sweep_plan/CLARIN_ORACLE.md`) — bez wąsów CI (decyzja autora
      2026-09-04; kolumny `x_real_{lo,hi}` w tabeli drabinowej zostają),
  (c) SQUIM SI-SDR na nagraniach niesymulowanych BEZ referencji — CLARIN_fragments
      (obszary nakładania ocenianych fragmentów, nieintruzyjna).

Rodziny, kolory, znaczniki, linia „bez separacji" (najsilniejszy wariant bez
separatora, `v41_merge_nosep_relB`) i formatowanie liczb są wspólne z
`_ch6_pq_common` — ta sama tabela drabinowa `ladder_table_<split>_r3.csv`
(`ladder_table.py`) i `family_<split>_perfrag.csv`. Rysunek NIE nosi podpisów
o liczbie fragmentów ani współczynników ρ — to należy do prozy (decyzja autora
2026-08-23).

Usage: python scripts/thesis_figures/fig_ch6_ladder.py [--split test|dev] [--metric cpwer|cpcer]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ch6_pq_common import (  # noqa: E402
    AXIS, DEFAULT_OUT, FAMILIES, GRID, INK, INK2, PL, YLAB, load_ladder, nosep_value,
)

PANELS = [
    ("x_indomain_all8", "(a) SI-SDRi [dB]\nzbiór testowy PolSESS_128k"),
    ("x_real_sisdri",   "(b) SI-SDRi [dB]\nCLARIN_oracle"),
    ("x_squim_sisdr",   "(c) SQUIM SI-SDR [dB]\nCLARIN_fragments"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--metric", default="cpwer", choices=("cpwer", "cpcer"))
    ap.add_argument("--csv", type=Path, default=None,
                    help="tabela drabinowa (domyślnie ladder_table_<split>_r3.csv)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    seps = load_ladder(args.split, args.csv)
    y = args.metric
    nosep = nosep_value(args.split, y)

    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.8), sharey=True)
    ylo = min(seps[y].min(), nosep if nosep is not None else 99) - 1.0
    yhi = seps[y].max() + 1.0

    for ax, (key, xlabel) in zip(axes, PANELS):
        sub = seps[seps[key].notna()]
        if nosep is not None:
            ax.axhline(nosep, ls=(0, (5, 2)), lw=1.0, color=INK2, zorder=2)
        for grp, _lab, col, mk in FAMILIES:
            g = sub[sub["grp"] == grp]
            ax.scatter(g[key], g[y], color=col, marker=mk, s=30,
                       edgecolor="white", lw=0.5, zorder=3)
        ax.set_xlabel(xlabel, fontsize=8.6)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_formatter(PL)
        ax.yaxis.set_major_formatter(PL)
        ax.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2.5)
    if nosep is not None:
        xl = axes[0].get_xlim()
        axes[0].text(xl[0] + 0.02 * (xl[1] - xl[0]), nosep + 0.2, "bez separacji",
                     fontsize=8.4, color=INK2, va="bottom", ha="left")
    axes[0].set_ylabel(YLAB[y])

    present = set(seps["grp"])
    handles = [Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white",
                      markersize=6, label=lab)
               for k, lab, col, mk in FAMILIES if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rys_ch6_ladder_{args.split}" + ("" if y == "cpwer" else "_cpcer")
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    n = {k: int(seps[k].notna().sum()) for k, _ in PANELS}
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  ({len(seps)} separatorów; "
          f"punkty na panelach {n}; bez separacji = {nosep:.2f})")


if __name__ == "__main__":
    main()
