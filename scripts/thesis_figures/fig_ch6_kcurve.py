"""Rysunek (ch6 §6.6.4) — krzywa K: ile daje kolejne dekodowanie w głosowaniu.

Dwa panele (cpWER po lewej, cpCER po prawej), wspólna oś X i wspólna legenda,
ale NIEZALEŻNE osie Y — pokazywane są wartości BEZWZGLĘDNE metryki, nie zysk
względem K = 1 (decyzja autora 2026-08-23; zyski podaje Tabela 6.12).
Cztery systemy, po jednym punkcie na zmierzone K ∈ {1, 3, 5, 7, 9, 11}.

Co ten rysunek ma pokazać:
  * poprawa jest MONOTONICZNA i WYHAMOWUJĄCA — ok. 56 % zysku już przy K = 3,
    75–82 % przy K = 5; jedenaste dekodowanie dokłada ułamek punktu,
  * moduł działa na KAŻDYM systemie, nie tylko na potoku — dlatego porównania
    „z separacją / bez separacji" pozostają ważne także po zastosowaniu
    głosowania (symetria protokołu),
  * słupki rozrzutu (rysowane tylko dla K = 1 i K = 3) to odchylenie standardowe
    MIĘDZY podzbiorami o danym K; kurczy się z K, a przy K = 11 znika, bo
    istnieje tylko jeden taki podzbiór.

Punkt K = 1 to ŚREDNIA z jedenastu pojedynczych dekodowań (`mean_*`), a nie
dekodowanie z przesunięciem 0 s — te dwie kotwice różnią się o 0,26–0,64 pkt
(efekt „doklejonej ciszy", rozłożony osobno w przypisie do Tabeli 6.11).

UWAGA METRYCZNA: wiersz mieszaniny to MIMO-WER / MIMO-CER, metryki wolne od
przypisania mówcy — ich POZIOM nie jest porównywalny z cpWER pozostałych
systemów, porównywalne są wyłącznie RÓŻNICE wewnątrz systemu. Dlatego krzywa
mieszaniny jest kreskowana i nazwana metryką w legendzie; na osi bezwzględnej
to zastrzeżenie jest ważniejsze niż było na osi zysku.

Punkty dla K = 5 i K = 7 policzono na 200 losowanych podzbiorach (pozostałe K
wyczerpują wszystkie kombinacje). Na rysunku nie są w żaden sposób wyróżnione
(decyzja autora 2026-08-23) — PODPIS MUSI więc powiedzieć, że drobne załamanie
krzywej nieprzetworzonego nagrania przy K = 7 jest wahaniem losowania, a nie
niemonotonicznością.

Dane: <eval>/_forensics/swap_family/ensemble_anatomy/kcurve_test.csv
      (ZBIÓR TESTOWY, 118 fragmentów; raport: ENSEMBLE_ANATOMY_TEST.md).

Usage: python scripts/thesis_figures/fig_ch6_kcurve.py [--out-dir DIR] [--csv PATH]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"
DEFAULT_CSV = (Path.home() / "datasets" / "eval" / "clarin_fragments" / "_forensics"
               / "swap_family" / "ensemble_anatomy" / "kcurve_test.csv")

INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"

# Tylko trzy systemy (decyzja autora 2026-08-23): potok, NAJSILNIEJSZY wariant
# bez separacji (`nosep_relB` — na rysunku po prostu „bez separacji"; słabszy
# `v41_merge_nosep` usunięty) oraz transkrypcja nieprzetworzonego nagrania.
SYSTEMS = [
    ("pipeline_v41_merge",         "potok",                            "#eb6834"),
    ("nosep_v41_merge_nosep_relB", "bez separacji",                    "#2a78d6"),
    ("mixture_enh_oa050",
     "transkrypcja nieprzetworzonego nagrania (MIMO-WER / MIMO-CER)",  "#898781"),
]

# Słupki rozrzutu rysowane tylko dla K = 1 i K = 3: przy większych K odchylenie
# jest już mniejsze niż grubość znacznika, a komplet słupków zaśmiecał rysunek.
ERRORBAR_K = (1, 3)

PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def panel(ax, df, metric, title):
    val, sd = f"mean_{metric}", f"sd_{metric}"
    for sid, label, color in SYSTEMS:
        g = df[df["system"] == sid].sort_values("K")
        ls = (0, (5, 2)) if sid == "mixture_enh_oa050" else "-"
        ax.plot(g["K"], g[val], color=color, lw=1.7, ls=ls, zorder=2, label=label)
        e = g[g["K"].isin(ERRORBAR_K)]
        ax.errorbar(e["K"], e[val], yerr=e[sd], fmt="none", ecolor=color,
                    elinewidth=0.9, capsize=2.4, capthick=0.9, alpha=0.8, zorder=3)
        ax.scatter(g["K"], g[val], s=25, color=color, zorder=4)
    ax.set_ylabel(title)

    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xlim(0.4, 11.6)
    ax.set_xlabel("Liczba głosów $K$")
    ax.yaxis.set_major_formatter(PL)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"wczytano {len(df)} wierszy, systemy: {sorted(df['system'].unique())}")
    for sid, label, _ in SYSTEMS:
        g = df[df["system"] == sid].set_index("K")
        print(f"  {label:<50} K=1 {g.loc[1,'mean_cpWER']:.2f}/{g.loc[1,'mean_cpCER']:.2f}"
              f"  ->  K=11 {g.loc[11,'mean_cpWER']:.2f}/{g.loc[11,'mean_cpCER']:.2f}")

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    panel(axes[0], df, "cpWER", "cpWER / MIMO-WER")
    panel(axes[1], df, "cpCER", "cpCER / MIMO-CER")

    handles = [Line2D([], [], color=c, lw=1.7, marker="o", markersize=5.2,
                      markerfacecolor=c, markeredgecolor=c, label=lab)
               for _, lab, c in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.0, bbox_to_anchor=(0.5, -0.10), columnspacing=2.0,
               handletextpad=0.5)

    fig.subplots_adjust(wspace=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_kcurve.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_kcurve.{{png,pdf}}")


if __name__ == "__main__":
    main()
