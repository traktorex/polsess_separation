"""Rysunek (ch6 §6.4) — mapa korpusu ewaluacyjnego: dwie osie zróżnicowania.

Jeden punkt = jeden fragment zamrożonego korpusu (141 = 23 strojeniowe + 118
testowych). Oś X: złożoność akustyczna (kompozyt DNSMOS-SIG + Brouhaha-SNR,
kalibrowany względem 16 ocen odsłuchowych autora). Oś Y: udział nakładania się
wypowiedzi w czasie mowy fragmentu (nakładanie / unia mowy), wyznaczony z
SUROWEJ diaryzacji Sortformer wdrożonego potoku — ta sama definicja i to samo
źródło co wiersz „Udział nakładania" Tabeli 6.4 (ustalenie autora 2026-08-25:
statystyki nakładania z diaryzacji, nie z referencji; wcześniejsza oś z doboru
pyannote wycofana). Pełna dokumentacja źródła i cross-checki znajdują się
w notatkach autora do pracy.

Rysunek ma unieść jedno zdanie rozdziału, którego proza inaczej tylko dowodzi
słownie: **warstwy LOW / MID / HIGH to warstwy ZŁOŻONOŚCI AKUSTYCZNEJ, nie
warstwy nakładania.** Obie osie są statystycznie niezależne (ρ Spearmana
−0,13 na 141 fragmentach, p = 0,11; −0,09 na samym zbiorze testowym),
a mediana udziału nakładania jest w trzech warstwach niemal identyczna
(0,12 / 0,12 / 0,10). Drugim, ubocznym odczytem jest to, że zbiór strojeniowy
jest ŁATWIEJSZY od testowego (mediana kompozytu −0,87 wobec 0,24) — dlatego
wyniki strojenia nie przenoszą się na zbiór testowy automatycznie.

Granice warstw wyznaczono dokładnie tak, jak robi to kampania: kompozyt
uśredniony do poziomu NAGRANIA (nie fragmentu), tercyle przez
`asr_pipeline.eval.stats.assign_strata` — stąd 35 / 35 / 37 nagrań i
35 / 36 / 47 fragmentów, zgodnie z Tabelą 6.4. Warstwy zdefiniowane są wyłącznie
na zbiorze testowym; fragmenty strojeniowe rysowane są osobnym znacznikiem i nie
biorą udziału w wyznaczaniu granic.

Źródło osi Y: <frag>/sweep/v41_merge_nosep/diarization.json — surowe tury
Sortformera, z których kampanijny routing.json odtwarza się dokładnie na
141/141 fragmentów. NIE używać diarization.json z v41_merge (artefakt
po-relabel, niezgodny z własnym routingiem na 40/141 fragmentów).

Dane: asr_pipeline/eval/clarin_split.csv (zamrożona przynależność, 141 wierszy)
      <eval>/clarin_fragments/composite_scores.csv (kompozyt złożoności)
      <eval>/clarin_fragments/<frag>/sweep/v41_merge_nosep/diarization.json

Usage: python scripts/thesis_figures/fig_ch6_corpus_map.py [--out-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from asr_pipeline.eval.stats import assign_strata  # noqa: E402

DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"
EVAL = Path.home() / "datasets" / "eval" / "clarin_fragments"
SPLIT_CSV = REPO / "asr_pipeline" / "eval" / "clarin_split.csv"

INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
# rampa jednobarwna: warstwy są UPORZĄDKOWANE, więc kodujemy je jasnością
STRATA = [("LOW", "#9ec9e2"), ("MID", "#4a90c4"), ("HIGH", "#1f5c8b")]

PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))


def _merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _intersect_len(a, b):
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def _diar_overlap_ratio(frag_id: str) -> float:
    """Nakładanie / unia mowy z surowych tur Sortformera (v41_merge_nosep)."""
    turns = json.loads(
        (EVAL / frag_id / "sweep" / "v41_merge_nosep" / "diarization.json").read_text()
    )["turns"]
    per = {}
    for t in turns:
        per.setdefault(t["speaker"], []).append((t["start"], t["end"]))
    spk = [_merge_intervals(v) for v in per.values()]
    assert len(spk) == 2, f"{frag_id}: oczekiwano 2 mówców, jest {len(spk)}"
    union = sum(e - s for s, e in _merge_intervals(spk[0] + spk[1]))
    return _intersect_len(*spk) / union if union else 0.0


def load():
    sp = pd.read_csv(SPLIT_CSV)
    comp = pd.read_csv(EVAL / "composite_scores.csv", usecols=["frag_id", "composite"])
    d = sp[["frag_id", "role"]].merge(comp, on="frag_id")
    assert len(d) == 141, f"oczekiwano 141 zamrożonych fragmentów, jest {len(d)}"
    d["overlap"] = d["frag_id"].map(_diar_overlap_ratio)
    d["rec"] = d["frag_id"].str.split("__").str[0]
    return d


def strata_boundaries(test):
    """Tercyle liczone na POZIOMIE NAGRANIA — tak jak w całej kampanii."""
    rec_mean = test.groupby("rec")["composite"].mean()
    strat = assign_strata(rec_mean.to_dict())
    order = sorted(rec_mean.items(), key=lambda kv: kv[1])
    t = len(order) // 3
    b1 = (order[t - 1][1] + order[t][1]) / 2
    b2 = (order[2 * t - 1][1] + order[2 * t][1]) / 2
    return strat, b1, b2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = load()
    dev, test = d[d.role == "dev"], d[d.role == "test"].copy()
    strat, b1, b2 = strata_boundaries(test)
    test["stratum"] = test["rec"].map(strat)

    rho_all = spearmanr(d.composite, d.overlap).statistic
    rho_test = spearmanr(test.composite, test.overlap).statistic
    med_ov = test.groupby("stratum")["overlap"].median()
    print(f"fragmenty: dev {len(dev)} / test {len(test)}")
    print(f"nagrania testowe w warstwach: "
          f"{pd.Series(strat).value_counts().to_dict()}")
    print(f"fragmenty testowe w warstwach: {test['stratum'].value_counts().to_dict()}")
    print(f"granice tercyli: LOW|MID {b1:.3f}   MID|HIGH {b2:.3f}")
    print(f"rho(kompozyt, nakładanie): 141 frag. {rho_all:.3f} | test {rho_test:.3f}")
    print(f"mediana nakładania w warstwach: "
          f"{ {k: round(v, 3) for k, v in med_ov.items()} }")
    print(f"mediana kompozytu: dev {dev.composite.median():.3f} / "
          f"test {test.composite.median():.3f}")

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    for x in (b1, b2):
        ax.axvline(x, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ylab_y = -0.026  # etykiety warstw pod osią 0 (fragmenty bez nakładania leżą na 0)
    for label, lo, hi in (("LOW", d.composite.min(), b1), ("MID", b1, b2),
                          ("HIGH", b2, d.composite.max())):
        ax.text((lo + hi) / 2, ylab_y, label, ha="center", va="center",
                fontsize=8.2, color=MUTED, style="italic")

    ax.scatter(dev.composite, dev.overlap, s=46, facecolor="white",
               edgecolor=MUTED, linewidth=1.3, marker="D", zorder=3)
    for name, color in STRATA:
        g = test[test.stratum == name]
        ax.scatter(g.composite, g.overlap, s=30, color=color, alpha=0.9,
                   linewidths=0, zorder=2)

    ax.set_xlabel("Złożoność akustyczna")
    ax.set_ylabel("Udział nakładania się wypowiedzi")
    ax.set_ylim(-0.045, 0.435)
    ax.xaxis.set_major_formatter(PL)
    ax.yaxis.set_major_formatter(PL)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)

    handles = [Line2D([], [], lw=0, marker="D", markersize=6.2,
                      markerfacecolor="white", markeredgecolor=MUTED,
                      markeredgewidth=1.3, label="zbiór DEV (23)")]
    handles += [Line2D([], [], lw=0, marker="o", markersize=6.2, color=c,
                       label=f"zbiór TEST, warstwa {n}"
                             f" ({int((test.stratum == n).sum())})")
                for n, c in STRATA]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.155),
              ncol=4, frameon=False, fontsize=8.0, columnspacing=1.5,
              handletextpad=0.45)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_corpus_map.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_corpus_map.{{png,pdf}}")


if __name__ == "__main__":
    main()
