"""Wspólna warstwa rysunkowa trójki wykresów „jakość separacji a wynik ASR" (ch6).

Trzy rysunki — po jednym na źródło pomiaru: `fig_ch6_oracle_pesq_stoi.py`
(kanały debleed, intruzyjne), `fig_ch6_squim_pesq_stoi.py` (SQUIM,
nieintruzyjne) i `fig_ch6_indomain_pesq_stoi.py` (PolSESS_128k, w domenie) —
pokazują każde PO TRZY metryki (SI-SDR(i) | PESQ | STOI) jednego źródła
(decyzja autora 2026-08-28: stratyfikacja per-źródło; wcześniejszy
`fig_ch6_ladder.py`, 3×SI-SDR w jednym rysunku, jest tym samym zastąpiony).
AKTUALIZACJA 2026-09-04: rozdział wrócił do `fig_ch6_ladder.py` (jeden rysunek,
3×SI-SDR(i), PESQ/STOI = jedno zdanie w prozie); trójka per-źródło zostaje jako
zapis układu z 2026-08-28, nie regenerować do rozdziału bez ponownej decyzji.
Rysunki różnią się WYŁĄCZNIE źródłem osi X. Oś Y, kolejność paneli, podział na
rodziny separatorów, kolory, znaczniki, linia „bez separacji" i formatowanie
liczb muszą być identyczne, bo rysunki porównuje się między sobą w pionie
(ten sam panel w trzech rysunkach = ta sama metryka na trzech źródłach).
Trzymanie tego w jednym miejscu jest jedynym sposobem, żeby nie rozjechały się
przy kolejnej edycji.

Moduł nie jest samodzielnym skryptem — nie ma `main()`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"
SWAP = Path.home() / "datasets" / "eval" / "clarin_fragments" / "_forensics" / "swap_family"

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SPK_A, SPK_B, OVERLAP, RED = "#2a78d6", "#1baf7a", "#eb6834", "#c0392b"

# identyczne z fig_ch6_ladder.py
GROUP = {"ref": "in128", "progress": "in128", "ladder16k": "in128", "arch": "in128",
         "swap": "diet", "cnew64": "cnew64", "external": "ext"}
GROUP_OF_ARM = {"v41_mf2full": "in128"}
FAMILIES = [
    ("in128",  "trenowane na PolSESS_128k",     SPK_A,   "o"),
    ("diet",   "MF2-matched: no-E / only-C",    OVERLAP, "o"),
    ("cnew64", "trenowane na PolSESS_64k",      RED,     "^"),
    ("ext",    "modele zewnętrzne",             SPK_B,   "s"),
]
NOSEP_ARM = "v41_merge_nosep_relB"
YLAB = {"cpwer": "cpWER [%]", "cpcer": "cpCER [%]"}
PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def load_ladder(split, csv=None):
    """Ramiona-separatory tabeli drabinowej z przypisaną rodziną (kolumna `grp`)."""
    df = pd.read_csv(csv or SWAP / f"ladder_table_{split}_r3.csv")
    seps = df[df["family"] != "nosep"].copy()
    seps["grp"] = [GROUP_OF_ARM.get(a, GROUP[f]) for a, f in zip(seps["arm"], seps["family"])]
    return seps


def nosep_value(split, metric):
    """Mikrouśredniony wynik najlepszego wariantu bez separacji — linia odniesienia."""
    df = pd.read_csv(SWAP / f"family_{split}_perfrag.csv")
    g = df[df["config"] == NOSEP_ARM]
    if g.empty:
        return None
    return (100 * g["cp_err"].sum() / g["cp_len"].sum() if metric == "cpwer"
            else 100 * g["cer_err"].sum() / g["cer_len"].sum())


def draw(seps, panels, metric, nosep, out_dir, stem, title=None):
    """Wielopanelowy wykres rozrzutu: oś X = `panels[i][0]`, oś Y = `metric`.

    `title` (opcjonalne) — tytuł całego rysunku (nazwa źródła pomiaru osi X).
    """
    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    width = {2: 5.9, 3: 8.4}.get(len(panels), 2.8 * len(panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.6), sharey=True)
    ylo = min(seps[metric].min(), nosep if nosep is not None else 99) - 1.0
    yhi = seps[metric].max() + 1.0

    for ax, (key, xlabel) in zip(axes, panels):
        sub = seps[seps[key].notna()]
        if nosep is not None:
            ax.axhline(nosep, ls=(0, (5, 2)), lw=1.0, color=INK2, zorder=2)
        for grp, _lab, col, mk in FAMILIES:
            g = sub[sub["grp"] == grp]
            ax.scatter(g[key], g[metric], color=col, marker=mk, s=30,
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
    axes[0].set_ylabel(YLAB[metric])

    present = set(seps["grp"])
    handles = [Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white",
                      markersize=6, label=lab)
               for k, lab, col, mk in FAMILIES if k in present]
    ncol = 3 if len(panels) >= 3 else 2
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, -0.02 if ncol == 2 else -0.01),
               columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.13 if ncol == 2 else 0.09, 1, 1))
    if title:
        # tytuł tuż nad panelami; miejsce na sam napis dokłada dopiero
        # bbox_inches="tight" przy zapisie
        fig.suptitle(title, fontsize=11, color=INK, va="bottom",
                     y=max(ax.get_position().y1 for ax in axes) + 0.03)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
