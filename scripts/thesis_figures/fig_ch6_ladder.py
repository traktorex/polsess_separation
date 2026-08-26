"""Rysunek (ch6 §6.6.2) — dawka–odpowiedź: jakość separatora a wynik ASR potoku.

Druga „money figure" rozdziału: 23–25 separatorów o różnej jakości wstawionych
do tego samego, zamrożonego potoku (`v41_merge`, pojedyncze dekodowanie). Oś Y:
cpWER (lub cpCER, `--metric`) potoku; oś X: jakość separatora mierzona na TRZECH
osiach, po jednym panelu na oś (decyzje autora 2026-08-23 — oś walidacyjna
PolSESS odrzucona jako zbędna obok osi testowej; jedna metryka naraz):
  (a) SI-SDRi na nagraniach niesymulowanych Z referencjami — CLARIN_oracle:
      sonda intruzyjna na kanałach debleed (6 nagrań × 15 okien 4 s, miks L+R,
      PIT SI-SDRi przy 8 kHz, `real_sep_probe.py`; specyfikacja zbioru:
      `thesis-log/sweep_plan/CLARIN_ORACLE.md`); poziome wąsy = 95 % CI
      bootstrap po oknach,
  (b) SQUIM SI-SDR na nagraniach niesymulowanych BEZ referencji — CLARIN_fragments
      (obszary nakładania ocenianych fragmentów, nieintruzyjna),
  (c) SI-SDRi w dziedzinie treningowej (zbiór testowy PolSESS_128k, all-8) —
      najsłabszy predyktor: separatory z C_new_64 leżą tu w paśmie 11,5–13,5 dB
      razem z dobrymi, a ich cpWER jest o 12 pkt gorsze.

Rodziny (kolor): trenowane na PolSESS_128k (bez względu na epokę, budżet danych,
architekturę i głębokość — to JEDNA rodzina, bo różni je tylko stopień
wytrenowania; separator wdrożony e46 jest jednym z tych punktów, bez
wyróżnienia — decyzja autora 2026-08-23), ablacje różnorodności MM-IPC
(bez E / tylko C), separatory trenowane na korpusie C_new_64 oraz — od
2026-08-26 (polecenie autora) — gotowe separatory zewnętrzne B1
(TF-Locoformer-M ×3 korpusy, SR-CorrNet-B ×2; §4.5.38/41). Rodzina C_new_64
(z drugim biegiem r2) występuje TYLKO na dev — nie ma dekodowań testowych
(dyscyplina dev-only tej rodziny); zewnętrzne są na obu splitach.

Linia „bez separacji" = NAJSILNIEJSZY wariant bez separatora (`v41_merge_nosep_relB`,
jak na rysunku krzywej K), mikro-uśredniony z `family_<split>_perfrag.csv`.

Rysunek NIE nosi podpisów o zbiorze, liczbie fragmentów ani współczynników ρ —
to należy do prozy (decyzja autora 2026-08-23).

Dane: `<eval>/_forensics/swap_family/ladder_table_<split>.csv` (generuje
`ladder_table.py`; DEV istnieje, TEST po kolejce 5/6 → `--split test`) +
`family_<split>_perfrag.csv` (linia bez separacji).

Usage: python scripts/thesis_figures/fig_ch6_ladder.py [--split dev|test] [--metric cpwer|cpcer]
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
SWAP = Path.home() / "datasets" / "eval" / "clarin_fragments" / "_forensics" / "swap_family"

INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP = "#2a78d6", "#1baf7a", "#eb6834"
RED = "#c0392b"

# rodzina wg `ladder_table.py` → rodzina na rysunku
GROUP = {"ref": "in128", "progress": "in128", "ladder16k": "in128", "arch": "in128",
         "swap": "diet", "cnew64": "cnew64", "external": "ext"}
GROUP_OF_ARM = {"v41_mf2full": "in128"}        # MF2-full to pojemność, nie dieta
FAMILIES = [
    ("in128",  "trenowane na PolSESS_128k",                             SPK_A, "o"),
    ("diet",   "ablacje różnorodności MM-IPC (bez E / tylko C)",      OVERLAP, "o"),
    ("cnew64", "trenowane na korpusie C_new_64",                        RED,   "^"),
    ("ext",    "gotowe separatory zewnętrzne (B1)",                     SPK_B, "s"),
]
COLOR = {k: c for k, _l, c, _m in FAMILIES}
MARKER = {k: m for k, _l, _c, m in FAMILIES}

PANELS = [
    ("x_real_sisdri",   "(a) SI-SDRi [dB]\nna nagraniach niesymulowanych\nz referencjami (CLARIN_oracle)"),
    ("x_squim_sisdr",   "(b) SQUIM SI-SDR [dB]\nna nagraniach niesymulowanych\nbez referencji (CLARIN_fragments)"),
    ("x_indomain_all8", "(c) SI-SDRi [dB]\nna zbiorze testowym\nPolSESS_128k"),
]
NOSEP_ARM = "v41_merge_nosep_relB"
YLAB = {"cpwer": "cpWER [%]", "cpcer": "cpCER [%]"}

PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def nosep_value(split, metric):
    """Mikro-średnia najsilniejszego ramienia bez separacji."""
    df = pd.read_csv(SWAP / f"family_{split}_perfrag.csv")
    g = df[df["config"] == NOSEP_ARM]
    if g.empty:
        return None
    return (100 * g["cp_err"].sum() / g["cp_len"].sum() if metric == "cpwer"
            else 100 * g["cer_err"].sum() / g["cer_len"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--metric", default="cpwer", choices=("cpwer", "cpcer"))
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    # _ext = rozszerzone tabele (2026-08-26): + 5 separatorów zewnętrznych (B1, oba splity)
    # + rodzina C_new_64 z r2 (tylko DEV — ta rodzina nie ma dekodowań testowych).
    # Zamrożone `ladder_table_{split}.csv` pozostają nietknięte.
    df = pd.read_csv(args.csv or SWAP / f"ladder_table_{args.split}_ext.csv")
    seps = df[df["family"] != "nosep"].copy()
    seps["grp"] = [GROUP_OF_ARM.get(a, GROUP[f]) for a, f in zip(seps["arm"], seps["family"])]
    y = args.metric
    nosep = nosep_value(args.split, y)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), sharey=True)
    ylo = min(seps[y].min(), nosep if nosep is not None else 99) - 1.0
    yhi = seps[y].max() + 1.0

    for ax, (key, xlabel) in zip(axes, PANELS):
        sub = seps[seps[key].notna()]
        if key == "x_real_sisdri":
            ax.errorbar(sub[key], sub[y], xerr=[sub[key] - sub["x_real_lo"],
                                               sub["x_real_hi"] - sub[key]],
                        fmt="none", ecolor=AXIS, elinewidth=0.7, capsize=0, zorder=2)
        if nosep is not None:
            ax.axhline(nosep, ls=(0, (5, 2)), lw=1.0, color=INK2, zorder=2)
        for grp, _lab, col, mk in FAMILIES:
            g = sub[sub["grp"] == grp]
            ax.scatter(g[key], g[y], color=col, marker=mk, s=30, edgecolor="white",
                       lw=0.5, zorder=3)
        ax.set_xlabel(xlabel, fontsize=7.6)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_formatter(PL)
        ax.yaxis.set_major_formatter(PL)
        ax.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2.5)
    if nosep is not None:
        axes[0].text(axes[0].get_xlim()[0] + 0.3, nosep + 0.2, "bez separacji",
                     fontsize=7.4, color=INK2, va="bottom", ha="left")
    axes[0].set_ylabel(YLAB[y])

    present = set(seps["grp"])
    handles = [Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white",
                      markersize=6, label=lab) for k, lab, col, mk in FAMILIES if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.6,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    stem = f"rys_ch6_ladder_{args.split}" + ("" if y == "cpwer" else "_cpcer")
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  ({len(seps)} separators; "
          f"bez separacji = {nosep:.2f})")


if __name__ == "__main__":
    main()
