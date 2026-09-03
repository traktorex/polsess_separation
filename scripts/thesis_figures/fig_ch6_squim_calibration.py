"""Rysunek (ch6 §6.3) — walidacja SQUIM wobec rzeczywistego SI-SDR (eksperyment B7).

Wersja podstawowa (rys_ch6_squim_calibration) — dwa panele:
  * A — wykres rozrzutu: rzeczywisty SI-SDR (intruzywny, PIT, punktacja w
    parytecie z potokiem) na osi X wobec SI-SDR estymowanego przez SQUIM na
    osi Y, po jednym kolorze na system. Trzy kotwice syntetyczne
    (anchor_noise / anchor_mix / anchor_ref20) rysowane szarością — stanowią
    ramę kalibracyjną, nie są przedmiotem walidacji. Sześć rzeczywistych
    separatorów w palecie rozdziału, z wyróżnionym mossformer2_e46
    (MF2-matched 128 tys. — separator używany w potoku). Linia y = x pokazuje
    obciążenie estymatora: pionowa odległość od niej JEST tym obciążeniem.
  * B — ρ Spearmana wewnątrz systemu, słupki poziome. Ten panel istnieje po
    to, żeby widać było ρ = 0,49 dla MF2-matched 128 tys. — zastrzeżenie,
    którego rozdział faktycznie potrzebuje.

Wersja rozszerzona (rys_ch6_squim_calibration_err) — te same panele plus
środkowy panel błędu estymaty per system: punkt = średnie obciążenie
(SQUIM − rzeczywisty), wąs = ±1 SD tego błędu, oś X = mediana rzeczywistego
SI-SDR systemu; wyłącznie szczeble rzeczywiste. Pokazuje wprost dwie
własności, o których mówi prozą 6.3: obciążenie rosnące z jakością systemu
oraz rozrzut błędu rosnący wraz z jakością (przyczyna niskiego ρ wewnątrz
najlepszego systemu — patrz 67_research_squim_rangerestriction.md). Uwaga:
wersja per-kosz (łączona po systemach) była myląca — uśrednianie po
szczeblach spłaszcza SD i czyni obciążenie niemonotonicznym; własności są
per-systemowe.

Uwaga metodologiczna, która musi trafić do podpisu: pierwszorzędowy wynik B7
to ρ ŁĄCZNE liczone WYŁĄCZNIE na rzeczywistych szczeblach drabiny (bez
kotwic; kotwice rozciągają zakres i zawyżałyby korelację). Wszystkie wartości
ρ liczone są tutaj z surowego CSV, nie przepisywane z raportu — skrypt
drukuje je obok wartości raportowych jako kontrolę zgodności.

Dane: thesis/thesis-log/sweep_plan/b7_squim/b7_scores.csv
      (10 368 wierszy = 9 systemów x 128 próbek testowych PolSESS
      C_final_128_v2 x warianty MM-IPC x 2 strumienie).
Źródło liczb w podpisie: B7_SQUIM_REPORT.md (ten sam katalog), 07_arc_evidence §10.

Usage: python scripts/thesis_figures/fig_ch6_squim_calibration.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"
DEFAULT_CSV = (REPO / "thesis" / "thesis-log" / "sweep_plan" / "b7_squim"
               / "b7_scores.csv")

INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"

# paleta rodzinami, zgodna z ch5 (SepFormer pomarańczowy, SPMamba niebieski,
# MossFormer2 fioletowy) — dwa szczeble SepFormera rozróżnione jasnością
ANCHORS = [
    ("anchor_noise", "szum", "#d8d7ce"),
    ("anchor_mix",   "miks", "#bdbcb1"),
    ("anchor_ref20", "sygnał wzorcowy +20 dB", "#9b9a93"),
]
REAL = [
    ("convtasnet_base", "ConvTasNet",       "#1baf7a", 12, 0.9),
    ("dprnn_base",      "DPRNN",            "#d2a03a", 12, 0.9),
    ("spmamba_64k",     "SPMamba 64 tys.",  "#2a78d6", 12, 0.9),
    ("sepformer_16k",   "SepFormer 16 tys.", "#f0a07d", 12, 0.9),
    ("sepformer_128k",  "SepFormer 128 tys.", "#eb6834", 12, 0.9),
    ("mossformer2_e46", "MF2-matched 128 tys.", "#7b52ab", 26, 1.6),
]

# wartości z B7_SQUIM_REPORT.md — tylko do kontroli zgodności w konsoli
REPORTED_RHO = {
    "anchor_noise": 0.011, "anchor_mix": 0.555, "convtasnet_base": 0.823,
    "dprnn_base": 0.783, "spmamba_64k": 0.834, "sepformer_16k": 0.831,
    "sepformer_128k": 0.656, "mossformer2_e46": 0.486, "anchor_ref20": 0.076,
}
REPORTED_POOLED, REPORTED_MEDIAN = 0.910, 0.803

PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=["system", "true_sisdr", "squim_sisdr"])
    df = df.dropna(subset=["true_sisdr", "squim_sisdr"])
    return df


def rho_table(df):
    """ρ wewnątrz systemu + ρ łączne na szczeblach rzeczywistych."""
    per = {s: spearmanr(g["true_sisdr"], g["squim_sisdr"]).statistic
           for s, g in df.groupby("system")}
    real_ids = [k for k, *_ in REAL]
    pooled = df[df["system"].isin(real_ids)]
    pooled_rho = spearmanr(pooled["true_sisdr"], pooled["squim_sisdr"]).statistic
    median_rho = float(np.median([per[s] for s in real_ids]))
    return per, pooled_rho, median_rho, len(pooled)


def style_axes(ax):
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)


def panel_scatter(axA, df):
    for sid, label, color in ANCHORS:
        g = df[df["system"] == sid]
        axA.scatter(g["true_sisdr"], g["squim_sisdr"], s=6, color=color,
                    alpha=0.35, linewidths=0, zorder=1, label=label)
    for sid, label, color, size, lw in REAL:
        g = df[df["system"] == sid]
        emph = sid == "mossformer2_e46"
        axA.scatter(g["true_sisdr"], g["squim_sisdr"], s=7 if not emph else 9,
                    color=color, alpha=0.24 if not emph else 0.38,
                    linewidths=0, zorder=2 if not emph else 3, label=label)

    lo, hi = -95, 35
    axA.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=4)
    axA.annotate("y = x", (30.5, 30.5), (-4, 7), "data",
                 textcoords="offset points", fontsize=8, color=INK2, ha="right")

    axA.set_xlim(-72, 32)
    axA.set_ylim(-22, 36)
    axA.set_xlabel("Rzeczywisty SI-SDR [dB]")
    axA.set_ylabel("SI-SDR estymowany przez SQUIM [dB]")
    style_axes(axA)

    leg = axA.legend(loc="upper left", fontsize=7.4, ncol=1,
                     handletextpad=0.35, labelspacing=0.32,
                     borderpad=0.4, markerscale=2.6, frameon=True,
                     facecolor="white", edgecolor="none", framealpha=0.9)
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def panel_bars(axB, per):
    order = sorted([r for r in REAL], key=lambda r: per[r[0]])
    ys = np.arange(len(order))
    axB.barh(ys, [per[r[0]] for r in order],
             color=[r[2] for r in order], height=0.62, zorder=2)
    for y, r in zip(ys, order):
        v = per[r[0]]
        axB.text(v + 0.02, y, f"{v:.3f}".replace(".", ","), va="center",
                 ha="left", fontsize=7.6, color=INK)
    axB.set_yticks(ys)
    axB.set_yticklabels([r[1] for r in order], fontsize=7.6)
    axB.set_xlim(0, 1.0)
    axB.set_xlabel("ρ Spearmana")
    axB.xaxis.set_major_formatter(PL)
    axB.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    style_axes(axB)
    axB.grid(axis="y", visible=False)


def panel_error(axC, df):
    """Błąd estymaty per system: punkt = średnie obciążenie (SQUIM −
    rzeczywisty), wąs = ±1 SD tego błędu, oś X = mediana rzeczywistego
    SI-SDR systemu. Pokazuje, że wraz z jakością systemu rośnie i
    obciążenie, i rozrzut błędu estymaty."""
    rows = []
    for sid, label, color, *_ in REAL:
        g = df[df["system"] == sid]
        err = g["squim_sisdr"] - g["true_sisdr"]
        rows.append((sid, label, color, float(g["true_sisdr"].median()),
                     float(err.mean()), float(err.std())))
    rows.sort(key=lambda r: r[3])

    axC.axhline(0.0, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)
    for sid, label, color, med, mean, sd in rows:
        emph = sid == "mossformer2_e46"
        axC.errorbar(med, mean, yerr=sd, fmt="o", color=color,
                     ms=5.5 if emph else 4.5, elinewidth=1.6 if emph else 1.1,
                     capsize=3.0, zorder=3 if emph else 2)
    axC.plot([r[3] for r in rows], [r[4] for r in rows], color=MUTED,
             lw=0.9, ls=(0, (2, 2)), zorder=1)

    axC.set_xlabel("Mediana rzeczywistego SI-SDR systemu [dB]")
    axC.set_ylabel("SQUIM − rzeczywisty SI-SDR [dB]\n(średnia ±1 SD)")
    style_axes(axC)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    per, pooled_rho, median_rho, n_pooled = rho_table(df)

    print("kontrola zgodności z B7_SQUIM_REPORT.md:")
    for s in sorted(per):
        print(f"  {s:<16} liczone {per[s]:.3f}  raport {REPORTED_RHO.get(s, float('nan')):.3f}")
    print(f"  łączne (rzeczywiste, n={n_pooled}) {pooled_rho:.3f}  raport {REPORTED_POOLED:.3f}")
    print(f"  mediana wewnątrz systemu          {median_rho:.3f}  raport {REPORTED_MEDIAN:.3f}")

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })

    # --- wersja podstawowa: A + B ----------------------------------------
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.45})
    panel_scatter(axA, df)
    panel_bars(axB, per)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_squim_calibration.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {args.out_dir}/rys_ch6_squim_calibration.{{png,pdf}}")

    # --- wersja rozszerzona: A + C (błąd estymaty) + B --------------------
    fig, (axA, axC, axB) = plt.subplots(
        1, 3, figsize=(12.4, 4.3),
        gridspec_kw={"width_ratios": [2.05, 1.05, 1.0], "wspace": 0.34})
    panel_scatter(axA, df)
    rows = panel_error(axC, df)
    panel_bars(axB, per)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_squim_calibration_err.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {args.out_dir}/rys_ch6_squim_calibration_err.{{png,pdf}}")
    print("panel C (per system, szczeble rzeczywiste):")
    for sid, label, color, med, mean, sd in rows:
        print(f"  {label:<22} mediana {med:+6.2f}  obciążenie {mean:+5.2f}  SD {sd:5.2f}")


if __name__ == "__main__":
    main()
