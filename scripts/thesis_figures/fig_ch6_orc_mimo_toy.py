"""Rysunek (ch6) — różnica między ORC-WER i MIMO-WER na przykładzie zabawkowym.

Obie metryki są wolne od przypisania mówcy: przydzielają wypowiedzi referencji
do strumieni hipotezy tak, aby łączna odległość edycyjna była najmniejsza.
Różnią się wyłącznie dopuszczalną kolejnością sklejania wypowiedzi w obrębie
strumienia: ORC wymaga kolejności według czasu startu (globalnej), MIMO
dopuszcza dowolny przeplot między mówcami, zachowując jedynie kolejność
wypowiedzi każdego mówcy z osobna. Kolejność ORC jest jednym z przeplotów
dopuszczanych przez MIMO, stąd zawsze MIMO-WER ≤ ORC-WER.

Przykład (jeden strumień hipotezy = transkrypcja miksu, jak wiersz
„mieszanina" w tabelach rozdziału 6):
  referencja: A1 „ala ma kota" (0,00–2,2 s), B1 „puszek śpi" (0,25–2,6 s —
  nakłada się na A1 niemal w całości, a różnica czasów startu jest w granicach
  marginesów anotacji, więc „pierwszeństwo" A1 to artefakt znaczników, nie
  fakt), A2 „kot pije mleko" (3,0–5,5 s); N = 8 słów.
  strumień:  „puszek śpi ala ma kota kot pije mleko" — wszystkie słowa
  poprawne, ale B1 przed A1 (kolejność, którą przy nakładaniu
  jednostrumieniowy ASR może wybrać równie dobrze jak odwrotną).

  ORC  — jedyna kolejność A1·B1·A2 → „ala ma kota puszek śpi kot pije mleko"
         → 4 błędy (2 wstawienia + 2 usunięcia) → 4/8 = 50 %
  MIMO — przeploty: B1·A1·A2 → 0, A1·B1·A2 → 4, A1·A2·B1 → 4 → min 0/8 = 0 %

(Odległości zweryfikowane ręcznie: dopasowanie „ala ma kota" + „kot pije
mleko" zostawia „puszek śpi" skrzyżowane po obu stronach — 2 usunięcia +
2 wstawienia; wariant z zamianami kosztuje 5, więc minimum to 4.)

Układ liczony w calach (1 jednostka danych = 1 cal, `set_aspect("equal")`),
jak w pozostałych schematach rozdziału 6.

Usage: python scripts/thesis_figures/fig_ch6_orc_mimo_toy.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch06"

# --- wspólna paleta rozdziału 6 ------------------------------------------
INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP, SILENCE = "#2a78d6", "#1baf7a", "#eb6834", "#e1e0d9"

W, H = 6.3, 4.35
MARGIN = 0.34
ROUND = 0.055
PADX, PADY = 0.115, 0.080

FT, FS = 6.9, 8.6            # nagłówek pudełka / zdanie
FP, FC, FR = 7.5, 6.8, 8.2   # tytuł panelu / wiersze opisowe / wynik

UTTS = {
    "A1": dict(spk="A", words="ala ma kota", t0=0.0, t1=2.2),
    "B1": dict(spk="B", words="puszek śpi", t0=0.25, t1=2.6),
    "A2": dict(spk="A", words="kot pije mleko", t0=3.0, t1=5.5),
}
HYP = "„puszek śpi ala ma kota kot pije mleko”"
ORC_REF = "„ala ma kota puszek śpi kot pije mleko”"
SPK_COL = {"A": SPK_A, "B": SPK_B}


def text_w(fig, s, fs, weight="normal"):
    t = fig.text(0, 0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def line_h(fs):
    return fs * 1.42 / 72


def panel_frame(ax, x0, x1, y0, y1, edge, lw):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                facecolor="white", edgecolor=edge, lw=lw,
                                zorder=1))


def chip(ax, x, y, w, h, label, color, fs):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.045",
                                facecolor="white", edgecolor=color, lw=1.0,
                                zorder=3))
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)


def chip_row(ax, cx_left, y, order, cw=0.32, ch=0.20, gap=0.07, fs=6.2):
    x = cx_left
    for lab in order:
        chip(ax, x, y, cw, ch, lab, SPK_COL[UTTS[lab]["spk"]], fs)
        x += cw + gap
    return x - gap  # prawa krawędź ostatniego chipa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # =========== oś czasu: referencja z nakładaniem ========================
    X0, X1 = 1.06, W - MARGIN
    SC = (X1 - X0) / 5.7
    yA, yB, BH = H - 0.42, H - 0.86, 0.30

    # pas nakładania (wspólny odcinek A1 i B1)
    ov0 = X0 + max(UTTS["A1"]["t0"], UTTS["B1"]["t0"]) * SC
    ov1 = X0 + min(UTTS["A1"]["t1"], UTTS["B1"]["t1"]) * SC
    ax.add_patch(Rectangle((ov0, yB - BH / 2 - 0.09), ov1 - ov0,
                           (yA + BH / 2 + 0.09) - (yB - BH / 2 - 0.09),
                           facecolor=OVERLAP, alpha=0.15, edgecolor="none",
                           zorder=0.5))
    ax.text((ov0 + ov1) / 2, yB - BH / 2 - 0.20, "nakładanie", ha="center",
            va="center", fontsize=6.2, color=INK2, zorder=4)

    for spk, y in (("A", yA), ("B", yB)):
        ax.text(X0 - 0.12, y, f"Mówca {spk}", ha="right", va="center",
                fontsize=7.2, color=SPK_COL[spk], zorder=4)

    for lab, u in UTTS.items():
        y = yA if u["spk"] == "A" else yB
        x0, x1 = X0 + u["t0"] * SC, X0 + u["t1"] * SC
        ax.add_patch(FancyBboxPatch((x0, y - BH / 2), x1 - x0, BH,
                                    boxstyle="round,pad=0,rounding_size=0.045",
                                    facecolor="white",
                                    edgecolor=SPK_COL[u["spk"]], lw=1.1,
                                    zorder=3))
        w_lab = text_w(fig, lab, 6.8, "bold")
        w_txt = text_w(fig, f"  „{u['words']}”", 6.8)
        xs = (x0 + x1) / 2 - (w_lab + w_txt) / 2
        ax.text(xs, y, lab, ha="left", va="center", fontsize=6.8, color=INK,
                fontweight="bold", zorder=4)
        ax.text(xs + w_lab, y, f"  „{u['words']}”", ha="left", va="center",
                fontsize=6.8, color=INK, zorder=4)

    # =========== strumień hipotezy =========================================
    cy = H - 1.52
    box_w = max(text_w(fig, HYP, FS),
                text_w(fig, "Strumień hipotezy — transkrypcja miksu", FT)) \
        + 2 * PADX
    box_h = line_h(FT) + line_h(FS) + 2 * PADY
    cx = W / 2
    ax.add_patch(FancyBboxPatch((cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                                boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                facecolor="white", edgecolor=AXIS, lw=1.1,
                                zorder=3))
    ax.text(cx, cy + box_h / 2 - PADY - line_h(FT) / 2,
            "Strumień hipotezy — transkrypcja miksu", ha="center",
            va="center", fontsize=FT, color=INK2, zorder=4)
    ax.text(cx, cy - box_h / 2 + PADY + line_h(FS) / 2, HYP, ha="center",
            va="center", fontsize=FS, color=INK, zorder=4)

    # =========== dolne panele: ORC vs MIMO =================================
    y_top = cy - box_h / 2 - 0.18
    y_bot = 0.12
    gap = 0.26
    pw = (W - 2 * MARGIN - gap) / 2

    # --- ORC (lewy) --------------------------------------------------------
    x0, x1 = MARGIN, MARGIN + pw
    cxl = (x0 + x1) / 2
    panel_frame(ax, x0, x1, y_bot, y_top, AXIS, 0.9)
    ax.text(cxl, y_top - 0.17, "ORC-WER", ha="center", va="center",
            fontsize=FP, color=INK, zorder=4)
    ax.text(cxl, y_top - 0.36, "kolejność: zawsze według czasu startu",
            ha="center", va="center", fontsize=6.3, color=INK2, zorder=4)

    cw, chh, cgap = 0.36, 0.22, 0.09
    row_w = 3 * cw + 2 * cgap
    chip_row(ax, cxl - row_w / 2, y_top - 0.72, ["A1", "B1", "A2"],
             cw=cw, ch=chh, gap=cgap, fs=6.4)
    ax.text(cxl, y_top - 1.06, ORC_REF, ha="center", va="center",
            fontsize=6.5, color=INK, zorder=4)
    ax.text(cxl, y_top - 1.36, "wobec strumienia: 4 błędy", ha="center",
            va="center", fontsize=FC, color=INK2, zorder=4)
    ax.text(cxl, y_top - 1.54, "(2 wstawienia + 2 usunięcia)", ha="center",
            va="center", fontsize=6.3, color=INK2, zorder=4)
    ax.text(cxl, y_bot + 0.38, "ORC-WER = 4/8 = 50%", ha="center",
            va="center", fontsize=FR, color=INK, fontweight="bold", zorder=4)
    ax.text(cxl, y_bot + 0.18, "brak wyboru — jedna dozwolona kolejność",
            ha="center", va="center", fontsize=6.2, color=INK2, zorder=4)

    # --- MIMO (prawy) ------------------------------------------------------
    x0, x1 = MARGIN + pw + gap, W - MARGIN
    cxr = (x0 + x1) / 2
    panel_frame(ax, x0, x1, y_bot, y_top, AXIS, 0.9)
    ax.text(cxr, y_top - 0.17, "MIMO-WER", ha="center", va="center",
            fontsize=FP, color=INK, zorder=4)
    ax.text(cxr, y_top - 0.36, "kolejność: dowolny przeplot między mówcami,",
            ha="center", va="center", fontsize=6.3, color=INK2, zorder=4)
    ax.text(cxr, y_top - 0.51, "A1 zawsze przed A2", ha="center", va="center",
            fontsize=6.3, color=INK2, zorder=4)

    rows = [
        (["B1", "A1", "A2"], "→ 0 błędów", True),
        (["A1", "B1", "A2"], "→ 4 błędy (= ORC)", False),
        (["A1", "A2", "B1"], "→ 4 błędy", False),
    ]
    cw, chh, cgap = 0.30, 0.19, 0.06
    row_w = 3 * cw + 2 * cgap
    x_chips = cxr - 1.18
    y_rows = [y_top - 0.80, y_top - 1.07, y_top - 1.34]
    for (order, res, win), y in zip(rows, y_rows):
        right = chip_row(ax, x_chips, y, order, cw=cw, ch=chh, gap=cgap,
                         fs=6.0)
        ax.text(right + 0.10, y, res, ha="left", va="center", fontsize=6.6,
                color=INK, fontweight="bold" if win else "normal", zorder=4)
        if win:
            res_w = text_w(fig, res, 6.6, "bold")
            hx0 = x_chips - 0.08
            hx1 = right + 0.10 + res_w + 0.08
            ax.add_patch(FancyBboxPatch((hx0, y - 0.135), hx1 - hx0, 0.27,
                                        boxstyle="round,pad=0,"
                                                 "rounding_size=0.05",
                                        facecolor="none", edgecolor=OVERLAP,
                                        lw=1.1, zorder=2))
    ax.text(cxr, y_bot + 0.58, "przeplot B1·A1·A2 odtwarza strumień",
            ha="center", va="center", fontsize=6.2, color=INK2, zorder=4)
    ax.text(cxr, y_bot + 0.38, "MIMO-WER = 0/8 = 0%", ha="center",
            va="center", fontsize=FR, color=INK, fontweight="bold", zorder=4)
    ax.text(cxr, y_bot + 0.18, "minimum po dozwolonych przeplotach",
            ha="center", va="center", fontsize=6.2, color=INK2, zorder=4)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_orc_mimo_toy.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_orc_mimo_toy.{{png,pdf}}")


if __name__ == "__main__":
    main()
