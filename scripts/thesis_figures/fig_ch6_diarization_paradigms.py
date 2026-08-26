"""Rysunek (ch6) — dwa paradygmaty diaryzacji: modułowy vs end-to-end.

Schemat porównawczy (bez danych): po lewej klasyczny łańcuch modułowy
(segmentacja lokalna → osadzenia → klastrowanie), którego reprezentantem w
pipelinie jest pyannote 3.1; po prawej pojedyncza sieć predykująca aktywność
wielu mówców ramka po ramce (Sortformer). Oba warianty kończą się tym samym
wynikiem — diaryzacją dopuszczającą nakładania — co jest sednem porównania.

Układ liczony jest w calach (1 jednostka danych = 1 cal, `set_aspect("equal")`),
dzięki czemu zaokrąglenia rogów pudełek są prawdziwymi łukami, a nie elipsami.

Usage: python scripts/thesis_figures/fig_ch6_diarization_paradigms.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"

# --- wspólna paleta rozdziału 6 ------------------------------------------
INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP, SILENCE = "#2a78d6", "#1baf7a", "#eb6834", "#e1e0d9"

W, H = 6.6, 3.0
CX_L, CX_R = 1.69, 4.91      # środki paneli (równe marginesy 0,38")
DIVIDER = 3.30
BOX_W = 2.62                 # szerokość szerokich pudełek (obu paneli)

Y_TOP = 2.66                 # górna krawędź pierwszego pudełka
Y_OUT_A = (0.72, 0.83)       # wiersze mini-schematu wyniku
Y_OUT_B = (0.56, 0.67)


def box(ax, cx, y0, y1, width, lines):
    """Zaokrąglone pudełko z wyśrodkowanymi wierszami tekstu."""
    ax.add_patch(FancyBboxPatch(
        (cx - width / 2, y0), width, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.055",
        facecolor="white", edgecolor=AXIS, lw=1.0, zorder=2))
    for y, text, size, color in lines:
        ax.text(cx, y, text, ha="center", va="center", fontsize=size,
                color=color, zorder=3)


def arrow(ax, x, y_from, y_to):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.1, color=INK2,
                                 shrinkA=0, shrinkB=0, zorder=2))


def strip(ax, x0, x1, y0, y1, classes, gap=0.022):
    """Pasek kolejnych ramek/okien pokolorowanych klasą powerset."""
    step = (x1 - x0) / len(classes)
    colors = {"": SILENCE, "A": SPK_A, "B": SPK_B, "AB": OVERLAP}
    for i, cls in enumerate(classes):
        ax.add_patch(Rectangle((x0 + i * step + gap / 2, y0), step - gap,
                               y1 - y0, color=colors[cls], lw=0, zorder=3))


def output_block(ax, cx):
    """Mini-schemat wyniku: dwie ścieżki mówców z widocznym nakładaniem."""
    half = 0.72
    x0, span = cx - half, 2 * half
    seg_a = [(0.00, 0.42), (0.72, 1.00)]
    seg_b = [(0.30, 0.66)]
    ovl = (0.30, 0.42)

    ax.add_patch(Rectangle((x0 + ovl[0] * span, Y_OUT_B[0] - 0.03),
                           (ovl[1] - ovl[0]) * span,
                           Y_OUT_A[1] - Y_OUT_B[0] + 0.06,
                           facecolor=OVERLAP, alpha=0.30, lw=0, zorder=1))
    for segs, (y0, y1), color in ((seg_a, Y_OUT_A, SPK_A), (seg_b, Y_OUT_B, SPK_B)):
        for s, e in segs:
            ax.add_patch(Rectangle((x0 + s * span, y0), (e - s) * span, y1 - y0,
                                   color=color, lw=0, zorder=2))
    for label, (y0, y1) in (("A", Y_OUT_A), ("B", Y_OUT_B)):
        ax.text(x0 - 0.07, (y0 + y1) / 2, label, ha="right", va="center",
                fontsize=7, color=INK2)
    ax.text(x0 + sum(ovl) / 2 * span, 0.455, "nakładanie", ha="center",
            va="center", fontsize=6.8, color=MUTED)
    ax.text(cx, 0.30, "diaryzacja (z nakładaniami)", ha="center", va="center",
            fontsize=7.6, color=INK2)


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

    # --- nagłówki paneli + separator ---
    ax.text(CX_L, 2.86, "Podejście modułowe (klastrowanie)", ha="center",
            va="center", fontsize=9, color=INK)
    ax.text(CX_R, 2.86, "Podejście end-to-end (EEND)", ha="center",
            va="center", fontsize=9, color=INK)
    ax.plot([DIVIDER, DIVIDER], [0.10, 2.96], color=AXIS, lw=0.8, zorder=0)

    # ================= panel lewy: łańcuch modułowy =================
    box(ax, CX_L, 2.44, Y_TOP, 1.50,
        [(2.55, "nagranie (miks)", 8, INK)])
    arrow(ax, CX_L, 2.44, 2.34)

    box(ax, CX_L, 1.66, 2.34, BOX_W, [
        (2.24, "segmentacja lokalna (okna)", 8, INK),
        (2.09, "klasy powerset: ∅ / A / B / A+B", 7.2, INK2),
    ])
    strip(ax, CX_L - 1.05, CX_L + 1.05, 1.77, 1.93,
          ["", "A", "A", "A", "AB", "AB", "B", "B", "", ""])
    arrow(ax, CX_L, 1.66, 1.56)

    box(ax, CX_L, 1.32, 1.56, BOX_W,
        [(1.44, "osadzenia mówców (na segment)", 8, INK)])
    arrow(ax, CX_L, 1.32, 1.22)

    box(ax, CX_L, 0.98, 1.22, BOX_W,
        [(1.10, "klastrowanie (2 mówców)", 8, INK)])
    arrow(ax, CX_L, 0.98, 0.90)

    output_block(ax, CX_L)
    ax.text(CX_L, 0.12, "pyannote 3.1", ha="center", va="center",
            fontsize=7, color=MUTED)

    # ================= panel prawy: jedna sieć =================
    box(ax, CX_R, 2.44, Y_TOP, 1.50,
        [(2.55, "nagranie (miks)", 8, INK)])
    arrow(ax, CX_R, 2.44, 2.34)

    box(ax, CX_R, 1.66, 2.34, BOX_W, [
        (2.17, "jedna sieć neuronowa", 8, INK),
        (1.97, "predykcja aktywności wielu mówców", 7.4, INK2),
        (1.82, "(ramka po ramce)", 7.4, INK2),
    ])
    ax.text(CX_R, 1.52, "Sortformer: kolejność mówców wg pierwszego",
            ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(CX_R, 1.38, "zabrania głosu (sort-loss)", ha="center", va="center",
            fontsize=7, color=MUTED)
    arrow(ax, CX_R, 1.28, 0.90)

    output_block(ax, CX_R)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_diarization_paradigms.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_diarization_paradigms.{{png,pdf}}")


if __name__ == "__main__":
    main()
