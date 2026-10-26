"""Rysunek (ch6) — cpWER na przykładzie zabawkowym.

Schemat (bez danych) pokazuje, dlaczego metryka cpWER jest odporna na to, że
separator zwraca strumienie bez tożsamości: obie referencje zestawia się z
obydwoma strumieniami w obu możliwych przypisaniach, dla każdego przypisania
liczy się sumę błędów słownych i przyjmuje minimum.

Referencje:  A = „ala ma kota" (3 słowa), B = „burek to ładny pies" (4 słowa),
łącznie N = 7 słów. Transkrypcje automatyczne: 1 = „ala to kota",
2 = „burek ma ładny pies".
  * przypisanie I   A↔1, B↔2 → 1 + 1 = 2 błędy  → 2/7 ≈ 28,6 %
  * przypisanie II  A↔2, B↔1 → 3 + 3 = 6 błędów → 6/7 ≈ 85,7 %
cpWER = min(28,6 %; 85,7 %) = 28,6 %.

Brak tożsamości strumieni sygnalizuje glif ze znakiem „?": każda z czterech
wstęg referencja↔transkrypcja wnika pionowo w węzeł w osobnym punkcie i tam
się urywa, więc nie widać, która linia z lewej jest kontynuacją której
z prawej — przypisanie dopiero trzeba wykonać (robią to panele poniżej).

Układ liczony jest w calach (1 jednostka danych = 1 cal, `set_aspect("equal")`),
tak jak w pozostałych schematach rozdziału 6.

Usage: python scripts/thesis_figures/fig_ch6_cpwer_toy.py [--out-dir DIR]
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch06"

# --- wspólna paleta rozdziału 6 ------------------------------------------
INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP, SILENCE = "#2a78d6", "#1baf7a", "#eb6834", "#e1e0d9"
REF_A_C, REF_B_C = SPK_A, OVERLAP          # kolory referencji (A/B)
STR_1_C, STR_2_C = SPK_B, "#8e5bc6"        # kolory strumieni (1/2)

W, H = 5.6, 4.60
MARGIN = 0.34        # marginesy boczne; szerokość rysunku dobrana tak, by
MID_GAP = 1.34       # bbox_inches="tight" nie zapisywał pustych pasów po bokach
PADX, PADY = 0.115, 0.080
ROUND = 0.055

FT, FS = 6.9, 8.6    # nagłówek pudełka / zdanie w pudełku
FP = 7.5             # tytuł panelu
FW = 7.4             # wiersze tabeli słów, wiersz Σ, wiersz min(...)
FE = 6.8             # "n błędów"
FR = 8.6             # wynik końcowy

RH = 0.165           # wysokość wiersza tabeli słów
LABEL_GAP = 0.10     # odstęp etykieta "(A)" → pierwsza kolumna słów
COL_GAP = 0.13       # odstęp między kolumnami słów

REF_A, REF_B = "„ala ma kota”", "„burek to ładny pies”"
HYP_1, HYP_2 = "„ala to kota”", "„burek ma ładny pies”"
TIT_A, TIT_B = "Referencja A", "Referencja B"
TIT_1 = "Automatyczna transkrypcja 1"
TIT_2 = "Automatyczna transkrypcja 2"

# --- glif „?": cztery możliwe połączenia referencja↔transkrypcja ---------
Q_A, Q_B = 0.115, 0.19   # półosie elipsy z pytajnikiem
Q_LW = 1.15              # grubość połączeń i obrysu elipsy
Q_EPS = 20               # punkt styku: tyle stopni nad/pod „równikiem" elipsy
Q_CH = 0.30              # wybieg poziomy przy pudełku (ułamek odległości)
Q_CD = 0.05              # opad wybiegu przy pudełku (start lekko w dół)
Q_TX, Q_TY = 0.03, 0.20  # wybieg przy styku: stromo z góry, lekko od zewnątrz

GAPTOK = "—"         # znacznik pustej pozycji (wstawienie/usunięcie)
MARK_OK = "–"        # znacznik pozycji bez błędu

PANELS = [
    dict(title="Przypisanie I",
         blocks=[
             dict(lab_ref="(A)", lab_hyp="(1)", lab_col=REF_A_C,
                  hyp_col=STR_1_C,
                  ref=["ala", "ma", "kota"],
                  hyp=["ala", "to", "kota"],
                  mark=[MARK_OK, "S", MARK_OK], err="1 błąd"),
             dict(lab_ref="(B)", lab_hyp="(2)", lab_col=REF_B_C,
                  hyp_col=STR_2_C,
                  ref=["burek", "to", "ładny", "pies"],
                  hyp=["burek", "ma", "ładny", "pies"],
                  mark=[MARK_OK, "S", MARK_OK, MARK_OK], err="1 błąd"),
         ],
         total="Σ = 2   →   2/7 ≈ 28,6 %"),
    dict(title="Przypisanie II",
         blocks=[
             dict(lab_ref="(A)", lab_hyp="(2)", lab_col=REF_A_C,
                  hyp_col=STR_2_C,
                  ref=["ala", "ma", "kota", GAPTOK],
                  hyp=["burek", "ma", "ładny", "pies"],
                  mark=["S", MARK_OK, "S", "I"], err="3 błędy"),
             dict(lab_ref="(B)", lab_hyp="(1)", lab_col=REF_B_C,
                  hyp_col=STR_1_C,
                  ref=["burek", "to", "ładny", "pies"],
                  hyp=["ala", "to", "kota", GAPTOK],
                  mark=["S", MARK_OK, "S", "D"], err="3 błędy"),
         ],
         total="Σ = 6   →   6/7 ≈ 85,7 %"),
]

MIN_LINE = "min(28,6 %; 85,7 %)"
RESULT = "cpWER = 28,6 %"


def text_w(fig, s, fs, weight="normal", style="normal"):
    t = fig.text(0, 0, s, fontsize=fs, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def line_h(fs):
    return fs * 1.42 / 72


def tok_style(row, tok, mark):
    """(fontweight, kolor, fontstyle) dla tokenu tabeli; na pozycjach błędnych
    pogrubione są tokeny referencji i hipotezy, a litera S/I/D — pogrubiona
    kursywą (jak w zapisie symboli w tekście); znacznik pustej pozycji —
    wyszarzony."""
    if row in ("ref", "hyp"):
        weight = "bold" if mark != MARK_OK else "normal"
        return weight, (MUTED if tok == GAPTOK else INK), "normal"
    # row == "mark"
    if tok != MARK_OK:
        return "bold", INK, "italic"
    return "normal", INK2, "normal"


def panel_col_widths(fig, blocks, fs):
    """Szerokości kolumn wspólne dla obu bloków panelu (ref+hyp+znaczniki)."""
    ncols = max(len(b["ref"]) for b in blocks)
    widths = [0.0] * ncols
    for b in blocks:
        for row in ("ref", "hyp", "mark"):
            for i, tok in enumerate(b[row]):
                wt, _, st = tok_style(row, tok, b["mark"][i])
                widths[i] = max(widths[i], text_w(fig, tok, fs, wt, st))
    return widths


def panel_frame(ax, x0, x1, y0, y1, edge, lw):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                facecolor="white", edgecolor=edge, lw=lw,
                                zorder=1))


def draw_block(ax, x, y_ref, b, widths, label_w, fs):
    """Jeden blok ref/hyp/znaczniki, kolumny słów wyrównane do lewej."""
    rows = [("ref", b["lab_ref"], b["lab_col"], y_ref),
            ("hyp", b["lab_hyp"], b["hyp_col"], y_ref - RH),
            ("mark", None, None, y_ref - 2 * RH)]
    for row, lab, lab_col, y in rows:
        if lab is not None:
            ax.text(x, y, lab, ha="left", va="center", fontsize=fs,
                    color=lab_col, zorder=4)
        tx = x + label_w + LABEL_GAP
        toks = b[row]
        for i, wcol in enumerate(widths):
            if i < len(toks):
                wt, col, st = tok_style(row, toks[i], b["mark"][i])
                ax.text(tx, y, toks[i], ha="left", va="center", fontsize=fs,
                        color=col, fontweight=wt, fontstyle=st, zorder=4)
            tx += wcol + COL_GAP


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

    # =========== górny blok: referencje (lewa) vs strumienie (prawa) =======
    x_left, x_right = MARGIN, W - MARGIN
    box_w = max(max(text_w(fig, s, FS) for s in (REF_A, REF_B, HYP_1, HYP_2))
                + 2 * PADX,
                max(text_w(fig, s, FT) for s in (TIT_A, TIT_B, TIT_1, TIT_2))
                + 2 * PADX,
                (x_right - x_left - MID_GAP) / 2)
    box_h = line_h(FT) + line_h(FS) + 2 * PADY

    cx_l, cx_r = x_left + box_w / 2, x_right - box_w / 2
    cy_hi, cy_lo = H - 0.42, H - 1.00

    def top_box(cx, cy, title, edge, sentence):
        ax.add_patch(FancyBboxPatch((cx - box_w / 2, cy - box_h / 2), box_w,
                                    box_h,
                                    boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                    facecolor="white", edgecolor=edge, lw=1.1,
                                    zorder=3))
        ax.text(cx, cy + box_h / 2 - PADY - line_h(FT) / 2, title, ha="center",
                va="center", fontsize=FT, color=INK2, zorder=4)
        ax.text(cx, cy - box_h / 2 + PADY + line_h(FS) / 2, sentence,
                ha="center", va="center", fontsize=FS, color=INK, zorder=4)

    top_box(cx_l, cy_hi, TIT_A, REF_A_C, REF_A)
    top_box(cx_l, cy_lo, TIT_B, REF_B_C, REF_B)
    top_box(cx_r, cy_hi, TIT_1, STR_1_C, HYP_1)
    top_box(cx_r, cy_lo, TIT_2, STR_2_C, HYP_2)

    # przypisanie ref↔transkrypcja jeszcze nie wykonane: cztery wstęgi
    # wtapiające się w elipsę z pytajnikiem
    xe_l, xe_r = cx_l + box_w / 2 + 0.12, cx_r - box_w / 2 - 0.12
    qx, qy = (xe_l + xe_r) / 2, (cy_hi + cy_lo) / 2

    def link(x_out, y_out, sx, sy):
        """Wstęga od pudełka do elipsy (sx = -1/+1 — pudełko po lewej/prawej,
        sy = +1/-1 — górne/dolne). Równomierne „S": wychodzi z pudełka lekko
        w dół, opada coraz stromiej i biegnie niemal pionowo tuż obok
        bocznego bieguna elipsy, po czym dotyka obrysu tuż nad/pod
        „równikiem" i tam się urywa. Wstęgi z danej strony są lustrzane
        względem osi poziomej i kończą się blisko siebie, więc nie da się
        odczytać, która linia z lewej jest kontynuacją której z prawej —
        a o to właśnie chodzi: przypisania jeszcze nie ma."""
        eps = math.radians(Q_EPS)
        x_end = qx + sx * Q_A * math.cos(eps)
        y_end = qy + sy * Q_B * math.sin(eps)
        verts = [(x_out, y_out),
                 (x_out - sx * Q_CH * abs(x_out - x_end), y_out - sy * Q_CD),
                 (x_end + sx * Q_TX, y_end + sy * Q_TY),
                 (x_end, y_end)]
        ax.add_patch(PathPatch(
            MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                            MplPath.CURVE4]),
            facecolor="none", edgecolor=INK, lw=Q_LW,
            capstyle="round", zorder=2))

    for x_out, sx in ((xe_l, -1), (xe_r, +1)):
        for y_out, sy in ((cy_hi, +1), (cy_lo, -1)):
            link(x_out, y_out, sx, sy)

    # elipsa zamyka oba punkty zbiegu
    ax.add_patch(Ellipse((qx, qy), 2 * Q_A, 2 * Q_B, facecolor="white",
                         edgecolor=INK, lw=Q_LW, zorder=4))
    ax.text(qx, qy, "?", ha="center", va="center", fontsize=9.5, color=INK,
            zorder=5)

    # =========== dolne panele: dwa przypisania =============================
    y_ptop = cy_lo - box_h / 2 - 0.24
    y_pbot = y_ptop - 2.10
    gap = 0.26
    pw = (x_right - x_left - gap) / 2
    label_w = max(text_w(fig, lab, FW)
                  for p in PANELS for b in p["blocks"]
                  for lab in (b["lab_ref"], b["lab_hyp"]))

    for k, p in enumerate(PANELS):
        x0 = x_left + k * (pw + gap)
        x1 = x0 + pw
        cx = (x0 + x1) / 2
        panel_frame(ax, x0, x1, y_pbot, y_ptop, AXIS, 0.9)
        ax.text(cx, y_ptop - 0.19, p["title"], ha="center", va="center",
                fontsize=FP, color=INK, zorder=4)

        widths = panel_col_widths(fig, p["blocks"], FW)
        table_w = (label_w + LABEL_GAP + sum(widths)
                   + COL_GAP * (len(widths) - 1))
        xt = cx - table_w / 2
        yA = y_ptop - 0.46
        for j, b in enumerate(p["blocks"]):
            y_ref = yA - j * 0.72
            draw_block(ax, xt, y_ref, b, widths, label_w, FW)
            ax.text(xt + table_w, y_ref - 2 * RH - 0.155, b["err"],
                    ha="right", va="center", fontsize=FE, color=INK2,
                    zorder=4)

        ysep = yA - 1.35
        ax.plot([x0 + 0.10, x1 - 0.10], [ysep] * 2, color=GRID, lw=0.9,
                zorder=2)
        ax.text(cx, yA - 1.50, p["total"], ha="center", va="center",
                fontsize=FW, color=INK, zorder=4)

    # =========== dół: minimum po przypisaniach =============================
    y_join = y_pbot - 0.14
    cxs = [x_left + pw / 2, x_left + pw + gap + pw / 2]
    cxm = W / 2
    for cxp in cxs:
        ax.plot([cxp, cxp], [y_pbot - 0.02, y_join], color=AXIS, lw=1.0,
                zorder=2)
    ax.plot(cxs, [y_join, y_join], color=AXIS, lw=1.0, zorder=2)
    ax.plot([cxm, cxm], [y_join, y_join - 0.10], color=AXIS, lw=1.0, zorder=2)

    y_min = y_join - 0.20
    ax.text(cxm, y_min, MIN_LINE, ha="center", va="center", fontsize=FW,
            color=INK, zorder=4)

    rb_w = text_w(fig, RESULT, FR, "bold") + 0.50
    rb_h = line_h(FR) + 0.20
    rb_cy = y_min - 0.12 - rb_h / 2
    ax.add_patch(FancyBboxPatch((cxm - rb_w / 2, rb_cy - rb_h / 2), rb_w, rb_h,
                                boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                facecolor="white", edgecolor=MUTED, lw=1.2,
                                zorder=3))
    ax.text(cxm, rb_cy, RESULT, ha="center", va="center", fontsize=FR,
            color=INK, fontweight="bold", zorder=4)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_cpwer_toy.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_cpwer_toy.{{png,pdf}}")


if __name__ == "__main__":
    main()
