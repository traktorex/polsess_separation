"""Rysunek (ch6) — schemat etapu routingu: podział osi czasu i okna separatora.

ILUSTRACYJNY (syntetyczny, wymyślony) ~60-sekundowy przebieg rozmowy dwóch
mówców. Nie są to dane z żadnego nagrania — aktywność mówców dobrano tak, aby
pokazać wszystkie mechanizmy decyzyjne etapu `routing` + rozszerzania okna:

  1. odrzucenie zbyt krótkiego nakładania  (RoutingConfig.min_overlap_dur = 0,20 s)
  2. scalenie dwóch bliskich nakładań       (RoutingConfig.merge_gap = 0,50 s)
  3. rozszerzenie regionu do okna 4 s       (SeparationConfig.context_window_mode
     = "expand_to_chunk", training_chunk_length_s = 4,0)

Geometria okien liczona jest tą samą arytmetyką co w kodzie produkcyjnym
(`asr_pipeline/stages/separation.py::_boundary_aware_pad`), łącznie z
asymetrycznym rozdziałem kontekstu przy brzegu nagrania — dzięki temu rysunek
nie może się rozjechać z implementacją.

Usage: python scripts/thesis_figures/fig_ch6_routing_timeline.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"

# --- wspólna paleta rozdziału 6 ------------------------------------------
INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP, SILENCE = "#2a78d6", "#1baf7a", "#eb6834", "#e1e0d9"

# --- parametry pipeline'u (odzwierciedlają configs/sweep_best_e31_refineplus) ---
MIN_OVERLAP_DUR = 0.20
MERGE_GAP = 0.50
CHUNK_S = 4.0

# --- wymyślony przebieg rozmowy ------------------------------------------
TOTAL = 60.0
SEG_A = [(0.8, 8.4), (12.2, 13.1), (17.0, 22.5), (24.6, 31.8), (45.2, 51.4), (54.6, 59.4)]
SEG_B = [(9.6, 11.4), (13.0, 16.2), (26.0, 26.9), (27.2, 28.4), (36.4, 42.0), (57.6, 59.8)]

# --- geometria (jednostki y: ~2,5 jednostki = 1 cal wysokości osi) --------
Y_A, Y_B = (6.55, 7.05), (5.80, 6.30)
Y_STRIP = (4.35, 5.05)
Y_WIN = (3.15, 3.80)
Y_CORE = (3.32, 3.63)
Y_ANN1, Y_ANN2 = 2.45, 1.72
Y_LIM = (1.20, 7.30)
GAP = 0.10  # przerwa (~3 px @300 dpi) między sąsiednimi segmentami paska


# --- logika routingu (lustro asr_pipeline/stages/routing.py) --------------
def overlaps(a, b):
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return sorted(out)


def merge_close(segs, gap):
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def expand_to_chunk(s, e, total, target):
    """Kopia `_boundary_aware_pad` z separation.py."""
    extra = target - (e - s)
    if extra <= 0:
        return s, e
    room_l, room_r = s, max(0.0, total - e)
    take_l = take_r = extra / 2.0
    if take_l > room_l:
        take_r += take_l - room_l
        take_l = room_l
    if take_r > room_r:
        take_l += take_r - room_r
        take_r = room_r
    return s - min(take_l, room_l), e + min(take_r, room_r)


def route_strip(seg_a, seg_b, total):
    """Podział osi czasu na segmenty: solo A / solo B / nakładanie / cisza."""
    edges = sorted({0.0, total} | {t for iv in seg_a + seg_b for t in iv})
    raw = []
    for s, e in zip(edges[:-1], edges[1:]):
        mid = (s + e) / 2
        in_a = any(x <= mid <= y for x, y in seg_a)
        in_b = any(x <= mid <= y for x, y in seg_b)
        raw.append((s, e, "OVL" if in_a and in_b else "A" if in_a else "B" if in_b else "SIL"))
    out = [list(raw[0])]
    for s, e, k in raw[1:]:
        if k == out[-1][2]:
            out[-1][1] = e
        else:
            out.append([s, e, k])
    return [(s, e, k) for s, e, k in out]


def bar(ax, x0, x1, y0, y1, color, gap=GAP, min_w=0.16, **kw):
    """Prostokąt zwężony o `gap`, z zachowaniem minimalnej widocznej szerokości."""
    w = max((x1 - x0) - gap, min_w)
    cx = (x0 + x1) / 2
    ax.add_patch(Rectangle((cx - w / 2, y0), w, y1 - y0, color=color, lw=0, **kw))


def leader(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", lw=1.0,
                                 color=MUTED, shrinkA=0, shrinkB=0, zorder=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ovl = overlaps(SEG_A, SEG_B)
    kept = merge_close([o for o in ovl if o[1] - o[0] >= MIN_OVERLAP_DUR], MERGE_GAP)
    dropped = [o for o in ovl if o[1] - o[0] < MIN_OVERLAP_DUR]
    windows = [expand_to_chunk(s, e, TOTAL, CHUNK_S) for s, e in kept]

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    fig.subplots_adjust(left=0.185, right=0.985, top=0.985, bottom=0.29)

    # --- wiersze 1-2: aktywność mówców ---
    for segs, (y0, y1), color in ((SEG_A, Y_A, SPK_A), (SEG_B, Y_B, SPK_B)):
        for s, e in segs:
            bar(ax, s, e, y0, y1, color)

    # --- wiersz 3: pasek wyniku routingu ---
    for s, e, kind in route_strip(SEG_A, SEG_B, TOTAL):
        if kind == "SIL":
            ax.plot([s + GAP / 2, e - GAP / 2], [Y_STRIP[0]] * 2, color=AXIS, lw=0.9,
                    solid_capstyle="butt", zorder=2)
        else:
            # 0,1-sekundowe nakładanie ma na tej skali ~2 px — rysowane z
            # minimalną widoczną szerokością, żeby w ogóle było czytelne.
            bar(ax, s, e, *Y_STRIP, {"A": SPK_A, "B": SPK_B, "OVL": OVERLAP}[kind],
                min_w=0.26 if kind == "OVL" else 0.16)

    # --- wiersz 4: okna separatora ---
    for (ws, we), (cs, ce) in zip(windows, kept):
        ax.add_patch(Rectangle((ws, Y_WIN[0]), we - ws, Y_WIN[1] - Y_WIN[0],
                               facecolor=OVERLAP, alpha=0.14, lw=0, zorder=1))
        ax.add_patch(Rectangle((ws, Y_WIN[0]), we - ws, Y_WIN[1] - Y_WIN[0],
                               facecolor="none", edgecolor=OVERLAP, lw=1.1, zorder=3))
        bar(ax, cs, ce, *Y_CORE, OVERLAP, gap=0.0)
        ax.text(min((ws + we) / 2, TOTAL - 2.4), Y_WIN[1] + 0.24, "okno 4,0 s",
                ha="center", va="center", fontsize=7, color=INK2)

    for s, e in dropped:
        x = (s + e) / 2
        ax.text(x, sum(Y_CORE) / 2, "×", ha="center", va="center",
                fontsize=11, color=MUTED)
        ax.text(x + 1.1, sum(Y_CORE) / 2, "0,1 s", ha="left", va="center",
                fontsize=7, color=MUTED)

    # --- etykiety wierszy ---
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for label, (y0, y1) in (("Mówca A", Y_A), ("Mówca B", Y_B),
                            ("Wynik routingu", Y_STRIP), ("Okna separatora", Y_WIN)):
        ax.text(-0.015, (y0 + y1) / 2, label, transform=trans, ha="right",
                va="center", fontsize=8.4, color=INK2)

    # --- adnotacje mechanizmów ---
    dx = (dropped[0][0] + dropped[0][1]) / 2
    ax.text(dx, Y_ANN1, "za krótkie — pominięte", ha="center", va="center",
            fontsize=8, color=INK2)
    leader(ax, dx, Y_ANN1 + 0.22, dx, Y_CORE[0] - 0.04)

    mx = (kept[0][0] + kept[0][1]) / 2
    ax.text(mx, Y_ANN2, "bliskie nakładania — scalone", ha="center", va="center",
            fontsize=8, color=INK2)
    leader(ax, mx, Y_ANN2 + 0.22, mx, Y_WIN[0] - 0.04)
    gx = (ovl[1][1] + ovl[2][0]) / 2  # etykieta przerwy, po której nastąpiło scalenie
    ax.text(gx, Y_STRIP[1] + 0.30, "0,3 s", ha="center", va="center",
            fontsize=7, color=MUTED)
    leader(ax, gx, Y_STRIP[1] + 0.17, gx, Y_STRIP[1] + 0.03)

    ex = (windows[1][0] + windows[1][1]) / 2
    ax.text(TOTAL, Y_ANN1, "rozszerzenie okna do 4 s", ha="right", va="center",
            fontsize=8, color=INK2)
    leader(ax, ex, Y_ANN1 + 0.22, ex, Y_WIN[0] - 0.04)

    # --- oś czasu ---
    ax.set_xlim(0, TOTAL + 0.6)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks([])
    ax.set_xlabel("czas [s]")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", length=3, color=AXIS)

    handles = [(Patch(facecolor=SPK_A), Patch(facecolor=SPK_B)),
               Patch(facecolor=OVERLAP),
               Patch(facecolor=SILENCE, edgecolor=AXIS, lw=0.8)]
    leg = ax.legend(handles,
                    ["solo → poprawa jakości", "nakładanie → separacja",
                     "cisza → pomijana"],
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
                    loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
                    frameon=False, fontsize=8.2, handlelength=1.5,
                    handletextpad=0.5, columnspacing=1.8)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch6_routing_timeline.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch6_routing_timeline.{{png,pdf}}")


if __name__ == "__main__":
    main()
