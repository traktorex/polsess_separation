"""Rysunek (ch6) — schemat potoku przetwarzania: od nagrania mono do
transkrypcji z etykietami mówców.

Dwa renderingi tej samej topologii:
  * rys_ch6_pipeline_small — szkielet (jeden ciąg pudełek + rozgałęzienie na
    ścieżkę całego nagrania i ścieżkę nakładań); ma pozostać czytelny przy
    szerokości druku ~13,5 cm,
  * rys_ch6_pipeline_full  — wersja pełna: potok tak, jak działa w
    konfiguracji `v41_merge` (`asr_pipeline/configs/sweep_best_e31_refineplus.yaml`),
    z nazwami użytych modeli, kluczowymi parametrami i mechanizmami, które
    faktycznie się wykonują.

Topologia wersji pełnej — kluczowe fakty, które schemat ma oddawać wiernie:
  * potok ma OSIEM etapów; drugi przebieg etykietowania (B+, `stages/relabel.py`)
    jest osobnym etapem między przetwarzaniem po separacji a składaniem i to
    W NIM zbiegają się obie ścieżki (nie w składaniu),
  * nie istnieje osobna „ścieżka solo": istnieje ścieżka NAKŁADAŃ (separacja →
    AP-BWE → maska VAD, fragmentami) oraz ścieżka CAŁEGO NAGRANIA (poprawa
    jakości, jednym przebiegiem); składanie korzysta z wyjść obu,
  * poprawa jakości pracuje na surowej mieszaninie i nie jest karmiona przez
    routing; separator również czyta sygnał SUROWY, nie poprawiony,
  * maska VAD jest liczona wewnątrz etapu separacji (potrzebuje jej dobór szwu)
    i nakładana PO rozszerzeniu pasma, nie przed nim,
  * poza stagem separacji nic nie przekracza granicy etapu przy 8 kHz —
    zamiana 16 → 8 → 16 kHz dzieje się w środku separatora, a AP-BWE odtwarza
    PASMO, nie częstotliwość próbkowania,
  * transkrypcja to łańcuch: dekodowanie → wykryj-i-powtórz (okna zapadnięte,
    pętle tokenowe, pętle frazowe) → wyrównanie czasowe; wszystkie trzy
    powtórki działają na segmentach sprzed wyrównania,
  * głosowania K = 11 tu NIE MA — to moduł nakładany na wyjścia transkrypcji,
    nieobecny w konfiguracji potoku (p. 6.6.4).

Układ liczony jest w calach (1 jednostka danych = 1 cal, `set_aspect("equal")`),
tak jak w pozostałych schematach rozdziału 6 — dzięki temu zaokrąglenia rogów
pudełek są prawdziwymi łukami, a nie elipsami.

Usage: python scripts/thesis_figures/fig_ch6_pipeline.py [--out-dir DIR]
"""

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch06"

# --- wspólna paleta rozdziału 6 ------------------------------------------
INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP, SILENCE = "#2a78d6", "#1baf7a", "#eb6834", "#e1e0d9"

PADX, PADY = 0.105, 0.075     # wewnętrzne marginesy pudełek [cal]
ROUND = 0.055                 # promień zaokrąglenia rogów [cal]
TINT = to_rgba(OVERLAP, 0.10)  # bardzo jasne wypełnienie pudełka separacji


# --- podstawowe prymitywy rysunku ----------------------------------------
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def text_w(fig, s, fs):
    """Szerokość napisu w calach — mierzona, nie szacowana."""
    t = fig.text(0, 0, s, fontsize=fs)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def line_h(fs):
    return fs * 1.42 / 72


def block(fig, lines):
    """(szerokość, wysokość) pudełka mieszczącego podane wiersze."""
    w = max(text_w(fig, s, fs) for s, fs, _c in lines) + 2 * PADX
    h = sum(line_h(fs) for _s, fs, _c in lines) + 2 * PADY
    return w, h


def draw_box(ax, cx, cy, w, h, lines, edge=AXIS, face="white", lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=face, edgecolor=edge, lw=lw, zorder=3))
    total = sum(line_h(fs) for _s, fs, _c in lines)
    y = cy + total / 2
    for s, fs, col in lines:
        y -= line_h(fs)
        ax.text(cx, y + line_h(fs) / 2, s, ha="center", va="center",
                fontsize=fs, color=col, zorder=4)


def arrow(ax, p0, p1, color=INK2, lw=1.1, ms=7.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 zorder=2))


def elbow(ax, pts, color=INK2, lw=1.1, r=0.13, head=True, ms=7.0):
    """Łamana z zaokrąglonymi narożnikami (opcjonalnie zakończona grotem)."""
    verts, codes = [pts[0]], [MplPath.MOVETO]
    for i in range(1, len(pts) - 1):
        (xa, ya), (xb, yb), (xc, yc) = pts[i - 1], pts[i], pts[i + 1]
        d1 = math.hypot(xa - xb, ya - yb)
        d2 = math.hypot(xc - xb, yc - yb)
        rr = min(r, d1 / 2, d2 / 2)
        verts += [(xb + (xa - xb) / d1 * rr, yb + (ya - yb) / d1 * rr),
                  (xb, yb),
                  (xb + (xc - xb) / d2 * rr, yb + (yc - yb) / d2 * rr)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    verts.append(pts[-1])
    codes.append(MplPath.LINETO)
    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes),
                                 arrowstyle="-|>" if head else "-",
                                 mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=2))


def spread(x0, x1, widths, weights):
    """Rozłóż pudełka między x0 a x1; zwraca listę środków."""
    free = (x1 - x0) - sum(widths)
    unit = free / sum(weights)
    xs, x = [], x0
    for i, w in enumerate(widths):
        xs.append(x + w / 2)
        x += w + (unit * weights[i] if i < len(weights) else 0)
    return xs


# =========================================================================
# 1a — szkielet
# =========================================================================
def render_small(out_dir):
    W, H = 6.6, 1.55
    fig, ax = canvas(W, H)
    FS, FSL = 7.3, 6.5

    def L(*rows):
        return [(s, FS, INK) for s in rows]

    nodes = {
        "in": L("Nagranie", "mono 16 kHz"),
        "dia": L("Diaryzacja"),
        "rou": L("Routing"),
        "enh": L("Poprawa", "jakości"),
        "sep": L("Separacja"),
        "asm": L("Składanie", "strumieni"),
        "asr": L("Transkrypcja"),
        "out": L("Transkrypcja", "z etykietami", "mówców"),
    }
    size = {k: block(fig, v) for k, v in nodes.items()}
    h_chain = max(size[k][1] for k in ("in", "dia", "rou", "asm", "asr", "out"))
    h_lane = max(size["enh"][1], size["sep"][1])
    w_fork = max(size["enh"][0], size["sep"][0])

    order = ["in", "dia", "rou", "fork", "asm", "asr", "out"]
    widths = [w_fork if k == "fork" else size[k][0] for k in order]
    cxs = spread(0.16, W - 0.16, widths, [1, 1, 1.75, 1.75, 1, 1])
    cx = dict(zip(order, cxs))

    y_mid = 0.78
    y_up = y_mid + 0.29
    y_lo = y_mid - 0.29

    # ---- pudełka
    for k in ("dia", "rou", "asm", "asr"):
        draw_box(ax, cx[k], y_mid, size[k][0], h_chain, nodes[k])
    for k in ("in", "out"):
        draw_box(ax, cx[k], y_mid, size[k][0], h_chain,
                 [(s, fs, INK2) for s, fs, _ in nodes[k]],
                 edge=AXIS, face="none")
    draw_box(ax, cx["fork"], y_up, w_fork, h_lane, nodes["enh"])
    draw_box(ax, cx["fork"], y_lo, w_fork, h_lane, nodes["sep"],
             edge=OVERLAP, face=TINT, lw=1.2)

    # ---- strzałki
    def right(k, w=None):
        return cx[k] + (w or size[k][0]) / 2

    def left(k, w=None):
        return cx[k] - (w or size[k][0]) / 2

    arrow(ax, (right("in"), y_mid), (left("dia"), y_mid))
    arrow(ax, (right("dia"), y_mid), (left("rou"), y_mid))
    arrow(ax, (right("asm"), y_mid), (left("asr"), y_mid))
    arrow(ax, (right("asr"), y_mid), (left("out"), y_mid))
    # rozwidlenie
    arrow(ax, (right("rou"), y_mid), (cx["fork"] - w_fork / 2, y_up), color=INK2)
    arrow(ax, (right("rou"), y_mid), (cx["fork"] - w_fork / 2, y_lo),
          color=OVERLAP)
    # scalenie
    arrow(ax, (cx["fork"] + w_fork / 2, y_up), (left("asm"), y_mid), color=INK2)
    arrow(ax, (cx["fork"] + w_fork / 2, y_lo), (left("asm"), y_mid),
          color=OVERLAP)

    # ---- etykiety ścieżek
    ax.text(cx["fork"], y_up + h_lane / 2 + 0.10, "całe nagranie", ha="center",
            va="bottom", fontsize=FSL, color=INK2)
    ax.text(cx["fork"], y_lo - h_lane / 2 - 0.10, "nakładania", ha="center",
            va="top", fontsize=FSL, color=INK2)

    save(fig, out_dir, "rys_ch6_pipeline_small")


def waveform(ax, cx, cy, w, h, color=INK, seed=7, n=520):
    """Mały glif przebiegu czasowego (jak w schematach rozdz. 3)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    env = np.abs(np.sin(np.pi * t * 3.1)) ** 1.4
    env *= 0.55 + 0.45 * np.sin(2 * np.pi * t * 0.9 + 0.7)
    y = np.clip(env, 0.06, None) * rng.standard_normal(n)
    y /= np.abs(y).max()
    ax.plot(cx - w / 2 + t * w, cy + y * h / 2, color=color, lw=0.55,
            solid_joinstyle="round", zorder=3)


# =========================================================================
# 1b — wersja pełna: potok tak, jak działa w konfiguracji `v41_merge`
# =========================================================================
def render_full(out_dir):
    W, H = 6.90, 7.05
    fig, ax = canvas(W, H)
    FT, FD, FL = 7.6, 6.2, 6.1     # tytuł / opis / etykieta krawędzi

    def N(title, *detail):
        return [(title, FT, INK)] + [(s, FD, INK2) for s in detail]

    dia = N("Diaryzacja",
            "Sortformer v1 (EEND), 2 mówców",
            "zwijanie głów nadmiarowych (ECAPA2)",
            "podproces w osobnym środowisku")
    rou = N("Routing",
            "wybór regionów nakładania",
            "odrzuć < 0,20 s, scal < 0,50 s")
    sep = N("Separacja",
            "MossFormer2-matched e46 (zamrożony)",
            "okno rozszerzane do 4 s",
            "16 → 8 → 16 kHz, sum_equals_mix",
            "+ obliczenie masek VAD (Silero)")
    bwe = N("AP-BWE", "odtworzenie", "pasma 4–8 kHz")
    msk = N("Nałożenie maski VAD", "wygaszenie energii", "resztkowej")
    enh = N("Poprawa jakości",
            "FRCRN 16 kHz, całe nagranie",
            "domieszka obserwacji r = 0,5")
    rel = N("Drugi przebieg etykietowania (B+)",
            "ECAPA2: fragmenty solo + strumienie nakładań",
            "wspólne klastrowanie 2-means",
            "ratunek podziałów zdegenerowanych")
    asm = N("Składanie strumieni",
            "kotwice ECAPA z fragmentów solo",
            "przypisanie nakładań: B+ / kosinus",
            "dopasowanie RMS, przenikanie 5 ms",
            "wyjście pełnej długości")
    asr = N("Transkrypcja",
            "WhisperX large-v2 (pl), wiązka 5",
            "wykryj i powtórz: okna zapadnięte,",
            "pętle tokenowe, pętle frazowe",
            "wyrównanie czasowe wav2vec2")
    out = [("transkrypcja per mówca", 7.0, INK2), ("+ znaczniki czasu", 7.0, INK2)]

    S = {k: block(fig, v) for k, v in
         dict(dia=dia, rou=rou, sep=sep, bwe=bwe, msk=msk, enh=enh, rel=rel,
              asm=asm, asr=asr, out=out).items()}

    # --- geometria pasów -------------------------------------------------
    ML, MR = 0.30, 0.28                  # marginesy boczne
    x_raw = 0.66                         # magistrala surowej mieszaniny
    x_lane = 1.10                        # lewa krawędź pasów 2 i 3
    x_ret = W - 0.20                     # prawy kanał powrotny (ścieżka nakładań)

    y1 = H - 0.60                        # wejście / diaryzacja / routing
    y2 = y1 - 1.55                       # ścieżka nakładań
    y3 = y2 - 1.32                       # ścieżka całego nagrania
    y4 = y3 - 1.22                       # drugi przebieg etykietowania
    y5 = y4 - 1.48                       # składanie / transkrypcja / wyjście

    def R(k):
        return P[k] + S[k][0] / 2

    def L(k):
        return P[k] - S[k][0] / 2

    P = {}

    # --- pas 1: nagranie → diaryzacja → routing --------------------------
    h1 = max(S["dia"][1], S["rou"][1])
    P["dia"] = 1.36 + S["dia"][0] / 2
    P["rou"] = R("dia") + 0.34 + S["rou"][0] / 2
    draw_box(ax, P["dia"], y1, S["dia"][0], h1, dia)
    draw_box(ax, P["rou"], y1, S["rou"][0], h1, rou)

    waveform(ax, x_raw, y1 + 0.13, 0.82, 0.30, color=INK)
    ax.text(x_raw, y1 - 0.13, "nagranie mono", ha="center", va="center",
            fontsize=FD, color=INK2)
    ax.text(x_raw, y1 - 0.13 - line_h(FD), "16 kHz", ha="center", va="center",
            fontsize=FD, color=INK2)
    arrow(ax, (x_raw + 0.46, y1), (L("dia"), y1))
    arrow(ax, (R("dia"), y1), (L("rou"), y1))
    ax.text((R("dia") + L("rou")) / 2, y1 + h1 / 2 + 0.07,
            "oś czasu wypowiedzi i nakładań", ha="center", va="bottom",
            fontsize=FL, color=INK2)

    # --- pas 2: ścieżka nakładań (fragmenty) -----------------------------
    h2 = max(S[k][1] for k in ("sep", "bwe", "msk"))
    P["sep"] = x_lane + S["sep"][0] / 2
    P["bwe"] = R("sep") + 0.26 + S["bwe"][0] / 2
    P["msk"] = R("bwe") + 0.26 + S["msk"][0] / 2
    draw_box(ax, P["sep"], y2, S["sep"][0], h2, sep, edge=OVERLAP, face=TINT,
             lw=1.2)
    draw_box(ax, P["bwe"], y2, S["bwe"][0], h2, bwe, edge=OVERLAP)
    draw_box(ax, P["msk"], y2, S["msk"][0], h2, msk, edge=OVERLAP)
    arrow(ax, (R("sep"), y2), (L("bwe"), y2), color=OVERLAP)
    arrow(ax, (R("bwe"), y2), (L("msk"), y2), color=OVERLAP)
    ax.text(x_lane, y2 + h2 / 2 + 0.10, "ścieżka nakładań — fragmenty",
            ha="left", va="bottom", fontsize=FL, color=OVERLAP)
    ax.text((R("bwe") + L("msk")) / 2, y2 - h2 / 2 - 0.07,
            "pasmo 0–4 → 0–8 kHz", ha="center", va="top", fontsize=FL,
            color=MUTED)

    # --- pas 3: ścieżka całego nagrania ----------------------------------
    P["enh"] = x_lane + S["enh"][0] / 2
    draw_box(ax, P["enh"], y3, S["enh"][0], S["enh"][1], enh)
    ax.text(x_lane, y3 + S["enh"][1] / 2 + 0.10, "ścieżka całego nagrania",
            ha="left", va="bottom", fontsize=FL, color=INK2)

    # --- magistrala surowej mieszaniny (ctx.audio) -----------------------
    elbow(ax, [(x_raw, y1 - 0.42), (x_raw, y2), (L("sep"), y2)], head=True,
          color=MUTED, r=0.10)
    elbow(ax, [(x_raw, y2), (x_raw, y3), (L("enh"), y3)], head=True,
          color=MUTED, r=0.10)
    ax.plot([x_raw], [y2], marker="o", ms=2.6, color=MUTED, zorder=3)
    ax.text(x_raw - 0.11, (y2 + y3) / 2, "surowa mieszanina, pełne nagranie",
            ha="center", va="center", rotation=90, fontsize=FL, color=MUTED)

    # --- routing → separacja (metadane, nie audio) -----------------------
    y_meta = y1 - 0.78
    elbow(ax, [(P["rou"], y1 - h1 / 2), (P["rou"], y_meta),
               (P["sep"] + 0.62, y_meta), (P["sep"] + 0.62, y2 + h2 / 2)],
          color=OVERLAP, r=0.12)
    ax.text(P["rou"] - 0.10, y_meta + 0.10, "regiony nakładania (metadane)",
            ha="right", va="bottom", fontsize=FL, color=OVERLAP)

    # --- pas 4: drugi przebieg etykietowania -----------------------------
    P["rel"] = x_lane + 0.55 + S["rel"][0] / 2
    draw_box(ax, P["rel"], y4, S["rel"][0], S["rel"][1], rel)
    h5 = max(S[k][1] for k in ("asm", "asr", "out"))
    P["asm"] = ML + 0.06 + S["asm"][0] / 2
    P["asr"] = R("asm") + 0.82 + S["asr"][0] / 2
    P["out"] = R("asr") + 0.42 + S["out"][0] / 2

    # nakładania → relabel (osadzenia) ORAZ → składanie (audio nakładań);
    # ten sam sygnał trafia w dwa miejsca — B+ tylko go osadza, a wycinane do
    # strumieni fragmenty bierze bezpośrednio składanie
    y_gs = (y4 + y5) / 2 - 0.20
    elbow(ax, [(R("msk"), y2), (x_ret, y2), (x_ret, y4), (R("rel"), y4)],
          color=OVERLAP, r=0.14)
    elbow(ax, [(x_ret, y4), (x_ret, y_gs), (P["asm"] + 0.55, y_gs),
               (P["asm"] + 0.55, y5 + h5 / 2)], color=OVERLAP, r=0.12)
    ax.plot([x_ret], [y4], marker="o", ms=2.8, color=OVERLAP, zorder=3)
    ax.text(x_ret - 0.06, (y2 + y4) / 2, "strumienie s1 / s2 (bramkowane)",
            ha="center", va="center", rotation=90, fontsize=FL, color=OVERLAP)
    ax.text(R("rel") + 0.10, y4 + 0.07, "do osadzeń", ha="left", va="bottom",
            fontsize=FL, color=OVERLAP)
    ax.text(P["asm"] + 0.65, y_gs + 0.06, "audio nakładań — wklejane w strumienie",
            ha="left", va="bottom", fontsize=FL, color=OVERLAP)

    # enhanced_full → relabel oraz → składanie (magistrala po lewej)
    y_eb = (y3 + y4) / 2
    x_eb = 0.20
    elbow(ax, [(P["enh"], y3 - S["enh"][1] / 2), (P["enh"], y_eb),
               (P["rel"] - 0.55, y_eb), (P["rel"] - 0.55, y4 + S["rel"][1] / 2)],
          color=INK2, r=0.11)
    elbow(ax, [(P["enh"], y_eb), (x_eb, y_eb), (x_eb, y5), (L("asm"), y5)],
          color=INK2, r=0.11)
    ax.plot([P["enh"]], [y_eb], marker="o", ms=2.6, color=INK2, zorder=3)
    ax.text(P["enh"] + 0.08, y_eb + 0.06,
            "sygnał po poprawie jakości — fragmenty solo i kotwice mówców",
            ha="left", va="bottom", fontsize=FL, color=INK2)

    # --- pas 5: składanie → transkrypcja → wyjście -----------------------
    draw_box(ax, P["asm"], y5, S["asm"][0], h5, asm)
    draw_box(ax, P["asr"], y5, S["asr"][0], h5, asr)
    ax.text(P["out"], y5 + line_h(7.0) / 2, "transkrypcja per mówca",
            ha="center", va="center", fontsize=7.0, color=INK2)
    ax.text(P["out"], y5 - line_h(7.0) / 2, "+ znaczniki czasu",
            ha="center", va="center", fontsize=7.0, color=INK2)

    # relabel → składanie
    elbow(ax, [(P["rel"], y4 - S["rel"][1] / 2), (P["rel"], (y4 + y5) / 2),
               (P["asm"] - 0.55, (y4 + y5) / 2 + 0.12),
               (P["asm"] - 0.55, y5 + h5 / 2)], color=INK2, r=0.12)
    ax.text(P["rel"] + 0.10, (y4 + y5) / 2 + 0.18,
            "etykiety mówców + przypisanie nakładań", ha="left", va="bottom",
            fontsize=FL, color=INK2)

    xa, xb = R("asm"), L("asr")
    for dy, col, lab in ((0.12, SPK_A, "mówca A"), (-0.12, SPK_B, "mówca B")):
        arrow(ax, (xa, y5 + dy), (xb, y5 + dy), color=col, lw=1.2)
        ax.text((xa + xb) / 2, y5 + dy + (0.09 if dy > 0 else -0.09), lab,
                ha="center", va="bottom" if dy > 0 else "top", fontsize=FL,
                color=INK)
    arrow(ax, (R("asr"), y5), (L("out") - 0.02, y5))

    save(fig, out_dir, "rys_ch6_pipeline_full")


# =========================================================================
# 1c — wersja docelowa (projekt autora): etapy + symbole sygnałów
# =========================================================================
# Wszystkie symbole sygnału rysowane są z tą samą gęstością — tyle próbek i
# tyle grup energii na cal rysunku — więc przebieg pełnego nagrania wygląda na
# równie „gęsty" jak fragment czy strumień, mimo innej szerokości glifu. Bez
# tego dłuższy sygnał ściśnięty do tej samej szerokości wyglądałby na
# rozciągnięty, a krótszy — na gęsty, co sugerowałoby nieistniejącą różnicę.
SPI = 900.0    # próbek na cal
BPI = 3.4      # grup energii (sylab) na cal


def _trace(seed, span, floor=0.08, gate=None):
    """Znormalizowany, powtarzalny przebieg o obwiedni „mowopodobnej".

    `span` to szerokość rysowanego odcinka w calach — z niej wynika liczba
    próbek i liczba grup energii, więc gęstość jest niezależna od glifu.
    """
    n = max(160, int(span * SPI))
    bursts = max(0.9, span * BPI)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    env = np.abs(np.sin(np.pi * (t * bursts + 0.13 * seed))) ** 1.35
    env *= 0.55 + 0.45 * np.sin(2 * np.pi * t * (0.45 * bursts) + 0.6 * seed)
    env = np.clip(np.abs(env), floor, None)
    y = env * rng.standard_normal(n)
    y /= np.abs(y).max()
    if gate is not None:
        y = y * gate(t)
    return t, y, env


def glyph_wave(ax, cx, cy, w, h, color=INK, seed=7, floor=0.08, gated=False,
               lw=0.55):
    """`gated=True` wycisza tło między grupami energii — obraz sygnału po
    poprawie jakości, a nie innego sygnału."""
    t, y, env = _trace(seed, w, floor=0.035 if gated else floor)
    if gated:
        y = y * (env > 0.17 * env.max())
    ax.plot(cx - w / 2 + t * w, cy + y * h / 2, color=color, lw=lw, zorder=3)


def glyph_frags(ax, cx, cy, w, h, colors, seed=11, k=3, gap=0.055,
                windows=None):
    """k krótkich fragmentów; jeden wiersz na kolor (strumień).

    `windows[r][i] = (a, b)` wyznacza, w której części fragmentu i dany mówca r
    jest aktywny — dwaj mówcy nakładają się częściowo, nie identycznie.
    """
    rows = len(colors)
    sw = (w - gap * (k - 1)) / k
    rh = h / rows if rows == 1 else (h - 0.05 * (rows - 1)) / rows
    for r, col in enumerate(colors):
        ry = cy + (h - rh) / 2 - r * (rh + 0.05) if rows > 1 else cy
        for i in range(k):
            x0 = cx - w / 2 + i * (sw + gap)
            g = None
            if windows is not None:
                a, b = windows[r][i]
                g = (lambda a, b: (lambda t: ((t > a) & (t < b)).astype(float)))(a, b)
            t, y, _ = _trace(seed + 7 * i + 31 * r, sw, floor=0.14, gate=g)
            ax.plot(x0 + t * sw, ry + y * rh / 2, color=col, lw=0.5, zorder=3)
        ax.plot([cx - w / 2, cx + w / 2], [ry, ry], color=GRID, lw=0.4, zorder=2)


# Kto kiedy mówi w każdym z trzech fragmentów nakładania: mówcy zachodzą na
# siebie częściowo, a nie w identycznych momentach. Po bramkowaniu VAD-em
# krawędzie są dodatkowo dociągnięte do faktycznej aktywności.
WIN_SEP = [[(0.04, 0.88), (0.00, 0.68), (0.20, 1.00)],
           [(0.20, 1.00), (0.26, 1.00), (0.00, 0.64)]]
WIN_VAD = [[(0.12, 0.78), (0.06, 0.56), (0.32, 0.94)],
           [(0.30, 0.92), (0.38, 0.90), (0.08, 0.50)]]


def glyph_streams(ax, cx, cy, w, h, seed=23):
    """Dwa strumienie pełnej długości, uzupełniające się w czasie."""
    rh = (h - 0.06) / 2
    turns = [(0.02, 0.34), (0.44, 0.72), (0.78, 0.99)]      # mówca A
    turns_b = [(0.30, 0.42), (0.62, 0.80), (0.86, 0.98)]    # mówca B (z nakładką)
    for r, (col, segs) in enumerate(((SPK_A, turns), (SPK_B, turns_b))):
        ry = cy + (h - rh) / 2 - r * (rh + 0.06)
        ax.plot([cx - w / 2, cx + w / 2], [ry, ry], color=GRID, lw=0.4, zorder=2)
        for j, (a, b) in enumerate(segs):
            t, y, _ = _trace(seed + 5 * j + 17 * r, (b - a) * w, floor=0.16)
            ax.plot(cx - w / 2 + (a + t * (b - a)) * w, ry + y * rh / 2,
                    color=col, lw=0.5, zorder=3)


def glyph_doc(ax, cx, cy, w, h):
    """Dwie kartki transkrypcji, jedna za drugą."""
    off = 0.055
    for k, (dx, dy, face) in enumerate(((off, off, "white"), (0, 0, "white"))):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2 + dx, cy - h / 2 + dy), w, h,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=face, edgecolor=AXIS, lw=0.9, zorder=3 + k))
    rng = np.random.default_rng(5)
    n = 6
    for i in range(n):
        y = cy + h / 2 - 0.085 - i * (h - 0.15) / (n - 1)
        frac = 0.55 + 0.35 * rng.random()
        x0 = cx - w / 2 + 0.09
        ax.plot([x0, x0 + (w - 0.18) * frac], [y, y], color=MUTED, lw=0.9,
                solid_capstyle="round", zorder=5)


def render_v2(out_dir):
    W, H = 6.20, 8.80
    fig, ax = canvas(W, H)
    FT, FL = 8.2, 6.6

    xC = W / 2
    xL, xR = xC - 1.42, xC + 1.42
    W_WIDE, HB = 4.80, 0.46

    def stage(cx, cy, label, w=None, **kw):
        bw = w or (text_w(fig, label, FT) + 0.44)
        draw_box(ax, cx, cy, bw, HB, [(label, FT, INK)], **kw)
        return bw

    def link(x, y_top, y_bot, sym=None, caps=(), sh=0.28, color=INK2,
             cap_color=None):
        """Pionowe połączenie przerwane symbolem sygnału i jego podpisem."""
        cap_color = cap_color or color
        # blok „symbol + podpis" kotwiczony od dołu, tak by strzałka wchodząca
        # do kolejnego etapu miała widoczną długość, a nie sam grot; górne
        # ograniczenie chroni przed wejściem symbolu w pudełko nad nim
        y_sym = min(y_bot + 0.20 + len(caps) * line_h(FL) + 0.04 + sh / 2,
                    y_top - 0.10 - sh / 2)
        if sym is None:
            arrow(ax, (x, y_top), (x, y_bot), color=color)
            return
        elbow(ax, [(x, y_top), (x, y_sym + sh / 2 + 0.06)], head=False,
              color=color)
        sym(x, y_sym, sh)
        y = y_sym - sh / 2 - 0.04
        for s in caps:
            ax.text(x, y, s, ha="center", va="top", fontsize=FL, color=cap_color)
            y -= line_h(FL)
        arrow(ax, (x, y - 0.04), (x, y_bot), color=color)

    # --- poziomy ---------------------------------------------------------
    y_wav = H - 0.30
    y_spl = y_wav - 0.55
    y_dia = y_spl - 0.38
    y_rou = y_dia - 1.00
    y_st2 = y_rou - 1.34
    y_bwe = y_st2 - 1.25
    y_rel = y_bwe - 1.38
    y_asr = y_rel - 1.29
    y_doc = y_asr - 0.88

    # --- wejście i rozdział sygnału --------------------------------------
    glyph_wave(ax, xC, y_wav, 1.00, 0.30, color=INK, seed=7)
    ax.text(xC, y_wav - 0.21, "Pełne nagranie", ha="center", va="top",
            fontsize=FL, color=INK2)

    elbow(ax, [(xC, y_wav - 0.40), (xC, y_spl)], head=False, color=INK2)
    elbow(ax, [(xC, y_spl), (xL, y_spl), (xL, y_dia + HB / 2)], color=INK2,
          r=0.10)
    elbow(ax, [(xC, y_spl), (xR, y_spl), (xR, y_rou + HB / 2)], color=INK2,
          r=0.10)
    ax.plot([xC], [y_spl], marker="o", ms=2.8, color=INK2, zorder=4)

    # --- diaryzacja → routing (metadane, bez sygnału) --------------------
    stage(xL, y_dia, "Diaryzacja")
    link(xL, y_dia - HB / 2, y_rou + HB / 2)
    ymid = (y_dia - y_rou) / 2 + y_rou
    ax.text(xL - 0.10, ymid + line_h(FL) / 2, "znaczniki", ha="right",
            va="center", fontsize=FL, color=INK2)
    ax.text(xL - 0.10, ymid - line_h(FL) / 2, "czasowe", ha="right",
            va="center", fontsize=FL, color=INK2)

    stage(xC, y_rou, "Routing", w=W_WIDE)

    # --- rozgałęzienie na dwie ścieżki -----------------------------------
    link(xL, y_rou - HB / 2, y_st2 + HB / 2,
         sym=lambda x, y, h: glyph_wave(ax, x, y, 1.02, h, color=INK2, seed=7),
         caps=("Pełne nagranie",), sh=0.26)
    link(xR, y_rou - HB / 2, y_st2 + HB / 2, cap_color=INK2,
         sym=lambda x, y, h: glyph_frags(ax, x, y, 1.02, h, [OVERLAP], seed=11),
         caps=("fragmenty", "z nakładaniem"), sh=0.24)

    stage(xL, y_st2, "Enhancement")
    stage(xR, y_st2, "Separator")

    # --- ścieżka całego nagrania -----------------------------------------
    link(xL, y_st2 - HB / 2, y_rel + HB / 2, sh=0.26,
         sym=lambda x, y, h: glyph_wave(
             ax, x, y, 1.02, h, color=INK2, seed=7, gated=True),
         caps=("Pełne nagranie", "o poprawionej jakości"))

    # --- ścieżka nakładań -------------------------------------------------
    link(xR, y_st2 - HB / 2, y_bwe + HB / 2,
         sym=lambda x, y, h: glyph_frags(ax, x, y, 1.06, h, [SPK_A, SPK_B],
                                         seed=11, windows=WIN_SEP),
         caps=("rozdzielone fragmenty",), sh=0.30)
    stage(xR, y_bwe, "BWE i VAD")
    link(xR, y_bwe - HB / 2, y_rel + HB / 2,
         sym=lambda x, y, h: glyph_frags(ax, x, y, 1.06, h, [SPK_A, SPK_B],
                                         seed=4, windows=WIN_VAD),
         caps=("rozdzielone fragmenty", "po obróbce"), sh=0.30)

    # --- relabel + assembly → transkrypcja -------------------------------
    stage(xC, y_rel, "Relabel i Assembly", w=W_WIDE)
    link(xC, y_rel - HB / 2, y_asr + HB / 2, sh=0.34,
         sym=lambda x, y, h: glyph_streams(ax, x, y, 1.34, h),
         caps=("w pełni odseparowane sygnały obu mówców",))

    w_asr = stage(xC, y_asr, "WhisperX")
    ax.add_patch(FancyArrowPatch(
        (xC + w_asr / 2, y_asr + 0.13), (xC + w_asr / 2, y_asr - 0.13),
        connectionstyle="arc3,rad=-1.85", arrowstyle="-|>", mutation_scale=7.0,
        lw=1.0, color=INK2, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(xC + w_asr / 2 + 0.44, y_asr, "D&R", ha="left",
            va="center", fontsize=FL, color=INK2)

    arrow(ax, (xC, y_asr - HB / 2), (xC, y_doc + 0.42), color=INK2)
    glyph_doc(ax, xC, y_doc, 0.80, 0.56)
    ax.text(xC, y_doc - 0.36, "Transkrypcje", ha="center", va="top",
            fontsize=FL, color=INK2)

    save(fig, out_dir, "rys_ch6_pipeline_v2")


# --- symbole z PRAWDZIWEGO nagrania -----------------------------------------
# Wariant zwarty rysuje symbole sygnałów z jednego rzeczywistego nagrania
# CLARIN przepuszczonego przez potok, tak by szczyty i dynamika strumieni po
# składaniu odpowiadały temu, co widać w nagraniu wejściowym. Źródłem jest
# zrzut pośrednich sygnałów z webapp (`~/webapp_jobs/<job>/spill/`: nagranie,
# enhanced_full, okna nakładań 4 s wejściowe/rozdzielone/bramkowane, strumienie
# po składaniu). Obwiednie (min/maks w koszykach) są buforowane w
# `data/pipeline_signals_<nagranie>.npz`, więc rysunek odtwarza się bez dostępu
# do zrzutu; `--refresh-signals` przelicza bufor ze źródła.
#
# Skalowanie: nagranie / po poprawie / strumienie dzielą JEDNĄ skalę (maksimum
# nagrania surowego), więc amplitudy strumieni są porównywalne z mieszaniną.
# Symbole fragmentów mają własną wspólną skalę (maksimum w obrębie symbolu).
SIGNALS_ROOT = Path.home() / "webapp_jobs"
DATA_DIR = Path(__file__).resolve().parent / "data"

# nagranie CLARIN → (job webapp, okno [s] pokazywane na rysunku)
RECORDINGS = {
    "026eafb1__seg00": ("6aafc365a18c", (0.0, 22.0)),
    "88741282__seg01": ("ef97d4cd1a87", (8.0, 28.0)),
    "f258d8fa__seg01": ("7b019332f54d", (0.0, 22.0)),
    "ce622a38__seg00": ("bf135d7ed131", (0.0, 16.0)),
    "0b54e2ca__seg00": ("013584e12337", (10.0, 30.0)),
    "d9c4f5df__seg00": ("f244d15929f6", (8.0, 28.0)),
    "18810179__cand00": ("96a0e52fd7b4", (20.0, 40.0)),
}
RECORDING_DEFAULT = "026eafb1__seg00"      # wybór autora 2026-08-23
N_FRAGS = 3                                # tyle okien nakładania pokazuje symbol
ENV_BINS, ENV_BINS_FRAG = 4000, 1200       # rozdzielczość buforowanych obwiedni


def envelope(sig, nb):
    """Obwiednia min/maks sygnału w `nb` koszykach → tablica (2, nb)."""
    sig = np.asarray(sig, dtype=np.float32)
    n = len(sig) // nb
    x = sig[: n * nb].reshape(nb, n)
    return np.stack([x.min(axis=1), x.max(axis=1)])


def rebin(env, nb):
    """Zgrubienie obwiedni (2, N) do (2, nb): minimum z minimów, maksimum z maksimów."""
    n = env.shape[1] // nb
    lo = env[0, : n * nb].reshape(nb, n).min(axis=1)
    hi = env[1, : n * nb].reshape(nb, n).max(axis=1)
    return np.stack([lo, hi])


def _read_recording(name, window):
    """Czyta zrzut webapp i zwraca obwiednie (sygnały główne + okna nakładań)."""
    import json
    import soundfile as sf
    job, _ = RECORDINGS[name]
    d = SIGNALS_ROOT / job
    sp = d / "spill"
    raw, sr = sf.read(d / f"{job}.wav")
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    enh, _ = sf.read(sp / "enhanced_full.wav")
    a, _ = sf.read(sp / "assembled_A.wav")
    b, _ = sf.read(sp / "assembled_B.wav")
    n = min(len(raw), len(enh), len(a), len(b))
    i0, i1 = (0, n) if window is None else (int(window[0] * sr), int(window[1] * sr))
    scale = np.abs(raw[i0:i1]).max()
    out = {k: envelope(np.clip(v[i0:i1] / scale, -1, 1), ENV_BINS)
           for k, v in dict(raw=raw, enh=enh, A=a, B=b).items()}

    # okna nakładań: najdłuższe N_FRAGS regiony wewnątrz okna, chronologicznie;
    # mieszanina = wycinek nagrania o granicach okna separatora (pad_start..pad_end),
    # rozdzielone / bramkowane = pliki zrzutu (to samo okno 4 s)
    meta = json.load(open(sp / "separation_metadata.json"))
    t0, t1 = (0.0, n / sr) if window is None else window
    ovs = [o for o in meta["overlaps"] if o["pad_start"] >= t0 and o["pad_end"] <= t1]
    ovs = sorted(sorted(ovs, key=lambda o: o["end"] - o["start"])[-N_FRAGS:],
                 key=lambda o: o["start"])
    frags = {"mix": [], "s1": [], "s2": [], "g1": [], "g2": []}
    for o in ovs:
        k = o["idx"]
        frags["mix"].append(raw[int(o["pad_start"] * sr): int(o["pad_end"] * sr)])
        frags["s1"].append(sf.read(sp / f"overlap_{k}_s1_raw.wav")[0])
        frags["s2"].append(sf.read(sp / f"overlap_{k}_s2_raw.wav")[0])
        frags["g1"].append(sf.read(sp / f"overlap_{k}_s1_gated_bwe_ap_bwe.wav")[0])
        frags["g2"].append(sf.read(sp / f"overlap_{k}_s2_gated_bwe_ap_bwe.wav")[0])
    for key, group in (("mix", ("mix",)), ("sep", ("s1", "s2")), ("gat", ("g1", "g2"))):
        m = max(np.abs(x).max() for g in group for x in frags[g])
        for g in group:
            out[f"frag_{g}"] = np.stack([envelope(x / m, ENV_BINS_FRAG) for x in frags[g]])
    return out


def load_recording(name=RECORDING_DEFAULT, window=None, refresh=False):
    """Obwiednie z bufora `data/`; przy braku (lub `refresh`) — ze zrzutu webapp."""
    if window is None:
        window = RECORDINGS[name][1]
    cache = DATA_DIR / f"pipeline_signals_{name}_{int(window[0])}-{int(window[1])}s.npz"
    if cache.exists() and not refresh:
        return dict(np.load(cache))
    rec = _read_recording(name, window)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, **rec)
    print(f"cached: {cache}")
    return rec


def glyph_real(ax, cx, cy, w, h, env, color=INK, lw=0.5):
    """Obwiednia prawdziwego sygnału, tą samą gęstością (SPI) co symbole
    syntetyczne — w druku wygląda jak wypełniony przebieg."""
    nb = max(60, int(w * SPI / 2))
    env = rebin(env, min(nb, env.shape[1]))
    ys = np.empty(2 * env.shape[1])
    ys[0::2], ys[1::2] = env[0], env[1]
    xs = np.repeat(np.linspace(0.0, 1.0, env.shape[1]), 2)
    ax.plot(cx - w / 2 + xs * w, cy + ys * h / 2, color=color, lw=lw, zorder=3)


def glyph_streams_real(ax, cx, cy, w, h, rec):
    rh = (h - 0.06) / 2
    for r, (col, key) in enumerate(((SPK_A, "A"), (SPK_B, "B"))):
        ry = cy + (h - rh) / 2 - r * (rh + 0.06)
        ax.plot([cx - w / 2, cx + w / 2], [ry, ry], color=GRID, lw=0.4, zorder=2)
        glyph_real(ax, cx, ry, w, rh, rec[key], color=col, lw=0.5)


def glyph_frags_real(ax, cx, cy, w, h, rec, keys, colors, gap=0.055):
    """Okna nakładań z prawdziwego nagrania; jeden wiersz na strumień."""
    rows = len(keys)
    k = rec[f"frag_{keys[0]}"].shape[0]
    sw = (w - gap * (k - 1)) / k
    rh = h if rows == 1 else (h - 0.05 * (rows - 1)) / rows
    for r, (key, col) in enumerate(zip(keys, colors)):
        ry = cy + (h - rh) / 2 - r * (rh + 0.05) if rows > 1 else cy
        for i in range(k):
            x0 = cx - w / 2 + i * (sw + gap)
            glyph_real(ax, x0 + sw / 2, ry, sw, rh, rec[f"frag_{key}"][i], color=col)
        ax.plot([cx - w / 2, cx + w / 2], [ry, ry], color=GRID, lw=0.4, zorder=2)


def render_v2_compact(out_dir, rec_name=RECORDING_DEFAULT, window=None,
                      stem="rys_ch6_pipeline_v2_compact", refresh=False):
    """Wariant zwarty: te same etapy i symbole, ale podpisy stoją OBOK
    symboli, a nie pod nimi — pionowy koszt każdego połączenia spada z
    „symbol + podpis" do „max(symbol, podpis)".

    Diaryzacja stoi W OSI rysunku, nad routingiem — jest wspólna dla obu
    ścieżek, a nie częścią ścieżki poprawy jakości; surowy sygnał omija ją
    dwoma bocznymi odgałęzieniami. `rec_name` (klucz `RECORDINGS`) podmienia
    symbole pełnego nagrania / po poprawie / strumieni na prawdziwe sygnały
    (`window` = wycinek w sekundach, None = okno domyślne z `RECORDINGS`);
    `rec_name=None` przywraca symbole syntetyczne."""
    rec = load_recording(rec_name, window, refresh) if rec_name else None
    W, H = 6.20, 6.55
    fig, ax = canvas(W, H)
    FT, FL = 9.0, 7.4

    xC = W / 2
    xL, xR = xC - 1.20, xC + 1.20
    W_WIDE, HB = 4.40, 0.40

    def stage(cx, cy, label, w=None, **kw):
        bw = w or (text_w(fig, label, FT) + 0.40)
        draw_box(ax, cx, cy, bw, HB, [(label, FT, INK)], **kw)
        return bw

    def link(x, y_top, y_bot, sym=None, caps=(), sh=0.26, side="right",
             sym_w=1.30, color=INK2, cap_color=INK2):
        if sym is None:
            arrow(ax, (x, y_top), (x, y_bot), color=color)
            return
        blk = max(sh, len(caps) * line_h(FL))
        y_sym = min(y_bot + 0.14 + blk / 2, y_top - 0.06 - blk / 2)
        elbow(ax, [(x, y_top), (x, y_sym + sh / 2 + 0.03)], head=False,
              color=color)
        sym(x, y_sym, sh)
        arrow(ax, (x, y_sym - sh / 2 - 0.03), (x, y_bot), color=color)
        xa = x + sym_w / 2 + 0.13 if side == "right" else x - sym_w / 2 - 0.13
        ha = "left" if side == "right" else "right"
        y = y_sym + (len(caps) - 1) * line_h(FL) / 2
        for s in caps:
            ax.text(xa, y, s, ha=ha, va="center", fontsize=FL,
                    color=cap_color, style="italic")
            y -= line_h(FL)

    # --- poziomy ---------------------------------------------------------
    y_wav = H - 0.26
    y_spl = y_wav - 0.44
    y_dia = y_spl - 0.32
    y_rou = y_dia - 0.86
    y_st2 = y_rou - 0.88
    y_bwe = y_st2 - 0.88
    y_rel = y_bwe - 0.88
    y_asr = y_rel - 0.92
    y_doc = y_asr - 0.66

    # --- wejście ---------------------------------------------------------
    def sym_full(x, y, h, color):
        if rec is None:
            glyph_wave(ax, x, y, 1.02, h, color=color, seed=7)
        else:
            glyph_real(ax, x, y, 1.02, h, rec["raw"], color=color)

    def sym_enh(x, y, h):
        if rec is None:
            glyph_wave(ax, x, y, 1.02, h, color=INK2, seed=7, gated=True)
        else:
            glyph_real(ax, x, y, 1.02, h, rec["enh"], color=INK2)

    def sym_streams(x, y, h):
        if rec is None:
            glyph_streams(ax, x, y, 1.30, h)
        else:
            glyph_streams_real(ax, x, y, 1.30, h, rec)

    sym_full(xC, y_wav, 0.28, INK)
    ax.text(xC + 0.60, y_wav + line_h(FL) / 2, "pełne nagranie", ha="left",
            va="center", fontsize=FL, color=INK2, style="italic")
    ax.text(xC + 0.60, y_wav - line_h(FL) / 2, "mono, 16 kHz", ha="left",
            va="center", fontsize=FL, color=INK2, style="italic")

    # --- diaryzacja w osi (wspólna dla obu ścieżek) → routing ------------
    # surowy sygnał: pień do punktu rozdziału, stamtąd do diaryzacji ORAZ
    # dwoma bocznymi odgałęzieniami — omijając ją — wprost do routingu
    elbow(ax, [(xC, y_wav - 0.18), (xC, y_spl)], head=False, color=INK2)
    arrow(ax, (xC, y_spl), (xC, y_dia + HB / 2), color=INK2)
    for xs in (xL, xR):
        elbow(ax, [(xC, y_spl), (xs, y_spl), (xs, y_rou + HB / 2)],
              color=INK2, r=0.09)
    ax.plot([xC], [y_spl], marker="o", ms=2.6, color=INK2, zorder=4)

    stage(xC, y_dia, "Diaryzacja")
    link(xC, y_dia - HB / 2, y_rou + HB / 2)
    ax.text(xC + 0.09, (y_dia - y_rou) / 2 + y_rou, "znaczniki czasowe",
            ha="left", va="center", fontsize=FL, color=INK2,
            style="italic")

    stage(xC, y_rou, "Routing", w=W_WIDE)

    # --- dwie ścieżki -----------------------------------------------------
    link(xL, y_rou - HB / 2, y_st2 + HB / 2, side="left", sh=0.24,
         sym=lambda x, y, h: sym_full(x, y, h, INK2),
         caps=("pełne", "nagranie"))
    def sym_frags(x, y, h, which, w):
        if rec is None:
            if which == "mix":
                glyph_frags(ax, x, y, w, h, [OVERLAP], seed=11)
            else:
                glyph_frags(ax, x, y, w, h, [SPK_A, SPK_B],
                            seed=11 if which == "sep" else 4,
                            windows=WIN_SEP if which == "sep" else WIN_VAD)
        elif which == "mix":
            glyph_frags_real(ax, x, y, w, h, rec, ("mix",), (OVERLAP,))
        else:
            keys = ("s1", "s2") if which == "sep" else ("g1", "g2")
            glyph_frags_real(ax, x, y, w, h, rec, keys, (SPK_A, SPK_B))

    link(xR, y_rou - HB / 2, y_st2 + HB / 2, side="right", sh=0.22,
         sym=lambda x, y, h: sym_frags(x, y, h, "mix", 1.02),
         caps=("fragmenty", "z nakładaniem"))

    stage(xL, y_st2, "Enhancement")
    w_sep = stage(xR, y_st2, "Separator")
    # częstotliwości próbkowania — jedyne miejsce, gdzie potok schodzi do 8 kHz,
    # to wnętrze separatora; AP-BWE odtwarza pasmo, nie częstotliwość próbkowania
    ax.text(xR + w_sep / 2 + 0.10, y_st2, "16 → 8 → 16 kHz", ha="left",
            va="center", fontsize=FL - 0.8, color=MUTED, style="italic")

    link(xL, y_st2 - HB / 2, y_rel + HB / 2, side="left", sh=0.24,
         sym=sym_enh, caps=("pełne nagranie", "o poprawionej jakości"))

    link(xR, y_st2 - HB / 2, y_bwe + HB / 2, side="right", sh=0.26,
         sym_w=1.06, sym=lambda x, y, h: sym_frags(x, y, h, "sep", 1.06),
         caps=("rozdzielone", "fragmenty"))
    w_bwe = stage(xR, y_bwe, "BWE i VAD")
    ax.text(xR + w_bwe / 2 + 0.10, y_bwe, "pasmo 4–8 kHz", ha="left",
            va="center", fontsize=FL - 0.8, color=MUTED, style="italic")
    link(xR, y_bwe - HB / 2, y_rel + HB / 2, side="right", sh=0.26,
         sym_w=1.06, sym=lambda x, y, h: sym_frags(x, y, h, "gat", 1.06),
         caps=("rozdzielone fragmenty", "po obróbce"))

    # --- relabel + assembly → transkrypcja -------------------------------
    stage(xC, y_rel, "Relabel i Assembly", w=W_WIDE)
    link(xC, y_rel - HB / 2, y_asr + HB / 2, side="right", sh=0.30,
         sym_w=1.30, sym=sym_streams,
         caps=("w pełni odseparowane", "sygnały obu mówców"))

    w_asr = stage(xC, y_asr, "WhisperX")
    ax.add_patch(FancyArrowPatch(
        (xC + w_asr / 2, y_asr + 0.11), (xC + w_asr / 2, y_asr - 0.11),
        connectionstyle="arc3,rad=-1.85", arrowstyle="-|>", mutation_scale=6.5,
        lw=1.0, color=INK2, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(xC + w_asr / 2 + 0.38, y_asr, "D&R", ha="left", va="center",
            fontsize=FL, color=INK2,
            style="italic")

    arrow(ax, (xC, y_asr - HB / 2), (xC, y_doc + 0.36), color=INK2)
    glyph_doc(ax, xC, y_doc, 0.68, 0.48)
    ax.text(xC + 0.46, y_doc, "transkrypcje", ha="left", va="center",
            fontsize=FL, color=INK2,
            style="italic")

    save(fig, out_dir, stem)


def save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out_dir}/{stem}.{{png,pdf}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--refresh-signals", action="store_true",
                    help="przelicz bufor obwiedni w data/ ze zrzutu webapp")
    ap.add_argument("--variants", type=Path, default=None,
                    help="katalog na warianty zwartego schematu z prawdziwymi "
                         "nagraniami (każde z RECORDINGS: całe + wycinek)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "font.family": "sans-serif",
    })
    render_small(args.out_dir)
    render_full(args.out_dir)
    render_v2(args.out_dir)
    render_v2_compact(args.out_dir, refresh=args.refresh_signals)
    if args.variants is not None:
        args.variants.mkdir(parents=True, exist_ok=True)
        render_v2_compact(args.variants, rec_name=None, stem="compact__synthetic")
        for name, (_job, win) in RECORDINGS.items():
            render_v2_compact(args.variants, rec_name=name, window=win, refresh=True,
                              stem=f"compact__{name}__win{int(win[0])}-{int(win[1])}")


if __name__ == "__main__":
    main()
