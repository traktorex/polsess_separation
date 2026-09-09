"""Rysunek 4.2 — struktura projektu `polsess_separation`.

Pionowy tor pojedynczego eksperymentu (definicje → trening → zapisane pliki →
wykorzystanie modeli) oraz biblioteka projektu jako gałąź boczna, z której
korzystają zarówno trening, jak i ewaluacja. Druga strzałka z biblioteki niesie
argument o rzetelności porównań: ewaluator odtwarza modele przez ten sam
rejestr, którego użył trening.

Rysunek powstał pierwotnie skryptem, którego nie zachowano; niniejszy plik
odtwarza go z pomiarów opublikowanej wersji (2160×1559 px, 300 dpi), więc
wszystkie stałe układu podane są w pikselach tamtego renderu i przeliczane na
cale. Odtworzenie wprowadza dwie zmiany zamówione po recenzji promotora:

  * usunięto wiersz `train_sweep.py` — słowo „sweep" pojawiało się na rysunku
    ponad sto wierszy przed podrozdziałem 4.6, w którym jest wprowadzane,
  * pudełko „ARTEFAKTY" przemianowano na „ZAPISANE PLIKI" — w rozdziałach 2, 6
    i 7 „artefakt" oznacza zniekształcenie sygnału, więc znaczenie
    inżynierskie kolidowało z terminologią pracy.

Układ liczony jest w calach (1 jednostka danych = 1 cal, `set_aspect("equal")`),
tak jak w schematach rozdziału 6 — dzięki temu zaokrąglenia rogów są prawdziwymi
łukami, a nie elipsami.

Usage: python scripts/thesis_figures/fig_ch4_project_structure.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch04"

INK, INK2 = "#0b0b0b", "#52514e"          # tekst główny / opisy po prawej

DPI = 300
PX = 1.0 / DPI                             # 1 piksel oryginału w calach

# --- metryka odtworzona z opublikowanego renderu [px] ---------------------
PAD_L, PAD_R = 62 * PX, 61 * PX            # marginesy wewnętrzne pudełka
TITLE_BASE = 58 * PX                       # linia bazowa tytułu od górnej krawędzi
ROW1_BASE = 140.5 * PX                     # linia bazowa 1. wiersza od górnej krawędzi
ROW_PITCH = 61.5 * PX                      # odstęp między wierszami
PAD_B = 58.5 * PX                          # margines pod ostatnim wierszem
BOX_GAP = 77.5 * PX                        # odstęp pionowy między pudełkami toru
ROUND = 11 * PX                            # promień zaokrąglenia rogów
LW_BOX = 3 * PX * 72                       # grubość ramki [pt]
LW_ARROW = 5 * PX * 72                     # grubość strzałki [pt]
HEAD = 23 * PX * 72 / 0.4                  # mutation_scale dla grotu „-|>"

FS_TITLE, FS_ROW = 8.6, 7.3

MARGIN_T, MARGIN_B = 42 * PX, 56 * PX
MARGIN_L, MARGIN_R = 49.5 * PX, 47.5 * PX

SPINE_W = 1188 * PX                        # szerokość pudełek toru
LIB_W = 799 * PX                           # szerokość pudełka biblioteki
COL_GAP = 76 * PX                          # odstęp między kolumnami

# --- treść ----------------------------------------------------------------
SPINE = [
    ("KONFIGURACJA", [
        ("experiments/", "konfiguracje eksperymentów"),
        ("sweeps/", "konfiguracje strojeń hiperparametrów"),
        ("config.py", "moduł konfiguracyjny"),
    ]),
    ("TRENING", [
        ("train.py", "skrypt treningowy"),
        ("training/", "implementacja procesu treningu"),
    ]),
    ("ZAPISANE PLIKI", [
        ("checkpoints/", "zapisane wagi modeli"),
        ("wandb/", "zapisane logi treningów"),
    ]),
    ("WYKORZYSTANIE MODELI", [
        ("evaluate.py /", "ewaluacja: pojedyncza /"),
        ("evaluate_all.py", "zbiorcza"),
        ("test_model_interactive.ipynb", "testowanie modeli na przykładach"),
    ]),
]

LIBRARY = ("BIBLIOTEKA PROJEKTU", [
    ("models/", "implementacje modeli"),
    ("datasets/", "implementacje zbiorów danych"),
    ("utils/", "metryki, logowanie"),
    ("tests/", "testy automatyczne"),
    ("scripts/", "skrypty"),
])


def box_height(rows):
    return ROW1_BASE + (len(rows) - 1) * ROW_PITCH + PAD_B


def draw_box(ax, x0, y_top, w, title, rows):
    """Pudełko kotwiczone górną krawędzią; y rośnie w górę."""
    h = box_height(rows)
    ax.add_patch(FancyBboxPatch(
        (x0, y_top - h), w, h,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor="white", edgecolor=INK, lw=LW_BOX, zorder=3))
    ax.text(x0 + PAD_L, y_top - TITLE_BASE, title, ha="left", va="baseline",
            fontsize=FS_TITLE, fontweight="bold", color=INK, zorder=4)
    for i, (path, desc) in enumerate(rows):
        y = y_top - ROW1_BASE - i * ROW_PITCH
        ax.text(x0 + PAD_L, y, path, ha="left", va="baseline",
                fontsize=FS_ROW, family="monospace", color=INK, zorder=4)
        ax.text(x0 + w - PAD_R, y, desc, ha="right", va="baseline",
                fontsize=FS_ROW, color=INK2, zorder=4)
    return h


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=HEAD,
                                 lw=LW_ARROW, color=INK, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", zorder=2))


def elbow(ax, pts):
    """Łamana o ostrych narożnikach zakończona grotem."""
    path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=HEAD,
                                 lw=LW_ARROW, color=INK, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", zorder=2))


def render(out_dir):
    heights = [box_height(rows) for _t, rows in SPINE]
    lib_h = box_height(LIBRARY[1])

    spine_h = sum(heights) + BOX_GAP * (len(heights) - 1)
    # biblioteka zaczyna się na wysokości drugiego pudełka toru
    lib_top_drop = heights[0] + BOX_GAP
    content_h = max(spine_h, lib_top_drop + lib_h)
    W = MARGIN_L + SPINE_W + COL_GAP + LIB_W + MARGIN_R
    H = MARGIN_T + content_h + MARGIN_B

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    x_spine = MARGIN_L
    x_lib = MARGIN_L + SPINE_W + COL_GAP
    top = H - MARGIN_T

    tops, bots = [], []
    y = top
    for (title, rows), h in zip(SPINE, heights):
        draw_box(ax, x_spine, y, SPINE_W, title, rows)
        tops.append(y)
        bots.append(y - h)
        y -= h + BOX_GAP

    lib_top = top - lib_top_drop
    draw_box(ax, x_lib, lib_top, LIB_W, *LIBRARY)
    lib_bot = lib_top - lib_h

    # strzałki toru: 5 px pod pudełkiem, grot 11 px nad następnym
    for i in range(len(SPINE) - 1):
        arrow(ax, (x_spine + SPINE_W / 2, bots[i] - 5 * PX),
                  (x_spine + SPINE_W / 2, tops[i + 1] + 11 * PX))

    # biblioteka → trening (w pionową oś pudełka treningu)
    y_tren = (tops[1] + bots[1]) / 2
    arrow(ax, (x_lib - 5 * PX, y_tren), (x_spine + SPINE_W + 11 * PX, y_tren))

    # biblioteka → wykorzystanie modeli (kolano)
    y_use = (tops[3] + bots[3]) / 2
    elbow(ax, [(x_lib + LIB_W / 2, lib_bot),
               (x_lib + LIB_W / 2, y_use),
               (x_spine + SPINE_W + 11 * PX, y_use)])

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rys_ch4_project_structure.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"zapisano {out}  ({round(W * DPI)}×{round(H * DPI)} px)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    render(p.parse_args().out_dir)


if __name__ == "__main__":
    main()
