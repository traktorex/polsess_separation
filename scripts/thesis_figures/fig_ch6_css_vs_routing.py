"""Rysunek (ch6, money figure) — CSS sliding-window separation vs region-routed
separation with anchor-based identity (the ch6 leitmotif L1 contrast).

Fully schematic (invented illustrative timeline, no data). Two panels sharing
one visual vocabulary: the same ~60 s conversation strip, speaker colors and
overlap color as everywhere in ch6 (A blue #2a78d6, B aqua #1baf7a, overlap
orange #eb6834).

Panel A (CSS): separation runs everywhere via overlapping sliding windows; PIT
makes the channel->speaker assignment arbitrary per window, so adjacent windows
must be re-aligned on their shared frames (correlation stitching) to keep each
speaker on one output channel.

Panel B (ours): separation runs only on detected overlap regions; speaker
identity is decided against recording-level ECAPA anchors built from solo
audio — there are no windows to align.

Renders: rys_ch6_css_vs_routing (both panels), rys_ch6_css_window (A alone),
rys_ch6_routing_anchor (B alone).

Usage: python scripts/thesis_figures/fig_ch6_css_vs_routing.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch06"

INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SPK_A, SPK_B, OVERLAP = "#2a78d6", "#1baf7a", "#eb6834"

# Illustrative conversation: (start, end, kind) with kind in a|b|ab|sil
STRIP = [
    (0, 8, "a"), (8, 9, "sil"), (9, 17, "b"), (17, 22, "ab"), (22, 30, "a"),
    (30, 31, "sil"), (31, 40, "b"), (40, 44, "ab"), (44, 52, "a"), (52, 60, "b"),
]
KIND_COLOR = {"a": SPK_A, "b": SPK_B, "ab": OVERLAP}
GAP = 0.25  # white gap between adjacent strip segments (time units)

# CSS windows (start, end) and their arbitrary per-window channel assignment:
# (speaker carried on channel 1, speaker carried on channel 2)
WINDOWS = [(0, 16), (12, 28), (24, 40), (36, 52), (48, 60)]
ASSIGN = [("a", "b"), ("b", "a"), ("a", "b"), ("a", "b"), ("b", "a")]
ICON_INSET = 3.0  # per-window channel icons: inset from window edges


def _bar(ax, x0, x1, y0, y1, color, alpha=1.0, z=2, ec="none", lw=0.0):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                           edgecolor=ec, lw=lw, alpha=alpha, zorder=z))


def draw_strip(ax, y0, y1):
    for s, e, kind in STRIP:
        if kind == "sil":
            ax.plot([s + GAP / 2, e - GAP / 2], [y0 + 0.02, y0 + 0.02],
                    color=AXIS, lw=1.0, zorder=2)
            continue
        _bar(ax, s + GAP / 2, e - GAP / 2, y0, y1, KIND_COLOR[kind])


def draw_strip_legend(ax, y):
    items = [(SPK_A, "mówca A"), (SPK_B, "mówca B"),
             (OVERLAP, "nakładanie (A+B)"), (None, "cisza")]
    x = 0.0
    for col, lab in items:
        if col is None:
            ax.plot([x, x + 1.6], [y + 0.14, y + 0.14], color=AXIS, lw=1.0)
        else:
            _bar(ax, x, x + 1.6, y, y + 0.32, col)
        ax.text(x + 2.1, y + 0.14, lab, ha="left", va="center", fontsize=7.6,
                color=INK2)
        x += 2.1 + len(lab) * 0.92 + 3.2
    return y


def draw_output_streams(ax, y_a0, y_a1, y_b0, y_b1):
    """Per-speaker assembled output rows: filled where the speaker talks;
    lighter shade where the speech was recovered from an overlap."""
    for s, e, kind in STRIP:
        if kind in ("a", "ab"):
            _bar(ax, s + GAP / 2, e - GAP / 2, y_a0, y_a1, SPK_A,
                 alpha=1.0 if kind == "a" else 0.5)
        if kind in ("b", "ab"):
            _bar(ax, s + GAP / 2, e - GAP / 2, y_b0, y_b1, SPK_B,
                 alpha=1.0 if kind == "b" else 0.5)
    ax.text(-1.2, (y_a0 + y_a1) / 2, "A", ha="right", va="center",
            fontsize=8, color=INK2)
    ax.text(-1.2, (y_b0 + y_b1) / 2, "B", ha="right", va="center",
            fontsize=8, color=INK2)


def row_label(ax, y, text):
    ax.text(-3.4, y, text, ha="right", va="center", fontsize=8.6, color=INK)


def _icon_span(i):
    s, e = WINDOWS[i]
    return s + ICON_INSET, e - ICON_INSET


def panel_a(ax):
    ax.set_xlim(-17, 76)
    ax.set_ylim(0.7, 13.2)
    ax.set_axis_off()

    ax.text(0, 13.1, "a)  separacja ciągła (CSS) — okna przesuwne na całym nagraniu",
            ha="left", va="top", fontsize=9.2, color=INK, fontweight="bold")
    draw_strip_legend(ax, 11.85)

    row_label(ax, 10.6, "nagranie")
    draw_strip(ax, 10.2, 11.0)

    # shared-frame zones (windows overlap here) shaded behind the window row
    for i in range(len(WINDOWS) - 1):
        z0, z1 = WINDOWS[i + 1][0], WINDOWS[i][1]
        _bar(ax, z0, z1, 7.6, 9.35, GRID, alpha=0.55, z=1)

    # sliding windows, staggered on two sub-rows so their overlap is visible
    row_label(ax, 8.45, "okna separacji")
    for i, (s, e) in enumerate(WINDOWS):
        y0, y1 = (8.6, 9.2) if i % 2 == 0 else (7.75, 8.35)
        ax.add_patch(FancyBboxPatch((s + 0.25, y0), e - s - 0.5, y1 - y0,
                                    boxstyle="round,pad=0.02",
                                    facecolor=OVERLAP, alpha=0.14,
                                    edgecolor=OVERLAP, lw=1.1, zorder=2))
        ax.text((s + e) / 2, (y0 + y1) / 2, f"okno {i + 1}", ha="center",
                va="center", fontsize=7.6, color=INK2)
        # window -> its channel icon
        x0, x1 = _icon_span(i)
        ax.plot([(s + e) / 2, (x0 + x1) / 2], [y0 - 0.06, 6.85], color=AXIS,
                lw=0.7, zorder=1)
    ax.text(63.5, 8.45, "separacja\nw każdym oknie", ha="left", va="center",
            fontsize=7.8, color=INK2)

    # per-window output-channel icons (two bars: channel 1 above, channel 2
    # below); color = the speaker that channel happens to carry in that window
    row_label(ax, 6.15, "kanały wyjściowe\n(na okno)")
    ch_y = {1: (6.25, 6.7), 2: (5.6, 6.05)}
    for i, (c1, c2) in enumerate(ASSIGN):
        x0, x1 = _icon_span(i)
        _bar(ax, x0, x1, *ch_y[1], KIND_COLOR[c1])
        _bar(ax, x0, x1, *ch_y[2], KIND_COLOR[c2])
    ax.text(_icon_span(0)[0], ch_y[1][1] + 0.16, "kanał 1", ha="left",
            va="bottom", fontsize=7.2, color=MUTED)
    ax.text(_icon_span(0)[0], ch_y[2][0] - 0.16, "kanał 2", ha="left",
            va="top", fontsize=7.2, color=MUTED)

    # stitching connectors between adjacent icons; crossed when the arbitrary
    # assignment flips between windows
    y_hi, y_lo = sum(ch_y[1]) / 2, sum(ch_y[2]) / 2
    for i in range(len(WINDOWS) - 1):
        xr = _icon_span(i)[1]
        xl = _icon_span(i + 1)[0]
        swap = ASSIGN[i] != ASSIGN[i + 1]
        if swap:
            ax.plot([xr, xl], [y_hi, y_lo], color=INK, lw=1.1, zorder=4)
            ax.plot([xr, xl], [y_lo, y_hi], color=INK, lw=1.1, zorder=4)
        else:
            ax.plot([xr, xl], [y_hi, y_hi], color=INK, lw=1.1, zorder=4)
            ax.plot([xr, xl], [y_lo, y_lo], color=INK, lw=1.1, zorder=4)
    ax.annotate("sklejanie: dopasowanie kanałów na wspólnych ramkach\n"
                "(skrzyżowanie = wykryta zamiana kanałów)",
                (_icon_span(0)[1] + 1.0, y_lo - 0.15), (17, 4.45), ha="left",
                va="center", fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(63.5, 6.15, "przypisanie kanałów\nlosowe w każdym\noknie (PIT)",
            ha="left", va="center", fontsize=7.8, color=INK2)

    # assembled per-speaker streams
    row_label(ax, 2.7, "strumienie\npo sklejeniu")
    draw_output_streams(ax, 2.9, 3.3, 2.1, 2.5)
    ax.text(30, 1.35, "jaśniejszy odcień — mowa odzyskana z nakładania",
            ha="center", va="center", fontsize=7.4, color=MUTED)


def panel_b(ax, standalone=False):
    ax.set_xlim(-17, 76)
    ax.set_ylim(0.4, 13.2)
    ax.set_axis_off()

    ax.text(0, 13.1,
            "b)  separacja wybiórcza (potok pracy) — tylko wykryte nakładania",
            ha="left", va="top", fontsize=9.2, color=INK, fontweight="bold")
    if standalone:
        draw_strip_legend(ax, 11.85)

    row_label(ax, 10.6, "nagranie")
    draw_strip(ax, 10.2, 11.0)

    # anchors built from solo audio
    anch = {"a": (12.0, SPK_A, "kotwica A (ECAPA)"),
            "b": (46.0, SPK_B, "kotwica B (ECAPA)")}
    ay0, ay1 = 7.6, 8.3
    for spk, (xc, col, lab) in anch.items():
        ax.add_patch(FancyBboxPatch((xc - 8.6, ay0), 17.2, ay1 - ay0,
                                    boxstyle="round,pad=0.02", facecolor="white",
                                    edgecolor=col, lw=1.3, zorder=3))
        ax.text(xc, (ay0 + ay1) / 2, lab, ha="center", va="center",
                fontsize=7.6, color=INK)
    row_label(ax, (ay0 + ay1) / 2, "kotwice mówców")
    # collector lines: each solo segment -> its speaker's anchor top edge,
    # attach points spread left-to-right to limit crossings
    solos = {"a": [seg for seg in STRIP if seg[2] == "a"],
             "b": [seg for seg in STRIP if seg[2] == "b"]}
    for spk, segs in solos.items():
        xc, col, _ = anch[spk]
        attach = [xc - 4.5, xc, xc + 4.5]
        for (s, e, _k), xa in zip(segs, attach):
            ax.plot([(s + e) / 2, xa], [10.15, ay1 + 0.04], color=col, lw=0.9,
                    alpha=0.45, zorder=1)
    ax.text(63.5, 7.95, "osadzenia z fragmentów\nsolo całego nagrania",
            ha="left", va="center", fontsize=7.8, color=INK2)

    # separator windows only on the overlap regions
    row_label(ax, 5.85, "separacja\n(tylko nakładania)")
    for (s, e) in [(17, 22), (40, 44)]:
        pad = 1.5  # context expansion beyond the overlap
        ax.add_patch(FancyBboxPatch((s - pad, 5.5), (e - s) + 2 * pad, 0.72,
                                    boxstyle="round,pad=0.02",
                                    facecolor=OVERLAP, alpha=0.14,
                                    edgecolor=OVERLAP, lw=1.1, zorder=2))
        _bar(ax, s, e, 5.62, 6.1, OVERLAP, z=3)
    ax.annotate("rozszerzenie okna\ndo długości 4 s", (15.6, 5.85), (1.5, 4.5),
                ha="center", va="center", fontsize=7.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(63.5, 5.85, "poza nakładaniami\nseparator nie działa",
            ha="left", va="center", fontsize=7.8, color=INK2)

    # anonymous separated streams under each window + assignment result caps
    caps = {(17, 22): (SPK_B, SPK_A), (40, 44): (SPK_A, SPK_B)}
    for (s, e), (cap1, cap2) in caps.items():
        for k, (yy, cap) in enumerate(zip((4.35, 3.8), (cap1, cap2))):
            _bar(ax, s, e, yy, yy + 0.4, "#b9b8b0", z=3)
            _bar(ax, e - 0.7, e, yy, yy + 0.4, cap, z=4)
            ax.text(s - 0.6, yy + 0.2, f"s{k + 1}?", ha="right", va="center",
                    fontsize=7.2, color=INK2)
        # one neutral arrow from this stream pair to each anchor
        xm = (s + e) / 2
        for spk in ("a", "b"):
            xc = anch[spk][0]
            ax.add_patch(FancyArrowPatch((xm, 4.9), (xc, ay0 - 0.1),
                                         arrowstyle="-|>", mutation_scale=7,
                                         color=INK2, lw=0.9, alpha=0.75,
                                         connectionstyle="arc3,rad=-0.12",
                                         zorder=4))
    ax.text(30.5, 3.15, "dopasowanie anonimowych strumieni do kotwic "
            "(podobieństwo cosinusowe)", ha="center", va="center",
            fontsize=7.8, color=INK)
    ax.text(63.5, 4.15, "kolorowy koniec —\nprzypisany mówca",
            ha="left", va="center", fontsize=7.4, color=MUTED)

    # assembled per-speaker streams
    row_label(ax, 2.25, "strumienie\nwyjściowe")
    draw_output_streams(ax, 2.45, 2.85, 1.65, 2.05)
    ax.text(30, 0.85, "brak okien do sklejania — tożsamość względem kotwic "
            "całego nagrania", ha="center", va="center", fontsize=7.8,
            color=INK)


def _style():
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "font.family": "sans-serif",
    })


def _save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out_dir}/{stem}.{{png,pdf}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    # combined two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 7.6))
    panel_a(ax1)
    panel_b(ax2)
    fig.tight_layout(h_pad=1.0)
    _save(fig, args.out_dir, "rys_ch6_css_vs_routing")

    # split variants
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    panel_a(ax)
    fig.tight_layout()
    _save(fig, args.out_dir, "rys_ch6_css_window")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    panel_b(ax, standalone=True)
    fig.tight_layout()
    _save(fig, args.out_dir, "rys_ch6_routing_anchor")


if __name__ == "__main__":
    main()
