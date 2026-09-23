"""Figures for the thesis-defence slides (thesis/my-writing/Prezentacja.md).

Builds, into ``thesis/my-writing/figures/prezentacja/``:

  prez_02_separacja_schemat.png   slide 2 – mixture → separator → two speaker signals (real pipeline
                                             output for one overlap window of the demo recording)
  prez_02b_asr_bez_separacji.png  slide 2B – mixture → ASR → one transcript missing the quieter speaker
  prez_02c_asr_z_separacja.png    slide 2B – mixture → separator → two signals → ASR → two transcripts
  prez_03_polsess_<VARIANT>.png   slide 3 – PolSESS layer stack (like Rys. 4.x) for one indoor test
                                             mix; one file per MM-IPC variant, absent layers drawn as
                                             hollow outlines and the top "Miks" restacked from the
                                             active layers (voices innermost, environment in greys)
  prez_04_potok.png               slide 4 – Rys. 6.1 (the ASR pipeline) re-laid left-to-right,
                                             "BWE i VAD" under "Separator"
  prez_06a_separacja_warstwy.png  slide 6 – Tab. 6.2 (no separator vs full pipeline, per stratum)
  prez_06b_nakladanie.png         slide 6 – Tab. 6.4 (overlapped vs non-overlapped utterances)
  prez_06c_sam_whisperx.png       slide 6 – Tab. 6.3 (WhisperX alone on the raw recording vs full
                                             pipeline, ORC-WER per stratum)
  prez_07a_slowa_<frag>[_luki].png slide 7 – reference words coloured by fate, one file per example in
                                             EXAMPLES, with lost words struck (default) or left as slots
  prez_07{b..e}_*.png             slide 7 – other layouts of the ccfbb9db window: (b) time axis,
                                             (c) mechanism diagram, (d) shorter example, (e) miss-rate /
                                             insertion bars (ch. 6 fn. 5)
  prez_08a_modele_dane.png        slide 8 – Tab. 6.5 subset: separators by training corpus
  prez_08b_pasmo.png              slide 8 – Tab. 6.7: 8 kHz MossFormer2 vs 16 kHz TIGER

Numbers are copied from the hand-in text of chapter 6. Audio and JSON come from the frozen
v41_merge / v41_merge_nosep arms of the CLARIN_fragments eval tree and from PolSESS_C_final_128_v2/test.

Usage:
    venv/bin/python scripts/thesis_figures/fig_prezentacja.py [--frag ccfbb9db__seg00] [--row 10868]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Patch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "thesis" / "my-writing" / "figures" / "prezentacja"
EVAL_ROOT = Path.home() / "datasets" / "eval" / "clarin_fragments"
POLSESS_TEST = Path.home() / "datasets" / "PolSESS_C_final_128_v2" / "test"

# ---- palette (thesis figures + webapp speaker colours) ---------------------------------------
INK, MUTED, GRID, AXIS = "#1f2328", "#59636e", "#e1e0d9", "#c3c2b7"
SPK_A, SPK_B = "#1f6feb", "#e36209"          # webapp --spkA / --spkB
ERR = "#cf222e"                              # inserted / lost words
AMBER = "#9a6700"                            # substituted words
NOSEP, FULL = "#9aa3ae", "#1f6feb"           # "bez separacji" vs "pełny potok"
NOISY_TRAIN, CLEAN_TRAIN = "#1f6feb", "#d29922"
BAND = {"0–4 kHz + BWE": "#4a90c4", "0–8 kHz": "#1f5c8b"}
GHOST_WAVE, GHOST_TEXT, GHOST_FACE, GHOST_EDGE = "#c9c9c9", "#b4b4b4", "#fbfbfb", "#e6e6e6"
WAVE, FACE, EDGE = "#111111", "#f1f1f1", "#cdcdcd"


def pl(x: float, nd: int = 1) -> str:
    """Polish decimal comma."""
    return f"{x:.{nd}f}".replace(".", ",")


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 12, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.titlesize": 13, "axes.titleweight": "normal",
    })


# ---- colour-vision-deficiency sanity check (the skill's validator needs node, absent here) ----
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _oklab(lin_rgb):
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(m1 @ lin_rgb)


_CVD = {  # Machado et al. 2009, severity 1.0
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def cvd_check(name: str, hexes: list[str]) -> None:
    for a, b in zip(hexes, hexes[1:]):
        la, lb = _srgb_to_linear(_hex_to_rgb(a)), _srgb_to_linear(_hex_to_rgb(b))
        normal = np.linalg.norm(_oklab(la) - _oklab(lb)) * 100
        worst = min(np.linalg.norm(_oklab(np.clip(m @ la, 0, 1)) - _oklab(np.clip(m @ lb, 0, 1))) * 100
                    for m in _CVD.values())
        flag = "ok" if (worst >= 8 and normal >= 15) else "CHECK"
        print(f"  [{name}] {a} vs {b}: normal ΔE={normal:.1f}, worst-CVD ΔE={worst:.1f} -> {flag}")


# ---- data helpers ---------------------------------------------------------------------------
def envelope(x: np.ndarray, sr: int, bin_ms: float = 10.0):
    n = int(sr * bin_ms / 1000)
    m = len(x) // n
    env = np.abs(x[: m * n]).reshape(m, n).max(axis=1)
    t = (np.arange(m) + 0.5) * n / sr
    return t, env


def load_streams(frag: str, arm: str = "v41_merge") -> dict:
    d = EVAL_ROOT / frag / "sweep" / arm
    mix, sr = sf.read(EVAL_ROOT / frag / f"{frag}.wav")
    a, _ = sf.read(d / "stream_A.wav")
    b, _ = sf.read(d / "stream_B.wav")
    return {"sr": sr, "mix": mix, "A": a, "B": b}


def draw_wave(ax, t, env, color):
    ax.fill_between(t, -env, env, color=color, linewidth=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


# ---- slide 2: mixture → separator → two signals ----------------------------------------------
def fig_02(data: dict, t0: float, t1: float) -> None:
    sr = data["sr"]
    sl = slice(int(t0 * sr), int(t1 * sr))
    tm, em = envelope(data["mix"][sl], sr)
    ta, ea = envelope(data["A"][sl], sr)
    tb, eb = envelope(data["B"][sl], sr)
    peak = max(em.max(), ea.max(), eb.max())
    em, ea, eb = em / peak, ea / peak, eb / peak

    fig = plt.figure(figsize=(12, 4.2))
    ax_mix = fig.add_axes([0.03, 0.30, 0.34, 0.42])
    ax_box = fig.add_axes([0.41, 0.15, 0.20, 0.70]); ax_box.axis("off")
    ax_a = fig.add_axes([0.65, 0.55, 0.33, 0.30])
    ax_b = fig.add_axes([0.65, 0.15, 0.33, 0.30])

    draw_wave(ax_mix, tm, em, INK)
    draw_wave(ax_a, ta, ea, SPK_A)
    draw_wave(ax_b, tb, eb, SPK_B)
    for ax in (ax_mix, ax_a, ax_b):
        ax.set_xlim(0, t1 - t0)
        ax.set_xticks([])

    ax_mix.set_title("nagranie wejściowe", color=INK, fontsize=13, pad=8)
    ax_a.set_title("sygnał mówcy A", color=SPK_A, fontsize=13, pad=6, loc="left")
    ax_b.set_title("sygnał mówcy B", color=SPK_B, fontsize=13, pad=6, loc="left")

    box = FancyBboxPatch((0.08, 0.30), 0.84, 0.40, boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=1.5, edgecolor=INK, facecolor="#f6f8fa", transform=ax_box.transAxes)
    ax_box.add_patch(box)
    ax_box.text(0.5, 0.50, "model\nseparacji", ha="center", va="center", fontsize=15, color=INK,
                transform=ax_box.transAxes)

    kw = dict(arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=INK)
    fig.add_artist(ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_mix.transAxes,
                                   xyB=(0.06, 0.5), coordsB=ax_box.transAxes, **kw))
    fig.add_artist(ConnectionPatch(xyA=(0.94, 0.5), coordsA=ax_box.transAxes,
                                   xyB=(0.0, 0.5), coordsB=ax_a.transAxes, **kw))
    fig.add_artist(ConnectionPatch(xyA=(0.94, 0.5), coordsA=ax_box.transAxes,
                                   xyB=(0.0, 0.5), coordsB=ax_b.transAxes, **kw))
    fig.savefig(OUT_DIR / "prez_02_separacja_schemat.png", dpi=220)
    plt.close(fig)


# ---- slide 2B/2C: ASR alone on overlapped speech vs. separator first ----------------------------------
# Same clip as the slide-7 example (b9bd9620__seg00, 0–4 s). Texts are real outputs: WhisperX run on the
# mixture (transcript_mixture.json of the v41_merge arm) and the two tracks of the full pipeline
# (transcript_A/B.json); the waveforms are the mixture and the pipeline's assembled tracks.
SAID_2B = [("A: „Jakby to działa, nie? Kurczę,", "A", False), ("a jeszcze jakby była jakaś taka…”", "A", True),
           ("B: „Ten poradnik działa. Te zasady działają.”", "B", False)]      # (line, speaker, continuation)
ASR_MIX_2B = "„Ten poradnik działa.\nTe zasady działają.”"
ASR_A_2B = "„Jakby to działa, nie?\nKurczę, a jeszcze jakby\nbyła jakaś taka…”"
ASR_B_2B = "„Ten poradnik działa.\nTe zasady działają.”"
ARROW = dict(arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=INK)


def _model_box(fig, rect, label, fs=15):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.08, 0.30), 0.84, 0.40, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=INK, facecolor="#f6f8fa", transform=ax.transAxes))
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=fs, color=INK, transform=ax.transAxes)
    return ax


def _fig_w(fig, text, fs, **kw) -> float:
    """Width of ``text`` in figure fraction."""
    t = fig.text(0, 0, text, fontsize=fs, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    t.remove()
    return w


def _two_tone(fig, x, y, prefix, letter, color, fs):
    """``prefix`` in ink followed by the speaker letter in its colour."""
    fig.text(x, y, prefix, fontsize=fs, color=INK, va="center")
    fig.text(x + _fig_w(fig, prefix, fs), y, letter, fontsize=fs, color=color, va="center")


def _mix_with_said(fig, data, t0, t1, rect, x_text):
    sr = data["sr"]
    sl = slice(int(t0 * sr), int(t1 * sr))
    tm, em = envelope(data["mix"][sl], sr)
    ta, ea = envelope(data["A"][sl], sr)
    tb, eb = envelope(data["B"][sl], sr)
    peak = max(em.max(), ea.max(), eb.max())
    ax = fig.add_axes(rect)
    draw_wave(ax, tm, em / peak, INK)
    ax.set_xlim(0, t1 - t0); ax.set_xticks([])
    ax.set_title("nagranie wejściowe", color=INK, fontsize=13, pad=8)
    fig.text(x_text, rect[1] - 0.07, "wypowiedzi w nagraniu:", fontsize=10.5, color=MUTED, va="center")
    indent = _fig_w(fig, "A: ", 11.5)
    for k, (line, who, cont) in enumerate(SAID_2B):
        fig.text(x_text + (indent if cont else 0), rect[1] - 0.145 - 0.068 * k, line, fontsize=11.5,
                 color=WHO_COL[who], va="center")
    return ax, (ta, ea / peak), (tb, eb / peak)


def fig_02bc(data: dict, t0: float, t1: float) -> None:
    # (2B) mixture → ASR → one transcript that holds only the louder speaker
    fig = plt.figure(figsize=(12, 4.4))
    ax_mix, _, _ = _mix_with_said(fig, data, t0, t1, [0.03, 0.44, 0.30, 0.40], 0.03)
    ax_box = _model_box(fig, [0.40, 0.29, 0.18, 0.70], "model ASR")
    fig.text(0.64, 0.86, "transkrypcja", fontsize=13, color=INK, va="center")
    fig.text(0.64, 0.64, ASR_MIX_2B, fontsize=13, color=MUTED, va="center", linespacing=1.5)
    fig.add_artist(ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_mix.transAxes,
                                   xyB=(0.06, 0.5), coordsB=ax_box.transAxes, **ARROW))
    fig.add_artist(ConnectionPatch(xyA=(0.94, 0.5), coordsA=ax_box.transAxes,
                                   xyB=(0.63, 0.64), coordsB="figure fraction", **ARROW))
    fig.savefig(OUT_DIR / "prez_02b_asr_bez_separacji.png", dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    # (2C) mixture → separator → two signals → ASR each → two transcripts
    fig = plt.figure(figsize=(14, 4.4))
    ax_mix, (ta, ea), (tb, eb) = _mix_with_said(fig, data, t0, t1, [0.02, 0.44, 0.24, 0.40], 0.02)
    ax_sep = _model_box(fig, [0.29, 0.29, 0.12, 0.70], "model\nseparacji", fs=14)
    ax_a = fig.add_axes([0.45, 0.66, 0.17, 0.26]); draw_wave(ax_a, ta, ea, SPK_A)
    ax_b = fig.add_axes([0.45, 0.26, 0.17, 0.26]); draw_wave(ax_b, tb, eb, SPK_B)
    for ax, letter, col, y_title in ((ax_a, "A", SPK_A, 0.955), (ax_b, "B", SPK_B, 0.555)):
        ax.set_xlim(0, t1 - t0); ax.set_xticks([])
        _two_tone(fig, 0.45, y_title, "sygnał mówcy ", letter, col, 12)
    ax_asr_a = _model_box(fig, [0.65, 0.66, 0.09, 0.26], "model\nASR", fs=12)
    ax_asr_b = _model_box(fig, [0.65, 0.26, 0.09, 0.26], "model\nASR", fs=12)
    _two_tone(fig, 0.77, 0.955, "transkrypcja mówcy ", "A", SPK_A, 12)
    fig.text(0.77, 0.79, ASR_A_2B, fontsize=12, color=SPK_A, va="center", linespacing=1.45)
    _two_tone(fig, 0.77, 0.555, "transkrypcja mówcy ", "B", SPK_B, 12)
    fig.text(0.77, 0.39, ASR_B_2B, fontsize=12, color=SPK_B, va="center", linespacing=1.45)
    fig.add_artist(ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_mix.transAxes,
                                   xyB=(0.06, 0.5), coordsB=ax_sep.transAxes, **ARROW))
    for ax_w, ax_m, y in ((ax_a, ax_asr_a, 0.79), (ax_b, ax_asr_b, 0.39)):
        fig.add_artist(ConnectionPatch(xyA=(0.94, 0.5), coordsA=ax_sep.transAxes,
                                       xyB=(0.0, 0.5), coordsB=ax_w.transAxes, **ARROW))
        fig.add_artist(ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_w.transAxes,
                                       xyB=(0.06, 0.5), coordsB=ax_m.transAxes, **ARROW))
        fig.add_artist(ConnectionPatch(xyA=(0.94, 0.5), coordsA=ax_m.transAxes,
                                       xyB=(0.76, y), coordsB="figure fraction", **ARROW))
    fig.savefig(OUT_DIR / "prez_02c_asr_z_separacja.png", dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---- slide 3: PolSESS layers per MM-IPC variant, one colour per layer -----------------------
# Layer arithmetic follows datasets/polsess_dataset.py: speakers always; scene with S; event with E;
# speaker reverb tails with R; event reverb only when both E and R are present.
LAYERS = [  # (display name, subtitle, csv column, folder, letters required, colour)
    ("Mówca 1", "", "speaker1File", "clean", "", "#1f6feb"),
    ("Pogłos", "mówcy 1", "reverbForSpeaker1", "sp1_reverb", "R", "#7fadf3"),
    ("Mówca 2", "", "speaker2File", "clean", "", "#e36209"),
    ("Pogłos", "mówcy 2", "reverbForSpeaker2", "sp2_reverb", "R", "#f0a565"),
    ("Zdarzenie", "dźwiękowe", "eventFile", "event", "E", "#4f5865"),      # environment in neutrals ("grafit")
    ("Pogłos", "zdarzenia", "reverbForEvent", "ev_reverb", "ER", "#b2b9c3"),
    ("Tło sceny", "", "sceneFile", "scene", "S", "#727c88"),
]
VARIANTS = ["SER", "SR", "ER", "R", "SE", "S", "E", "C"]
STACK = [0, 2, 1, 3, 4, 5, 6]                 # mix bands from the axis outwards: voices, their reverbs, environment
ABSENT_LINE, ABSENT_TEXT = "#c8c8c8", "#b4b4b4"   # layers missing from the variant: hollow outline, no fill


def load_polsess_layers(row_idx: int) -> list[np.ndarray]:
    csv = sorted(POLSESS_TEST.glob("*.csv"))[0]
    row = pd.read_csv(csv).iloc[row_idx]
    sigs = []
    for _, _, col, folder, _, _ in LAYERS:
        x, _ = sf.read(POLSESS_TEST / folder / row[col])
        sigs.append(np.asarray(x, dtype=float))
    n = min(len(s) for s in sigs)
    return [s[:n] for s in sigs]


def layer_envelopes(sigs: list[np.ndarray], bin_samples: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer max-abs envelopes (5 ms bins at 8 kHz) → (t, envs[layer, bin])."""
    m = len(sigs[0]) // bin_samples
    envs = np.stack([np.abs(x[: m * bin_samples]).reshape(m, bin_samples).max(axis=1) for x in sigs])
    return np.arange(m) / m, envs


def fig_03_variant(t: np.ndarray, envs: np.ndarray, variant: str, layer_peak: float, mix_peak: float) -> None:
    active = [all(ch in variant for ch in req) for *_, req, _ in LAYERS]

    W, P = 8.6, 0.66                              # waveform width, row pitch (inches)
    X0, Y0 = 3.3, 0.55                            # left edge / baseline of the lowest row
    AMP, AMP_MIX = 0.30, 0.62                     # half-heights: single layer / stacked mix
    fig = plt.figure(figsize=(12.5, 6.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.9); ax.axis("off")
    xs = X0 + t * W

    def label(yc, name, sub, color):
        ax.text(X0 - 0.55, yc + (0.13 if sub else 0.0), name, ha="right", va="center", fontsize=17,
                fontweight="bold", color=color)
        if sub:
            ax.text(X0 - 0.55, yc - 0.2, sub, ha="right", va="center", fontsize=12.5, color=color)
        ax.plot([X0 - 0.5, X0 - 0.05], [yc, yc], color=color, linewidth=0.9, linestyle=(0, (1.2, 2.2)))

    # individual layers: filled envelope in the layer colour; a layer absent from the variant is a hollow outline
    for i, (name, sub, *_, color) in enumerate(LAYERS):
        yc = Y0 + (len(LAYERS) - 1 - i) * P
        e = AMP * envs[i] / layer_peak
        if active[i]:
            ax.fill_between(xs, yc - e, yc + e, color=color, linewidth=0)
        else:
            ax.plot(xs, yc + e, color=ABSENT_LINE, linewidth=0.7)
            ax.plot(xs, yc - e, color=ABSENT_LINE, linewidth=0.7)
        label(yc, name, sub, color if active[i] else ABSENT_TEXT)

    # mix: layer envelopes stacked outwards from the axis in STACK order (voices innermost)
    yc = Y0 + len(LAYERS) * P + 0.55
    acc = np.zeros_like(t)
    for i in STACK:
        if not active[i]:
            continue
        e = AMP_MIX * envs[i] / mix_peak
        ax.fill_between(xs, yc - acc - e, yc - acc, color=LAYERS[i][-1], linewidth=0)
        ax.fill_between(xs, yc + acc, yc + acc + e, color=LAYERS[i][-1], linewidth=0)
        acc = acc + e
    label(yc, "Miks", "suma ścieżek", INK)
    ax.text(X0 + W / 2, 6.72, f"wariant MM-IPC: {variant}", ha="center", va="top", fontsize=17, color=INK)
    fig.savefig(OUT_DIR / f"prez_03_polsess_{variant}.png", dpi=200)
    plt.close(fig)


def fig_03(row_idx: int) -> None:
    t, envs = layer_envelopes(load_polsess_layers(row_idx))
    layer_peak = float(envs.max())
    mix_peak = float(envs.sum(axis=0).max())     # SER stack sets the scale for every variant
    for v in VARIANTS:
        fig_03_variant(t, envs, v, layer_peak, mix_peak)


# ---- slide 4: the ASR pipeline (Rys. 6.1) laid out left-to-right ---------------------------------
def fig_04() -> None:
    """Same topology, stages, symbols and captions as `render_v2_compact` in fig_ch6_pipeline.py,
    re-laid horizontally for a 16:9 slide with "BWE i VAD" under "Separator", reusing that script's
    box/arrow/glyph helpers and its cached real-recording envelopes. Speaker colours follow the
    slides (A blue, B orange)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fig_ch6_pipeline as P  # noqa: E402

    P.SPK_A, P.SPK_B, P.OVERLAP = SPK_A, SPK_B, ERR
    rec = P.load_recording()
    INK2 = P.INK2
    FT, FL, HB = 22.0, 16.5, 0.74
    H = 7.2
    fig, ax = P.canvas(24.0, H)
    yC = H / 2
    yT, yB = yC + 1.9, yC - 1.9
    GAP, SYM_W = 0.26, 1.45

    def bw_of(lines, pad=0.6):
        return max(P.text_w(fig, s, fs) for s, fs, _ in lines) + pad

    def stage(cx, cy, lines, w=None, h=HB):
        w = w or bw_of(lines)
        P.draw_box(ax, cx, cy, w, h, lines, lw=1.6)
        return w

    def caption(cx, y, lines, side):
        lh = P.line_h(FL)
        ys = [y + (len(lines) - 1 - i) * lh for i in range(len(lines))] if side == "above" \
            else [y - i * lh for i in range(len(lines))]
        for s, yy in zip(lines, ys):
            ax.text(cx, yy, s, ha="center", va="center", fontsize=FL, color=INK2, style="italic")

    def hlink(x0, x1, y, sym=None, sym_h=0.58, caps=(), side="above"):
        if sym is None:
            P.arrow(ax, (x0, y), (x1, y), color=INK2, lw=1.7, ms=17)
            return
        cx = (x0 + x1) / 2
        P.elbow(ax, [(x0, y), (cx - SYM_W / 2 - 0.05, y)], head=False, color=INK2, lw=1.7)
        sym(cx, y, SYM_W, sym_h)
        P.arrow(ax, (cx + SYM_W / 2 + 0.05, y), (x1, y), color=INK2, lw=1.7, ms=17)
        off = sym_h / 2 + 0.2
        caption(cx, y + off if side == "above" else y - off, caps, side)

    def vlink(x, y_top, y_bot, sym, sym_h=0.58, caps=()):
        """Downward link with the symbol on the arrow and captions to its right."""
        cy = (y_top + y_bot) / 2
        P.elbow(ax, [(x, y_top), (x, cy + sym_h / 2 + 0.05)], head=False, color=INK2, lw=1.7)
        sym(x, cy, SYM_W, sym_h)
        P.arrow(ax, (x, cy - sym_h / 2 - 0.05), (x, y_bot), color=INK2, lw=1.7, ms=17)
        lh = P.line_h(FL)
        for i, s in enumerate(caps):
            ax.text(x + SYM_W / 2 + 0.15, cy + (len(caps) - 1) * lh / 2 - i * lh, s, ha="left",
                    va="center", fontsize=FL, color=INK2, style="italic")

    def sym_raw(color):
        return lambda cx, cy, w, h: P.glyph_real(ax, cx, cy, w, h, rec["raw"], color=color, lw=0.6)

    def sym_enh(cx, cy, w, h):
        P.glyph_real(ax, cx, cy, w, h, rec["enh"], color=INK2, lw=0.6)

    def sym_frags(which):
        def f(cx, cy, w, h):
            if which == "mix":
                P.glyph_frags_real(ax, cx, cy, w, h, rec, ("mix",), (ERR,))
            else:
                keys = ("s1", "s2") if which == "sep" else ("g1", "g2")
                P.glyph_frags_real(ax, cx, cy, w, h, rec, keys, (SPK_A, SPK_B))
        return f

    def sym_streams(cx, cy, w, h):
        P.glyph_streams_real(ax, cx, cy, w, h, rec)

    L = lambda *ss: [(s, FT, INK) for s in ss]  # noqa: E731

    # input → split point → diarization; raw signal bypasses diarization on both branches
    x_in = 1.15
    sym_raw(INK)(x_in, yC, SYM_W, 0.62)
    caption(x_in, yC - 0.55, ("pełne nagranie", "mono, 16 kHz"), "below")
    x_spl = x_in + SYM_W / 2 + 0.45
    P.elbow(ax, [(x_in + SYM_W / 2 + 0.05, yC), (x_spl, yC)], head=False, color=INK2, lw=1.7)
    ax.plot([x_spl], [yC], marker="o", ms=5, color=INK2, zorder=4)
    w_dia = bw_of(L("Diaryzacja"))
    x_dia = x_spl + 0.5 + w_dia / 2
    P.arrow(ax, (x_spl, yC), (x_dia - w_dia / 2, yC), color=INK2, lw=1.7, ms=17)
    stage(x_dia, yC, L("Diaryzacja"))
    w_rou = bw_of(L("Routing"), pad=0.36)
    x_rou = x_dia + w_dia / 2 + 0.8 + w_rou / 2
    P.arrow(ax, (x_dia + w_dia / 2, yC), (x_rou - w_rou / 2, yC), color=INK2, lw=1.7, ms=17)
    for yy in (yT, yB):
        P.elbow(ax, [(x_spl, yC), (x_spl, yy), (x_rou - w_rou / 2, yy)], color=INK2, lw=1.7, r=0.16, ms=17)
    stage(x_rou, yC, L("Routing"), w=w_rou, h=yT - yB + HB + 0.35)
    xr = x_rou + w_rou / 2

    # first stage column: Separator (top) and Enhancement (bottom)
    w_sep, w_enh = bw_of(L("Separator")), bw_of(L("Enhancement"))
    x_col1 = xr + GAP + SYM_W + GAP
    w_col1 = max(w_sep, w_enh)
    x_sep = x_col1 + w_col1 / 2
    hlink(xr, x_col1, yT, sym=sym_frags("mix"), caps=("fragmenty", "z nakładaniem"))
    stage(x_sep, yT, L("Separator"), w=w_col1)
    hlink(xr, x_col1, yB, sym=sym_raw(INK2), caps=("pełne", "nagranie"), side="below")
    stage(x_sep, yB, L("Enhancement"), w=w_col1)

    w_rel = bw_of(L("Relabel", "i", "Assembly"), pad=0.36)
    # BWE i VAD under the separator; its output joins Relabel along the middle row
    vlink(x_sep, yT - HB / 2, yC + HB / 2, sym=sym_frags("sep"))
    stage(x_sep, yC, L("BWE i VAD"), w=w_col1)
    x_rel_l = x_col1 + w_col1 + GAP + SYM_W + GAP
    hlink(x_col1 + w_col1, x_rel_l, yC, sym=sym_frags("gat"))
    hlink(x_col1 + w_col1, x_rel_l, yB, sym=sym_enh)

    # relabel + assembly → streams → WhisperX → transcripts (below WhisperX)
    x_rel = x_rel_l + w_rel / 2
    stage(x_rel, yC, L("Relabel", "i", "Assembly"), w=w_rel, h=yT - yB + HB + 0.35)
    w_asr = bw_of(L("WhisperX"))
    x_asr_l = x_rel_l + w_rel + GAP + SYM_W + GAP + 0.6      # wider: the caption must clear the box
    x_asr = x_asr_l + w_asr / 2
    hlink(x_rel_l + w_rel, x_asr_l, yC, sym=sym_streams, sym_h=0.85,
          caps=("złożone sygnały", "obu mówców"), side="above")
    stage(x_asr, yC, L("WhisperX"))
    y_doc = yC - 1.35
    P.arrow(ax, (x_asr, yC - HB / 2), (x_asr, y_doc + 0.42), color=INK2, lw=1.7, ms=17)
    P.glyph_doc(ax, x_asr, y_doc, 1.0, 0.75)
    caption(x_asr, y_doc - 0.62, ("transkrypcje",), "below")

    W = x_asr + w_asr / 2 + 0.3
    ax.set_xlim(0, W)
    fig.set_size_inches(W, H)
    fig.savefig(OUT_DIR / "prez_04_potok.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---- slide 6: results bars -----------------------------------------------------------------------
def grouped_bars(fname: str, groups: list[str], series: list[tuple[str, str, list[float]]],
                 ylabel: str, figsize=(9, 4.6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    w = 0.34
    x = np.arange(len(groups))
    for k, (name, col, vals) in enumerate(series):
        off = (k - (n - 1) / 2) * (w + 0.03)
        bars = ax.bar(x + off, vals, width=w, color=col, label=name, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.35, pl(v), ha="center", va="bottom",
                    fontsize=11, color=INK)
    top = max(max(s[2]) for s in series)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, top * 1.15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=220)
    plt.close(fig)


def fig_06() -> None:
    strata = ["LOW", "MID", "HIGH", "całość"]
    grouped_bars("prez_06a_separacja_warstwy.png", strata,                       # Tab. 6.2
                 [("potok bez separacji", NOSEP, [22.7, 31.4, 29.3, 28.0]),
                  ("pełny potok", FULL, [16.5, 28.8, 26.0, 24.0])], "cpWER")
    grouped_bars("prez_06b_nakladanie.png",                                      # Tab. 6.4
                 ["wypowiedzi z nakładaniem", "wypowiedzi bez nakładania"],
                 [("potok bez separacji", NOSEP, [37.7, 25.0]),
                  ("pełny potok", FULL, [31.0, 24.6])], "cpWER", figsize=(7.5, 4.6))
    grouped_bars("prez_06c_sam_whisperx.png", strata,                             # Tab. 6.3 (ORC-WER:
                 [("sam WhisperX na nagraniu", NOSEP, [21.2, 29.5, 28.3, 26.5]),  # one transcript vs two)
                  ("pełny potok", FULL, [16.3, 27.3, 25.6, 23.3])], "ORC-WER")


# ---- slide 7: transcript example, five candidate visualisations ------------------------------------
# One overlap window of ccfbb9db__seg00 (TEST, MID; the demo recording): speaker A tells a long sentence and
# speaker B backchannels five times inside it. Texts and times were copied on 2026-09-09 from annotation.eaf
# (reference) and from transcript_{A,B}.json of the v41_merge_nosep / v41_merge arms. Without separation
# track A is complete but track B holds none of B's words and instead a copy of nine of A's words; the full
# pipeline keeps four of the five backchannels (one lost, one "Mhm." added). ``who`` = whose words a piece
# of text really is ("A", "B") or "ins" for a word with no counterpart in the reference.
EX_FRAG, EX_WINDOW = "ccfbb9db__seg00", (51.0, 59.0)
EX_OVERLAPS = [(51.44, 53.60), (54.56, 55.20), (55.76, 55.84), (56.80, 57.60), (58.16, 58.56)]  # routing.json
EX_CAPTION = "CLARIN_fragments · ccfbb9db__seg00 (TEST) · 51–59 s"
LOST_C = "#a8a8a8"
WHO_COL = {"A": SPK_A, "B": SPK_B, "ins": ERR}
LOST = "lost"

# Examples for the "words by fate" layout. ``ref`` holds the reference chunks of each speaker as
# (start, end, text); chunk boundaries are where the other speaker's chunks start, times come from the
# full-pipeline word timestamps. ``flow`` is the lane typeset left→right, the other lane is placed under it
# by time. Each system row lists, per track and in time order, what the track holds: ("A", i) = reference
# chunk i of speaker A in its own track (or a copy of it in the other track), ("A", i, LOST) = the chunk is
# missing from the output, ("ins", text) = words with no counterpart in the reference. Tracks are shown
# under the speaker they were assigned to by the cpWER permutation. Verified 2026-09-09 against
# annotation.eaf and transcript_{A,B}.json of the v41_merge / v41_merge_nosep arms.
EXAMPLES = {
    # A asks a question and starts a second sentence, B says two short sentences on top of it. Without
    # separation A's question and half of A's second sentence vanish and B's first sentence is copied
    # into A's track; the full pipeline is exact (both arms drop B's false start "ten").
    "b9bd9620": dict(
        caption="CLARIN_fragments · b9bd9620__seg00 (TEST) · 0–4 s", flow="A",
        ref={"A": [(0.19, 1.45, "Jakby to działa, nie?"), (1.51, 1.83, "Kurczę,"), (1.83, 2.29, "a jeszcze jakby"),
                   (2.29, 3.89, "była jakaś taka…")],
             "B": [(0.29, 0.81, "ten"), (0.96, 2.48, "Ten poradnik działa."), (2.60, 4.06, "Te zasady działają.")]},
        rows=[("bez separacji", {"A": [("A", 0, LOST), ("B", 1), ("A", 1, LOST), ("A", 2), ("A", 3, LOST)],
                                 "B": [("B", 0, LOST), ("B", 1), ("B", 2)]}),
              ("pełny potok", {"A": [("A", 0), ("A", 1), ("A", 2), ("A", 3)],
                               "B": [("B", 0, LOST), ("B", 1), ("B", 2)]})],
        # WhisperX run on the mixture alone (transcript_mixture.json): one transcript, B's sentences only
        mix=[("A", 0, LOST), ("B", 0, LOST), ("B", 1), ("A", 1, LOST), ("A", 2, LOST), ("B", 2), ("A", 3, LOST)]),
    # B talks continuously, A says one sentence over it. Without separation A's sentence loses its head
    # ("Więc tak, no") and tail ("rzeczywiście no." → "rzeczy.") and B's first clause is copied into A's
    # track; the full pipeline drops one word of A. Both arms drop B's "no,".
    "22cb9703": dict(
        caption="CLARIN_fragments · 22cb9703__seg00 (TEST) · 59–66 s", flow="B",
        ref={"A": [(58.89, 61.0, "Więc tak, no"), (61.91, 62.3, "to"), (62.3, 62.37, "już"),
                   (62.37, 63.4, "są takie zrobione"), (63.41, 64.45, "rzeczywiście no.")],
             "B": [(58.78, 60.5, "To jest naprawdę dość komfortowe"), (60.55, 61.93, "jak już jechać."),
                   (62.09, 62.5, "Ale"), (62.5, 63.0, "no,"), (63.09, 66.49, "on mówił i na razie mu to dobrze wyszło.")]},
        rows=[("bez separacji", {"A": [("B", 0), ("A", 0, LOST), ("A", 1), ("A", 2), ("A", 3), ("A", 4, LOST),
                                       ("ins", "rzeczy.")],
                                 "B": [("B", 0), ("B", 1), ("B", 2), ("B", 3, LOST), ("B", 4)]}),
              ("pełny potok", {"A": [("A", 0), ("A", 1), ("A", 2, LOST), ("A", 3), ("A", 4)],
                               "B": [("B", 0), ("B", 1), ("B", 2), ("B", 3, LOST), ("B", 4)]})]),
    # The demo recording: A tells a long sentence, B backchannels five times inside it. Without separation
    # track B holds none of B's words and a copy of nine of A's words; the full pipeline keeps four of the
    # five backchannels and adds one "Mhm.". Both arms drop the disfluent "z".
    "ccfbb9db": dict(
        caption=EX_CAPTION, flow="A",
        ref={"A": [(51.44, 52.04, "(…) no nie?"), (52.06, 52.46, "No i na przykład"), (52.50, 53.30, "jak się jedzie"),
                   (53.30, 53.76, "z"), (53.76, 54.56, "tutaj z tym,"), (54.62, 55.86, "że"),
                   (55.92, 56.70, "jak jest tutaj,"), (56.84, 57.36, "ze śniegiem"), (57.42, 58.48, "się ślizgasz, (…)")],
             "B": [(51.48, 52.37, "Aha, no no no."), (52.64, 53.62, "No."), (54.56, 55.07, "No."),
                   (56.72, 57.58, "Tak, tak…"), (57.95, 58.78, "No.")]},
        rows=[("bez separacji", {"A": [("A", i, LOST) if i == 3 else ("A", i) for i in range(9)],
                                 "B": [("B", 0, LOST), ("A", 1), ("A", 2), ("B", 1, LOST), ("B", 2, LOST), ("A", 7),
                                       ("B", 3, LOST), ("B", 4, LOST)]}),
              ("pełny potok", {"A": [("A", i, LOST) if i == 3 else ("A", i) for i in range(9)],
                               "B": [("B", 0), ("B", 1), ("ins", "Mhm."), ("B", 2, LOST), ("B", 3), ("B", 4)]})]),
}

# utterances with the arms' own timestamps, for the time-axis version: lane -> (start, end, text, who)
EX_TIMELINE = [
    ("referencja", {
        "A": [(46.36, 52.04, "(…) no nie?", "A"),
              (52.06, 65.57, "No i na przykład jak się jedzie z tutaj z tym, że jak jest tutaj, ze śniegiem "
                             "się ślizgasz, to jest tak, że (…)", "A")],
        "B": [(51.48, 52.37, "Aha, no no no.", "B"), (52.64, 53.62, "No.", "B"), (54.56, 55.07, "No.", "B"),
              (56.72, 57.58, "Tak, tak…", "B"), (57.95, 58.78, "No.", "B")]}),
    ("bez separacji", {
        "A": [(46.37, 52.05, "(…) no nie?", "A"),
              (52.07, 55.91, "No i na przykład jak się jedzie tutaj z tym, że…", "A"),
              (55.94, 58.16, "Jak jest tutaj, ze śniegiem się ślizgasz.", "A"),
              (58.50, 65.59, "To jest tak, że (…)", "A")],
        # one 51.52–57.41 segment in transcript_B.json; drawn as its two word groups (word timestamps)
        "B": [(51.52, 53.26, "No i na przykład jak się jedzie", "A"), (56.85, 57.41, "ze śniegiem,", "A")]}),
    ("pełny potok", {
        "A": [(46.36, 52.04, "(…) no nie?", "A"),
              (52.06, 65.58, "No i na przykład jak się jedzie tutaj z tym, że jak jest tutaj, ze śniegiem "
                             "się ślizgasz, to jest tak, że (…)", "A")],
        # "Tak, tak." is segment 53.28–57.39 in the JSON, but the aligner stretched its first word over 3.5 s
        # (word ends 56.81 / 57.39); drawn at the word ends, matching the reference utterance 56.72–57.58
        "B": [(51.58, 52.42, "Aha, no, no, no.", "B"), (52.48, 52.68, "No.", "B"), (52.72, 53.16, "Mhm.", "ins"),
              (56.45, 57.39, "Tak, tak.", "B"), (58.25, 58.49, "No.", "B")]}),
]


def _inch_canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, w); ax.set_ylim(0, h)
    return fig, ax


def _text_w(fig, ax, text: str, fs: float) -> float:
    t = ax.text(0, 0, text, fontsize=fs)
    w = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
    t.remove()
    return w


def _token(ax, x, y, text, who, fs, lost=False):
    ax.text(x, y, text, fontsize=fs, color=LOST_C if lost else WHO_COL[who], va="center", ha="left")


def _strike(ax, x0, x1, y):
    ax.plot([x0, x1], [y + 0.02, y + 0.02], color=LOST_C, linewidth=1.2, solid_capstyle="butt")


def _gap_mark(ax, x0, x1, y):
    ax.plot([x0 + 0.04, x1 - 0.04], [y - 0.02, y - 0.02], color=LOST_C, linewidth=1.3, linestyle=(0, (1, 2.2)))


def _fit(occupied, pref, w, gap, xmin):
    """x as close to ``pref`` as possible where [x, x+w] stays ``gap`` clear of every occupied interval;
    the nearest free interval on either side wins, the open interval on the right always fits."""
    free, prev = [], xmin - gap
    for a, b in sorted(occupied):
        free.append((prev + gap, a - gap)); prev = max(prev, b)
    free.append((prev + gap, float("inf")))
    best = None
    for lo, hi in free:
        if hi - lo < w:
            continue
        x = min(max(pref, lo), hi - w)
        if best is None or abs(x - pref) < abs(best - pref):
            best = x
    return best


def _layout_ref(fig, ax, ex, x0, fs, gap):
    """x of every reference chunk: the flow lane left→right, the other lane by time (interpolated inside the
    flow chunk that is being spoken when the chunk starts), shifted right on collision."""
    flow = ex["flow"]; other = "B" if flow == "A" else "A"
    widths = {(lane, i): _text_w(fig, ax, c[2], fs) for lane in "AB" for i, c in enumerate(ex["ref"][lane])}
    x, cur = {}, x0
    for i, _ in enumerate(ex["ref"][flow]):
        x[(flow, i)] = cur; cur += widths[(flow, i)] + gap
    fl, prev_end = ex["ref"][flow], -1.0
    for j, (s, _, _) in enumerate(ex["ref"][other]):
        k = max([i for i, c in enumerate(fl) if c[0] <= s] or [0])
        frac = min(max((s - fl[k][0]) / max(fl[k][1] - fl[k][0], 1e-6), 0.0), 1.0)
        xp = max(x[(flow, k)] + frac * widths[(flow, k)], prev_end + gap)
        x[(other, j)] = xp; prev_end = xp + widths[(other, j)]
    return x, widths, cur


def _draw_lane(fig, ax, ex, xref, widths, lane, items, y, fs, gap, hide_lost, x0, own=None) -> float:
    """Reference chunks of the lane's own speaker(s) keep their reference x (struck-through when lost, or left
    as a slot); copies of the other speaker's chunks and insertions are fitted into the nearest free space.
    ``own`` = the reference lanes anchored here ({lane} for a speaker track, {"A", "B"} for a single
    transcript of the mixture). Returns the right edge."""
    own = own or {lane}
    occupied, slots = [], [None] * len(items)
    for n, it in enumerate(items):                                   # own chunks first: reference positions
        if it[0] not in own:
            continue
        key, lost = (it[0], it[1]), len(it) > 2
        w = widths[key]
        if lost and hide_lost:
            slots[n] = (xref[key], xref[key] + w)
            continue
        xs = _fit(occupied, xref[key], w, gap, x0)
        slots[n] = (xs, xs + w)
        _token(ax, xs, y, ex["ref"][it[0]][it[1]][2], it[0], fs, lost)
        if lost:
            _strike(ax, xs, xs + w, y)
        occupied.append(slots[n])
    for n, it in enumerate(items):                                   # then copies and insertions, in order
        if it[0] in own:
            continue
        if it[0] == "ins":
            text, who, w = it[1], "ins", _text_w(fig, ax, it[1], fs)
            prev = items[n - 1] if n else None
            if prev is None:
                pref = x0
            elif prev[0] in own and len(prev) > 2 and hide_lost:    # substitution takes the empty slot
                pref = slots[n - 1][0]
            else:
                pref = slots[n - 1][1] + gap
        else:
            text, who, w = ex["ref"][it[0]][it[1]][2], it[0], widths[(it[0], it[1])]
            pref = xref[(it[0], it[1])]
        xs = _fit(occupied, pref, w, gap, x0)
        slots[n] = (xs, xs + w); occupied.append(slots[n])
        _token(ax, xs, y, text, who, fs)
    if hide_lost:                                                    # dotted marks on the free part of the slots
        empty = []                                                   # merged slots of hidden lost chunks
        for a, b in sorted(slots[n] for n, it in enumerate(items) if it[0] in own and len(it) > 2):
            if empty and a <= empty[-1][1]:
                empty[-1] = (empty[-1][0], max(empty[-1][1], b))
            else:
                empty.append((a, b))
        for a, b in empty:
            x = a
            for c, d in sorted(occupied):
                if d <= a or c >= b:
                    continue
                if c - x > 0.2:
                    _gap_mark(ax, x, c, y)
                x = max(x, d)
            if b - x > 0.2:
                _gap_mark(ax, x, b, y)
    return max(x1 for _, x1 in occupied) if occupied else x0


HAND_EDITED = {"prez_07a_slowa_b9bd9620_luki.png"}   # finished by hand on 2026-09-09; the script must not touch it


def fig_07a(key: str, hide_lost: bool, with_mix: bool = False) -> None:
    """(a) The same reference words in every row; colour = speaker, red = inserted; a lost word is either
    grey and struck through or left as an empty slot with a dotted mark. Copies of the other speaker's words
    sit at the position of the originals when that space is free. Each row: bold caption, then its lanes.
    ``with_mix`` adds a single-lane row with WhisperX run on the mixture alone (the example's ``mix`` list)."""
    out = OUT_DIR / f"prez_07a_slowa_{key}{'_luki' if hide_lost else ''}{'_whisperx' if with_mix else ''}.png"
    if out.name in HAND_EDITED:
        print("kept hand-edited", out.relative_to(REPO))
        return
    ex = EXAMPLES[key]
    rows = [("referencja", {"A": [("A", i) for i in range(len(ex["ref"]["A"]))],
                            "B": [("B", j) for j in range(len(ex["ref"]["B"]))]})]
    if with_mix:
        rows.append(("sam WhisperX", {"M": ex["mix"]}))
    rows += ex["rows"]
    labels = {"A": "ścieżka A", "B": "ścieżka B", "M": "transkrypcja"}
    heights = [0.55 + 0.43 * (len(lanes) - 1) + 0.47 for _, lanes in rows]
    H = 0.15 + sum(heights) + 0.55
    fig, ax = _inch_canvas(20, H)                                    # cut to the content at the end
    gap = 0.12
    used = {labels[l] for _, lanes in rows for l in lanes} | {"mówca A", "mówca B"}
    x0 = 0.2 + max(_text_w(fig, ax, t, 10.5) for t in used) + 0.15
    fs = 14.0
    while fs > 10.5:
        xref, widths, x_end = _layout_ref(fig, ax, ex, x0, fs, gap)
        if x_end - gap <= 12.5:
            break
        fs -= 0.5
    right, top = 0.0, H - 0.15
    tops = []
    for k, ((name, lanes), h) in enumerate(zip(rows, heights)):
        tops.append(top)
        ax.text(0.2, top - 0.12, name, fontsize=13, fontweight="bold", color=INK, va="center")
        for i, (lane, items) in enumerate(lanes.items()):
            y = top - 0.55 - 0.43 * i
            label = f"mówca {lane}" if k == 0 else labels[lane]
            ax.text(x0 - 0.12, y, label, fontsize=10.5, color=MUTED, ha="right", va="center")
            own = {"A", "B"} if lane == "M" else None
            right = max(right, _draw_lane(fig, ax, ex, xref, widths, lane, items, y, fs, gap, hide_lost, x0, own))
        top -= h
    items = [it for _, lanes in rows for lane in lanes.values() for it in lane]
    y, x, lfs = 0.25, 0.2, 11.5
    ax.text(x, y, "oznaczenia:", fontsize=10.5, color=MUTED, va="center")
    x += _text_w(fig, ax, "oznaczenia:", 10.5) + 0.2
    legend = [("słowa mówcy A", "A"), ("słowa mówcy B", "B")]
    if any(it[0] == "ins" for it in items):
        legend.append(("słowo wstawione", "ins"))
    for text, who in legend:
        _token(ax, x, y, text, who, lfs); x += _text_w(fig, ax, text, lfs) + 0.25
    cap_w = _text_w(fig, ax, ex["caption"], 10)
    W = max(right + 0.3, x + 0.4 + cap_w + 0.2)
    for t in tops[1:]:
        ax.plot([0.2, W - 0.2], [t + 0.15, t + 0.15], color=GRID, linewidth=1)
    ax.text(W - 0.2, y, ex["caption"], fontsize=10, color=MUTED, ha="right", va="center")
    fig.set_size_inches(W, H); ax.set_xlim(0, W)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _tint(h: str, k: float = 0.16):
    return tuple(1 - k * (1 - c) for c in mcolors.to_rgb(h))


def fig_07b() -> None:
    """(b) Time axis 51–59 s: utterances as boxes at their real timestamps, two lanes per system."""
    t0, t1 = EX_WINDOW
    lane_h, lane_gap, group_gap = 1.0, 0.18, 0.75
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.165, 0.10, 0.825, 0.80])
    total = 3 * (2 * lane_h + lane_gap) + 2 * group_gap
    ax.set_xlim(t0, t1); ax.set_ylim(0, total)
    for s, e in EX_OVERLAPS:
        ax.axvspan(s, e, color="#ececec", linewidth=0, zorder=0)
    y_top = total
    for name, lanes in EX_TIMELINE:
        for lane in ("A", "B"):
            y = y_top - lane_h
            ax.text(t0 - 0.08, y + lane_h / 2, lane, fontsize=12, fontweight="bold", color=WHO_COL[lane],
                    ha="right", va="center", clip_on=False)
            for s, e, text, who in lanes[lane]:
                r = Rectangle((s, y + 0.05), e - s, lane_h - 0.1, facecolor=_tint(WHO_COL[who]),
                              edgecolor=WHO_COL[who], linewidth=1.4, zorder=2)
                ax.add_patch(r)
                t = ax.text(max(s, t0) + 0.05, y + lane_h / 2, text, fontsize=10.5, color=INK, va="center", ha="left",
                            zorder=3)
                t.set_clip_path(r)
            y_top = y - lane_gap
        y_top += lane_gap
        ax.text(t0 - 0.36, y_top + lane_h + lane_gap / 2, name, fontsize=13, fontweight="bold", color=INK,
                ha="right", va="center", clip_on=False)
        ax.plot([t0, t1], [y_top - group_gap / 2] * 2, color=GRID, linewidth=1, clip_on=False)
        y_top -= group_gap
    ax.set_xticks(range(int(t0), int(t1) + 1))
    ax.set_xlabel("czas [s]", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=11, length=0)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    handles = [Patch(facecolor=_tint(WHO_COL[k]), edgecolor=WHO_COL[k], label=lab)
               for k, lab in (("A", "słowa mówcy A"), ("B", "słowa mówcy B"), ("ins", "słowo spoza referencji"))]
    handles.append(Patch(facecolor="#ececec", edgecolor="none", label="nakładanie wykryte przez diaryzację"))
    ax.legend(handles=handles, frameon=False, fontsize=11, loc="lower left", ncol=4,
              bbox_to_anchor=(0, 1.0), borderaxespad=0.3)
    fig.text(0.99, 0.015, EX_CAPTION, fontsize=10, color=MUTED, ha="right", va="bottom")
    fig.savefig(OUT_DIR / "prez_07b_os_czasu.png", dpi=220)
    plt.close(fig)


def fig_07c() -> None:
    """(c) Mechanism diagram: reference → no-separation tracks, with the two failure paths drawn as arrows."""
    W, H = 12.5, 4.75
    fig, ax = _inch_canvas(W, H)

    def box(x, y, w, h, color, title, body, body_color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=_tint(color, 0.07), edgecolor=color, linewidth=1.6))
        ax.text(x + 0.2, y + h - 0.17, title, fontsize=11, fontweight="bold", color=color, va="top")
        return ax.text(x + 0.2, y + h - 0.5, body, fontsize=12.5, color=body_color, va="top", linespacing=1.3)

    xl, xr, wl, wr = 0.3, 7.4, 4.9, 4.8
    ya, ha, yb, hb = 3.0, 1.2, 1.5, 1.15
    ax.text(xl + wl / 2, 4.45, "referencja", fontsize=14, fontweight="bold", color=INK, ha="center", va="center")
    ax.text(xr + wr / 2, 4.45, "bez separacji", fontsize=14, fontweight="bold", color=INK, ha="center",
            va="center")
    box(xl, ya, wl, ha, SPK_A, "mówca A",
        "No i na przykład jak się jedzie z tutaj z tym, że jak\njest tutaj, ze śniegiem się ślizgasz (…)", SPK_A)
    box(xl, yb, wl, hb, SPK_B, "mówca B", "Aha, no no no.     No.     No.     Tak, tak…     No.", SPK_B)
    box(xr, ya, wr, ha, SPK_A, "ścieżka A",
        "No i na przykład jak się jedzie tutaj z tym, że…\nJak jest tutaj, ze śniegiem się ślizgasz. (…)", SPK_A)
    box(xr, yb, wr, hb, SPK_B, "ścieżka B", "No i na przykład jak się jedzie ze śniegiem,", SPK_A)
    ax.text(xr + 0.2, yb + hb - 0.85, "(słowa mówcy A)", fontsize=11, color=ERR, va="top", style="italic")

    arrow = dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=1.8, shrinkA=0, shrinkB=0)
    ya_mid, yb_mid = ya + ha / 2, yb + hb / 2
    ax.annotate("", xy=(xr, ya_mid + 0.15), xytext=(xl + wl, ya_mid + 0.15),
                arrowprops=dict(color=SPK_A, **arrow))
    ax.text((xl + wl + xr) / 2, ya_mid + 0.3, "w całości", fontsize=11, color=SPK_A, ha="center", va="bottom")
    ax.annotate("", xy=(xr, yb_mid + 0.12), xytext=(xl + wl, ya_mid - 0.3),
                arrowprops=dict(color=SPK_A, linestyle="--", connectionstyle="arc3,rad=0.25", **arrow))
    ax.text((xl + wl + xr) / 2 + 0.05, (ya_mid - 0.3 + yb_mid) / 2 + 0.22, "kopia 9 słów", fontsize=11,
            color=SPK_A, ha="center", va="bottom")
    ax.annotate("", xy=(xl + wl + 1.1, yb_mid - 0.1), xytext=(xl + wl, yb_mid - 0.1),
                arrowprops=dict(color=SPK_B, **arrow))
    ax.text(xl + wl + 1.2, yb_mid - 0.1, "✕", fontsize=15, color=ERR, ha="left", va="center", fontweight="bold")
    ax.text(xl + wl + 0.08, yb_mid - 0.5, "9 słów zgubionych", fontsize=11, color=ERR, ha="left", va="top")

    ax.text(0.3, 0.85, "pełny potok: ścieżka A kompletna; ścieżka B zawiera 4 z 5 potakiwań mówcy B "
                      "(jedno zgubione, jedno dodane „Mhm.”)", fontsize=12, color=INK, va="center")
    ax.text(W - 0.2, 0.15, EX_CAPTION, fontsize=10, color=MUTED, ha="right", va="bottom")
    fig.savefig(OUT_DIR / "prez_07c_mechanizm.png", dpi=220)
    plt.close(fig)


def fig_07d() -> None:
    """(d) The shorter example: one 1.5 s overlap at 11,3–12,8 s, plain table in large type."""
    cells = [("referencja", "Spokojnie. Pojadę na około i tyle.", "Mhm. U, tutaj to jest trochę…"),
             ("bez separacji", "Spokojnie. To jadę na autobusie.", "Tutaj to jest."),
             ("pełny potok", "Spokojnie. Pojedę na około, jutro.", "Mhm. O, tutaj to jest trochę źle.")]
    W, H = 11.5, 4.0
    fig, ax = _inch_canvas(W, H)
    xl, xa, xb = 2.3, 2.5, 7.2
    ax.text(xa, 3.55, "Mówca A", fontsize=15, fontweight="bold", color=SPK_A, va="center")
    ax.text(xb, 3.55, "Mówca B", fontsize=15, fontweight="bold", color=SPK_B, va="center")
    ax.plot([0.2, W - 0.2], [3.22, 3.22], color=AXIS, linewidth=1)
    for k, (name, a, b) in enumerate(cells):
        y = 2.75 - k * 0.85
        ax.text(xl, y, name, fontsize=14, fontweight="bold", color=INK, ha="right", va="center")
        ax.text(xa, y, a, fontsize=15, color=INK, va="center")
        ax.text(xb, y, b, fontsize=15, color=INK, va="center")
        if k:
            ax.plot([0.2, W - 0.2], [y + 0.43, y + 0.43], color=GRID, linewidth=1)
    ax.text(W - 0.2, 0.18, "CLARIN_fragments · ccfbb9db__seg00 (TEST) · 9,6–12,9 s · nakładanie 11,3–12,8 s",
            fontsize=10, color=MUTED, ha="right", va="bottom")
    fig.savefig(OUT_DIR / "prez_07d_krotki.png", dpi=220)
    plt.close(fig)


def fig_07e() -> None:
    """(e) Numbers instead of text: missed reference words by region and insertions, TEST set (ch. 6, fn. 5)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [3, 1.05]})
    series = [("potok bez separacji", NOSEP), ("pełny potok", FULL)]

    def bars(ax, groups, values, ylabel, nd):
        x, w = np.arange(len(groups)), 0.34
        top = max(max(v) for v in values)
        for k, ((name, col), vals) in enumerate(zip(series, values)):
            off = (k - 0.5) * (w + 0.03)
            bs = ax.bar(x + off, vals, width=w, color=col, label=name, linewidth=0)
            for b, v in zip(bs, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, pl(v, nd), ha="center", va="bottom",
                        fontsize=11, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11.5)
        ax.set_ylabel(ylabel, fontsize=11.5)
        ax.set_ylim(0, top * 1.18)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        ax.spines["left"].set_visible(False)

    bars(ax1, ["w nakładaniu", "do 1 s od nakładania", "poza nakładaniem"],
         [[39.1, 24.9, 18.5], [31.6, 21.8, 17.6]], "pominięte słowa referencji [%]", 1)
    bars(ax2, ["wstawienia"], [[4.55], [3.10]], "wstawione słowa [pkt cpWER]", 2)
    ax1.legend(frameon=False, fontsize=11, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.text(0.99, 0.015, "CLARIN_fragments · zbiór TEST · 118 fragmentów", fontsize=10, color=MUTED,
             ha="right", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_DIR / "prez_07e_liczby.png", dpi=220)
    plt.close(fig)


def fig_07_all() -> None:
    for key in EXAMPLES:
        fig_07a(key, hide_lost=False); fig_07a(key, hide_lost=True)
    fig_07a("b9bd9620", hide_lost=True, with_mix=True)
    fig_07b(); fig_07c(); fig_07d(); fig_07e()


# ---- slide 8: table-style horizontal bars ---------------------------------------------------------
def table_barh(fname: str, headers: list[str], rows: list[tuple[str, ...]], values: list[float],
               colors: list[str], legend: list[Patch], xlim: tuple[float, float], xlabel: str,
               figsize, vline: float | None = None, vlabel: str = "") -> None:
    """Horizontal bars with aligned text columns on the left. Column positions are measured from
    the text itself: each column is as wide as its longest entry plus a fixed gap, and the chart
    starts one gap after the last column."""
    fig = plt.figure(figsize=figsize)
    renderer = fig.canvas.get_renderer()
    fs_cell, fs_head, col_gap, chart_gap = 12.5, 10.5, 0.45, 0.5   # points / inches

    def tw(text, fs):
        t = fig.text(0, 0, text, fontsize=fs)
        w = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return w

    col_w = [max([tw(h, fs_head)] + [tw(r[i], fs_cell) for r in rows]) for i, h in enumerate(headers)]
    col_x_in = [0.3]
    for w in col_w[:-1]:
        col_x_in.append(col_x_in[-1] + w + col_gap)
    ax_left_in = col_x_in[-1] + col_w[-1] + chart_gap
    fig_w = figsize[0]
    col_x = [x / fig_w for x in col_x_in]
    ax = fig.add_axes([ax_left_in / fig_w, 0.26, 0.97 - ax_left_in / fig_w, 0.60])

    n = len(rows)
    y = np.arange(n)[::-1]
    ax.barh(y, values, height=0.6, color=colors, linewidth=0)
    for yi, v in zip(y, values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, yi, pl(v), va="center", ha="left", fontsize=12, color=INK)
    if vline is not None:
        ax.axvline(vline, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(vline - (xlim[1] - xlim[0]) * 0.012, n + 0.05, vlabel, fontsize=11, color=INK,
                ha="right", va="bottom")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4 + (1.1 if vline is not None else 0.0))
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_visible(False)
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    for x, h in zip(col_x, headers):
        ax.text(x, n - 0.35, h, transform=tr, fontsize=fs_head, color=MUTED, ha="left", va="bottom")
    for yi, row in zip(y, rows):
        for x, cell in zip(col_x, row):
            ax.text(x, yi, cell, transform=tr, fontsize=fs_cell, color=INK, ha="left", va="center")
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.5, 0.01))
    fig.savefig(OUT_DIR / fname, dpi=220)
    plt.close(fig)


def fig_08a() -> None:
    rows = [
        ("MF2-matched", "PolSESS_128k", 24.0, False),
        ("TF-Locoformer-M", "Libri2Mix", 24.0, False),
        ("TF-Locoformer-M", "WHAMR!", 24.4, False),
        ("MF2-matched", "PolSESS_128k, tylko czysta mowa", 25.9, True),
        ("TF-Locoformer-M", "WSJ0-2mix", 26.4, True),
    ]
    table_barh("prez_08a_modele_dane.png", ["model separacji", "korpus treningowy"],
               [(r[0], r[1]) for r in rows], [r[2] for r in rows],
               [CLEAN_TRAIN if r[3] else NOISY_TRAIN for r in rows],
               [Patch(facecolor=NOISY_TRAIN, label="korpus z szumem i/lub pogłosem"),
                Patch(facecolor=CLEAN_TRAIN, label="korpus bez szumu i pogłosu")],
               xlim=(20, 30), xlabel="cpWER", figsize=(10.5, 4.6),
               vline=28.0, vlabel="potok bez separacji: 28,0")


def fig_08b() -> None:
    rows = [
        ("MF2-matched", "PolSESS_128k", "8 kHz", 24.0, "0–4 kHz + BWE"),
        ("TIGER", "EchoSet", "16 kHz", 22.8, "0–8 kHz"),
    ]
    table_barh("prez_08b_pasmo.png", ["model separacji", "korpus treningowy", "próbkowanie"],
               [(r[0], r[1], r[2]) for r in rows], [r[3] for r in rows],
               [BAND[r[4]] for r in rows],
               [Patch(facecolor=c, label=k) for k, c in BAND.items()],
               xlim=(20, 26), xlabel="cpWER", figsize=(10, 3.2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frag", default="ccfbb9db__seg00")
    ap.add_argument("--window", nargs=2, type=float, default=(73.5, 77.5),
                    help="seconds within the fragment for the slide-2 schematic")
    ap.add_argument("--row", type=int, default=10868, help="row of the PolSESS_128k test CSV for slide 3")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    print("CVD sanity check (OKLab ΔE×100; target ≥ 8 under CVD, ≥ 15 normal):")
    cvd_check("speakers", [SPK_A, SPK_B])
    cvd_check("results", [NOSEP, FULL])
    cvd_check("training", [NOISY_TRAIN, CLEAN_TRAIN])
    cvd_check("band", list(BAND.values()))
    cvd_check("layers", [c for *_, c in LAYERS])
    fig_02(load_streams(args.frag), *args.window)
    fig_02bc(load_streams("b9bd9620__seg00"), 0.15, 4.1)
    fig_03(args.row)
    fig_04()
    fig_06()
    fig_07_all()
    fig_08a()
    fig_08b()
    for p in sorted(OUT_DIR.glob("prez_*.png")):
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
