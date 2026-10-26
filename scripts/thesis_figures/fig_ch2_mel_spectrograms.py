"""Rysunek 2.5 (ch2 §2.1.3) — spektrogramy melowe: miks dwóch mówców i oba źródła.

Odtwarza — z polskimi opisami osi, tytułami paneli i czcionką tej samej
wielkości co w pozostałych rysunkach pracy — wykres, który pierwotnie powstał
w panelu spektrogramów notatnika `test_model_interactive.ipynb`
(`ModelTester._plot_spectrograms`) i był wklejony do rozdziału jako zrzut.
Parametry analizy są te same, co tam: n_fft 1024 (128 ms), hop 128 (16 ms),
480 pasm melowych, skala dB względem maksimum panelu, oś Y melowa do 4 kHz.
Długie okno daje przy 8 kHz rozdzielczość 7,8 Hz, potrzebną, żeby F0 i 2·F0
nie wpadały do sąsiednich prążków; skala melowa rozciąga rejon F0 i niższych
alikwotów — o nich mówi podpis rysunku.

Dane: PolSESS_C_final_128_v2/test, wariant C. Czysty miks = sp1_dry + sp2_dry,
bo wariant C kasuje w miksie wszystkie warstwy poza dwoma suchymi mówcami
(tożsamość MM-IPC, komentarz w `datasets/polsess_dataset.py`); sumowanie dwóch
plików z `clean/` jest więc dokładnie tym, co zwraca loader dla wariantu C.

Domyślnie skrypt wybiera parę polsko-polską (clarin_emu po obu stronach)
o największej różnicy mediany F0 między mówcami wśród pierwszych --scan
takich wierszy CSV, pod warunkiem że obaj mówcy są aktywni w ≥ 60 % ramek —
żeby rysunek pokazywał to, co opisuje podpis: różne F0 i alikwoty przy
nakładających się wypowiedziach. `--row N` wymusza konkretny wiersz CSV
(numeracja od 0, po nagłówku). Wybrany wiersz i pliki są wypisywane.

Usage: python scripts/thesis_figures/fig_ch2_mel_spectrograms.py [--row N] [--scan 300]
                                                                [--out-dir DIR]
"""

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "thesis" / "my-writing" / "figures" / "ch02"
DATA = Path.home() / "datasets" / "PolSESS_C_final_128_v2" / "test"
CSV = DATA / "corpus_PolSESS_C_final_128_v2_test_final.csv"

SR = 8000
NFFT, HOP, N_MELS = 1024, 128, 480     # jak w notatniku
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def load(name):
    y, sr = librosa.load(DATA / "clean" / name, sr=None, mono=True)
    assert sr == SR, (name, sr)
    return y


def activity(y):
    """Udział ramek (16 ms) z energią nie niższą niż 30 dB poniżej maksimum."""
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return float(np.mean(db > -30))


def median_f0(y):
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR,
                                 frame_length=NFFT, hop_length=HOP)
    f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else np.nan


def pick_row(df, scan):
    polish = df[df["speech1OryginalPath"].str.contains("clarin", case=False)
                & df["speech2OryginalPath"].str.contains("clarin", case=False)]
    best, best_score = None, -1.0
    for idx, row in polish.head(scan).iterrows():
        y1, y2 = load(row["speaker1File"]), load(row["speaker2File"])
        if min(activity(y1), activity(y2)) < 0.6:
            continue
        f1, f2 = median_f0(y1), median_f0(y2)
        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue
        score = abs(np.log(f1 / f2))
        if score > best_score:
            best, best_score = (idx, f1, f2), score
    if best is None:
        raise SystemExit("żaden z przeskanowanych wierszy nie spełnia warunków")
    idx, f1, f2 = best
    print(f"wybrany wiersz {idx}: mediana F0 mówca 1 = {f1:.0f} Hz, mówca 2 = {f2:.0f} Hz")
    return idx


def mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=NFFT, hop_length=HOP,
                                       n_mels=N_MELS, fmax=SR / 2)
    return librosa.power_to_db(m, ref=np.max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--scan", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    idx = args.row if args.row is not None else pick_row(df, args.scan)
    row = df.loc[idx]
    y1, y2 = load(row["speaker1File"]), load(row["speaker2File"])
    mix = y1 + y2
    print(f"wiersz {idx}: {row['speaker1File']} + {row['speaker2File']}  "
          f"(miks: {row['mixFile']})")

    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.4), sharex=True)
    for ax, (title, y) in zip(axes, [("miks", mix), ("mówca 1", y1), ("mówca 2", y2)]):
        librosa.display.specshow(mel_db(y), sr=SR, hop_length=HOP, x_axis="time",
                                 y_axis="mel", fmax=SR / 2, cmap="magma", ax=ax)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
        ax.set_ylabel("Częstotliwość [Hz]")
        ax.set_xlabel("")
        ax.xaxis.set_major_formatter(PL)
        ax.tick_params(length=3, color=AXIS)
    axes[-1].set_xlabel("Czas [s]")
    fig.subplots_adjust(hspace=0.28)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"rys_ch2_mel_spectrograms.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"written: {args.out_dir}/rys_ch2_mel_spectrograms.{{png,pdf}}")


if __name__ == "__main__":
    main()
