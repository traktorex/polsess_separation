"""Rysunek 2.4 (ch2 §2.1.2) — trzy reprezentacje sygnału mowy.

Przebieg czasowy, spektrogram STFT i wyjście uczonego enkodera (SepFormer,
Conv1d PRZED ReLU — wartości ze znakiem, mapa rozbieżna). Port komórki dawnego notatnika roboczego
(`plot_notebook.ipynb`, usunięty z repozytorium), z której rysunek pierwotnie pochodził jako
zrzut; dochodzi wybór koloru zera na mapie wyjścia enkodera:

  --zero black   niebieski → czarny → czerwony (jak w notatniku; cisza = czarna
                 plama, bo zero i małe aktywacje są czarne),
  --zero white   niebieski → biały → czerwony (cisza zlewa się z tłem strony,
                 widać tylko to, co enkoder faktycznie „zapala").

Zakres dynamiki aktywacji jest ściśnięty potęgą: sign(x)·(|x|/p99,5)^gamma,
gamma = 0,5 — jak w notatniku (--gamma, --pct zmieniają oba parametry);
podziałka paska kolorów pokazuje wartości surowe. Filtry posortowane wg częstotliwości szczytowej FFT jądra (i fazy
w obrębie tej samej częstotliwości), najwyższa częstotliwość u góry.

Dane: pierwsze 2 s pliku `PolSESS_C_new_64/train/clean/000mqbomjtudfgw5.wav`
(ten sam co w notatniku); enkoder z checkpointu SepFormera 64k (baseline,
kodowanie pozycyjne) — jądra enkodera wyglądają tak samo w każdym
checkpointcie tej architektury, więc konkretny run nie ma znaczenia dla
obrazka. Skrypt działa na CPU (enkoder to jeden Conv1d).

Usage: python scripts/thesis_figures/fig_ch2_representations.py [--zero black|white]
           [--gamma 0.5] [--ckpt PATH] [--audio PATH] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import load_model_for_inference  # noqa: E402

DEFAULT_OUT = REPO / "thesis" / "thesis-writing" / "figures" / "ch02"
DEFAULT_AUDIO = (Path.home() / "datasets" / "PolSESS_C_new_64" / "PolSESS_C_new_64"
                 / "train" / "clean" / "000mqbomjtudfgw5.wav")
DEFAULT_CKPT = (REPO / "checkpoints" / "sepformer" / "SB"
                / "64k_baseline_posenc_bym7223m" / "sepformer_SB_best.pt")

INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
BLUE, RED = (0.2, 0.4, 1.0), (1.0, 0.2, 0.2)
CMAPS = {
    "black": LinearSegmentedColormap.from_list("blue_black_red", [BLUE, (0, 0, 0), RED]),
    "white": LinearSegmentedColormap.from_list("blue_white_red", [BLUE, (1, 1, 1), RED]),
}
CLIP_SECONDS = 2.0
N_FFT, HOP = 512, 128
PL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero", choices=("black", "white"), default="black")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--pct", type=float, default=99.5,
                    help="percentyl |aktywacji| mapowany na pełne nasycenie")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    waveform, sr = torchaudio.load(str(args.audio))
    wave = waveform.mean(dim=0)[: int(CLIP_SECONDS * sr)]
    t_audio = np.arange(wave.numel()) / sr

    spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=2.0)(wave)
    spec_db = 10.0 * torch.log10(spec.clamp(min=1e-10)).numpy()
    t_spec = np.arange(spec_db.shape[1]) * HOP / sr
    freqs = np.linspace(0, sr / 2, spec_db.shape[0])

    model, _ = load_model_for_inference(str(args.ckpt), device="cpu")
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    W = m.encoder.conv1d.weight.detach().squeeze(1).numpy()          # [N, K]
    N, K = W.shape
    stride = m.encoder.conv1d.stride[0]
    F = np.fft.rfft(W, n=K, axis=1)
    peak = np.abs(F).argmax(axis=1)
    phase = np.angle(F)[np.arange(N), peak]
    order = np.lexsort((phase, peak))[::-1]

    with torch.no_grad():
        enc = m.encoder.conv1d(wave[None, None, :]).squeeze(0).numpy()  # pre-ReLU
    enc = enc[order]
    t_enc = np.arange(enc.shape[1]) * stride / sr
    p995 = np.percentile(np.abs(enc), args.pct)
    disp = np.sign(enc) * np.clip(np.abs(enc) / p995, 0, 1) ** args.gamma

    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "font.family": "sans-serif",
    })
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(3, 12, figure=fig, hspace=0.5, wspace=0.4, height_ratios=[1, 1.4, 1.7])

    ax_w = fig.add_subplot(gs[0, 0:11])
    ax_w.plot(t_audio, wave.numpy(), lw=0.6, color="tab:blue")
    ax_w.set_xlim(0, CLIP_SECONDS)
    ax_w.set_xlabel("Czas [s]")
    ax_w.set_ylabel("Amplituda")
    ax_w.set_title("1) Przebieg czasowy")
    ax_w.grid(alpha=0.3)

    ax_s = fig.add_subplot(gs[1, 0:11], sharex=ax_w)
    im_s = ax_s.imshow(spec_db, origin="lower", aspect="auto", cmap="magma",
                       extent=[t_spec[0], t_spec[-1], freqs[0], freqs[-1]],
                       vmin=spec_db.max() - 80, vmax=spec_db.max())
    ax_s.set_xlim(0, CLIP_SECONDS)
    ax_s.set_xlabel("Czas [s]")
    ax_s.set_ylabel("Częstotliwość [Hz]")
    ax_s.set_title("2) Spektrogram")
    fig.colorbar(im_s, cax=fig.add_subplot(gs[1, 11]), label="[dB]")

    ax_o = fig.add_subplot(gs[2, 0:11], sharex=ax_w)
    im_o = ax_o.imshow(disp, aspect="auto", cmap=CMAPS[args.zero],
                       extent=[t_enc[0], t_enc[-1], N, 0], vmin=-1, vmax=1,
                       interpolation="nearest")
    ax_o.set_xlim(0, CLIP_SECONDS)
    ax_o.set_xlabel("Czas [s]")
    ax_o.set_ylabel("Indeks filtru")
    ax_o.set_title("3) Wyjście enkodera SepFormer")
    raw = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    cbar = fig.colorbar(im_o, cax=fig.add_subplot(gs[2, 11]), label="aktywacja")
    cbar.set_ticks(np.sign(raw) * np.abs(raw) ** args.gamma)
    cbar.set_ticklabels([f"{v:+.2f}".replace(".", ",") if v else "0" for v in raw])
    for ax in (ax_w, ax_s, ax_o):
        ax.xaxis.set_major_formatter(PL)
        ax.yaxis.set_major_formatter(PL)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rys_ch2_representations_{args.zero}"
    if args.pct != 99.5 or args.gamma != 0.5:
        stem += f"_p{args.pct:g}_g{args.gamma:g}"
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  (N={N}, K={K}, stride={stride}, "
          f"p99,5={p995:.3f})")


if __name__ == "__main__":
    main()
