"""Objective acoustic-complexity scorer for the CLARIN eval fragments.

The corpus noise labels (``Niski``/``Średni``/``Wysoki``) proved unreliable
below ``Wysoki``, so this computes objective per-fragment acoustic metrics and
calibrates them against the author's by-ear grades (0 = studio-clean …
10 = barely comprehensible). Two purposes:

  (a) honestly characterise the 128-fragment eval set, and
  (b) inform a possible later re-selection of un-annotated fragments.

Metrics per fragment (chunked + median-aggregated where the underlying metric
is window-defined, mirroring ``asr_pipeline/eval/layer2.py``):

  - **SQUIM objective** (torchaudio) — squim_stoi / squim_pesq / squim_si_sdr.
    Reuses ``layer2.squim_chunked`` + the SQUIM loader directly.
  - **DNSMOS** (Microsoft DNS-Challenge ONNX) — SIG / BAK / OVR. Faithful
    replication of ``dnsmos_local.py``: 9.01 s windows, 1 s hop, raw samples to
    the ``sig_bak_ovr.onnx`` graph, non-personalized poly fit, mean over
    windows. (The official aggregation is mean; we keep it.)
  - **Brouhaha** (``pyannote/brouhaha``) — frame-level SNR + C50 reverb
    clarity, the two axes matching the author's by-ear criteria. Runs in an
    ISOLATED venv via subprocess (it pins an older numpy / pyannote.audio that
    would break the main venv). Falls back to WADA-SNR + ``nan`` C50 if the
    isolated venv / gated model is unavailable.
  - **WADA-SNR** — model-free SNR estimate, always computed; the Brouhaha-SNR
    fallback and an independent cross-check.
  - **Signal stats** — integrated LUFS (pyloudnorm), clipping rate.

Per-fragment failures never abort the batch: the offending metric cells go to
``nan``, a loud per-fragment stderr line is printed, and a summary count is
reported at the end (SCOPE no-silent-substitution spirit — every skip is
surfaced, never silent).

Outputs (under ``~/datasets/eval/clarin_fragments/``):

  - ``acoustic_scores.csv`` — one row per fragment, all metrics + n_chunks.
  - ``ACOUSTIC_SCORES_REPORT.md`` (``--report`` / always after scoring) —
    correlation vs grades, Wysoki separation, optional composite, dev-vs-test
    + corpus-label cross-tab, the 10 hardest un-annotated test fragments,
    and every failure/fallback that occurred.

Usage::

    python scripts/score_fragment_acoustics.py            # score all + report
    python scripts/score_fragment_acoustics.py --force    # recompute existing
    python scripts/score_fragment_acoustics.py --report   # report from cached CSV
    python scripts/score_fragment_acoustics.py --no-brouhaha   # skip the slow stage
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.eval.layer2 import (  # noqa: E402
    load_squim_model,
    squim_chunked,
    unload_squim_model,
)
from scripts.build_nzr_aids import parse_robione  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

FRAGMENTS_ROOT = Path(
    os.environ.get(
        "CLARIN_FRAGMENTS_ROOT",
        str(Path.home() / "datasets" / "eval" / "clarin_fragments"),
    )
)
MANIFEST_PATH = FRAGMENTS_ROOT / "manifest.csv"
SCORES_CSV = FRAGMENTS_ROOT / "acoustic_scores.csv"
REPORT_MD = FRAGMENTS_ROOT / "ACOUSTIC_SCORES_REPORT.md"
ROBIONE_PATH = Path("/mnt/f/clarin_fragments/robione.txt")

# Dir excluded from the eval universe (per task spec).
_EXCLUDE_DIR = "_bwe_benchmark_2026-06-10"

SR = 16_000

# DNSMOS (Microsoft DNS-Challenge sig_bak_ovr.onnx, P.835).
DNSMOS_CACHE = Path(os.environ.get("DNSMOS_CACHE", str(Path.home() / ".cache" / "dnsmos")))
DNSMOS_ONNX = DNSMOS_CACHE / "sig_bak_ovr.onnx"
DNSMOS_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_INPUT_LENGTH_S = 9.01
DNSMOS_HOP_S = 1.0
# Non-personalized (is_personalized_MOS=False) poly fits from dnsmos_local.py.
_DNSMOS_POLY_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_DNSMOS_POLY_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_DNSMOS_POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

# Brouhaha isolated venv (created out-of-band; see module docstring + the
# task's hard constraint about not breaking the main venv). The model weights
# are loaded from the repo's shipped checkpoint, NOT the HF hub — `pyannote/
# brouhaha` is gated and this token is not on its allow-list (the local ckpt is
# the same trained model, so this sidesteps the gating entirely).
BROUHAHA_VENV_PY = Path(
    os.environ.get("BROUHAHA_VENV_PY", "/tmp/brouhaha_venv/bin/python")
)
BROUHAHA_CKPT = Path(
    os.environ.get(
        "BROUHAHA_CKPT", "/tmp/brouhaha-vad/models/best/checkpoints/best.ckpt"
    )
)

# Clipping: a sample at/above this magnitude counts as clipped.
_CLIP_THRESHOLD = 0.999

# The author's by-ear grades — the calibration target (n=16). 0 = studio-clean,
# 10 = barely comprehensible.
AUTHOR_GRADES: dict[str, float] = {
    "065a9896__seg00": 3.0,
    "0ab10929__seg00": 3.0,
    "49e09a03__seg00": 3.0,
    "4f9251fe__seg00": 3.0,
    "543bf543__seg00": 3.0,
    "595aa511__seg00": 3.0,
    "62e43b45__seg00": 3.0,
    "72ca135e__seg00": 4.0,
    "94a0d89a__seg00": 4.0,
    "da1b5a78__seg00": 3.0,
    "33a47eae__seg00": 4.5,  # reverb, no scene noise
    "9a651086__seg00": 7.0,  # car noise + bleed
    "eaabbc3b__seg00": 2.5,
    "48fbaab6__seg00": 3.0,
    "f59efd6d__seg00": 3.5,  # keyboard clicks
    "eb33c0f0__seg00": 5.0,  # school-hall noise
}

# Dev set = the 16 fragments whose manifest Autor starts with one of these.
# (These coincide with AUTHOR_GRADES; both are kept so neither silently drifts.)
DEV_AUTOR_PREFIXES = ("64a6a2eccbbe", "68179aa6", "68493cf9", "69386555")

# Metrics whose larger value means HARDER audio (the rest mean EASIER). Used to
# sign-align the composite so higher == harder.
_HIGHER_IS_HARDER = {"wada_snr_neg", "clip_rate", "dnsmos_bak_neg"}


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_mono_16k(path: Path) -> np.ndarray:
    """Read a WAV as mono float32. Fragments are 16 kHz by construction; a
    mismatch is an error (we do not silently resample — SCOPE §4)."""
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr} != expected {SR}")
    return arr


# ---------------------------------------------------------------------------
# WADA-SNR (model-free, in-script)
# ---------------------------------------------------------------------------

# WADA lookup table from Kim & Stern, "Robust Signal-to-Noise Ratio Estimation
# Based on Waveform Amplitude Distribution Analysis" (Interspeech 2008), via the
# canonical Labrosa/snreval adaptation. `db_vals = arange(-20, 101)` (121 points,
# 1-dB steps); `g_vals` is the corresponding curve of the statistic
# `v3 = log(E[|z|]) - E[log|z|]` for Gamma-distributed speech + Gaussian noise.
# v3 is monotone INCREASING in SNR, so the inversion finds the last index where
# `g_vals < v3` and interpolates. Verbatim from the reference (do not re-derive).
_WADA_DB = np.arange(-20, 101).astype(float)
_WADA_G = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006,
    0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272,
    0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373,
    0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
    0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985,
    0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349,
    1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
    1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313,
    1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.3605727, 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619,
    1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
    1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915,
    1.5229097, 1.528578, 1.53389835, 1.5391211, 1.5439065, 1.54858517,
    1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
    1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134,
    1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905,
    1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143,
    1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
    1.63204102,
])


def wada_snr(x: np.ndarray, eps: float = 1e-10) -> float:
    """Model-free WADA SNR estimate in dB (Kim & Stern 2008).

    Peak-normalises ``x``, computes the amplitude-distribution statistic
    ``v3 = log(E[|x|]) - E[log|x|]`` and inverts the published WADA table.
    Higher = cleaner. Gain-invariant (peak-normalisation cancels any global
    scale). Returns ``nan`` on a degenerate (empty / all-silent) signal.

    Faithful to the canonical Labrosa/snreval adaptation; the reference's final
    energy-ratio re-derivation of the SNR is algebraically identical to the
    interpolated ``wav_snr`` (``10·log10(dFactor) == wav_snr``), so we return
    that directly.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    peak = np.abs(x).max()
    if peak <= 0:
        return float("nan")
    abs_wav = np.abs(x) / peak
    abs_wav[abs_wav < eps] = eps
    v1 = max(eps, float(abs_wav.mean()))          # E[|z|]
    v2 = float(np.log(abs_wav).mean())            # E[log|z|]
    v3 = np.log(v1) - v2                           # log(E[|z|]) - E[log|z|]
    below = np.where(_WADA_G < v3)[0]
    if below.size == 0:
        return float(_WADA_DB[0])                  # v3 below table → clamp low
    idx = int(below.max())
    if idx == len(_WADA_DB) - 1:
        return float(_WADA_DB[-1])                 # v3 above table → clamp high
    # Linear interpolation between adjacent table points.
    return float(
        _WADA_DB[idx]
        + (v3 - _WADA_G[idx]) / (_WADA_G[idx + 1] - _WADA_G[idx])
        * (_WADA_DB[idx + 1] - _WADA_DB[idx])
    )


# ---------------------------------------------------------------------------
# Signal stats: clipping rate, LUFS
# ---------------------------------------------------------------------------


def clipping_rate(x: np.ndarray, threshold: float = _CLIP_THRESHOLD) -> float:
    """Fraction of samples with ``|x| >= threshold``. Empty signal → ``nan``."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return float("nan")
    return float((np.abs(x) >= threshold).mean())


def integrated_lufs(x: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness (LUFS, ITU-R BS.1770) via pyloudnorm; falls back to
    RMS-dBFS if pyloudnorm is unavailable. Silence → ``nan``.

    The fallback uses a distinct code path (RMS, not gated LUFS) — the report
    flags which one ran so the number is never mistaken for the other.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.any(np.abs(x) > 0):
        return float("nan")
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x))
    except Exception:
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms <= 0:
            return float("nan")
        return float(20.0 * np.log10(rms))


# ---------------------------------------------------------------------------
# DNSMOS (faithful dnsmos_local.py replication)
# ---------------------------------------------------------------------------


def ensure_dnsmos_model() -> Path:
    """Download ``sig_bak_ovr.onnx`` to the cache dir if absent. Returns the
    path. Raises on a failed download or an LFS-pointer stub."""
    DNSMOS_CACHE.mkdir(parents=True, exist_ok=True)
    if not DNSMOS_ONNX.exists():
        print(f"[dnsmos] downloading {DNSMOS_URL} -> {DNSMOS_ONNX}", file=sys.stderr)
        urlretrieve(DNSMOS_URL, DNSMOS_ONNX)
    # Guard against a Git-LFS pointer text masquerading as the binary.
    if DNSMOS_ONNX.stat().st_size < 100_000:
        raise RuntimeError(
            f"{DNSMOS_ONNX} is {DNSMOS_ONNX.stat().st_size} bytes — likely an "
            "LFS pointer, not the real ONNX. Delete and re-download."
        )
    return DNSMOS_ONNX


def load_dnsmos_session(onnx_path: Optional[Path] = None):
    """Load the DNSMOS ONNX session (CPU). Lazy import so tests/imports that
    never score don't need onnxruntime."""
    import onnxruntime as ort

    if onnx_path is None:
        onnx_path = ensure_dnsmos_model()
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _dnsmos_polyfit(sig_raw: float, bak_raw: float, ovr_raw: float) -> tuple:
    return (
        float(_DNSMOS_POLY_SIG(sig_raw)),
        float(_DNSMOS_POLY_BAK(bak_raw)),
        float(_DNSMOS_POLY_OVR(ovr_raw)),
    )


def dnsmos_windowed(audio: np.ndarray, session, sr: int = SR,
                    batch_size: int = 32) -> dict:
    """DNSMOS SIG/BAK/OVR via the official windowing: 9.01 s windows, 1 s hop,
    short clips self-concatenated up to one window, trailing partials skipped,
    mean over windows.

    Windows are fed to the ONNX session in sub-batches of ``batch_size`` (the
    graph's first dim is dynamic) — numerically identical to one-by-one but
    several times faster on CPU. Returns ``{dnsmos_sig, dnsmos_bak,
    dnsmos_ovr, dnsmos_n_chunks}``; all-``nan`` with 0 chunks if no full
    window exists.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    len_samples = int(DNSMOS_INPUT_LENGTH_S * sr)
    hop = int(DNSMOS_HOP_S * sr)
    if audio.size == 0:
        return _dnsmos_nan()
    # Self-concatenate short clips, exactly as dnsmos_local.py does.
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / sr) - DNSMOS_INPUT_LENGTH_S) + 1
    windows = []
    for idx in range(num_hops):
        start = idx * hop
        seg = audio[start : start + len_samples]
        if len(seg) < len_samples:
            continue
        windows.append(seg)
    if not windows:
        return _dnsmos_nan()
    sig_vals, bak_vals, ovr_vals = [], [], []
    for i in range(0, len(windows), batch_size):
        feats = np.stack(windows[i : i + batch_size]).astype(np.float32)
        raw = session.run(None, {"input_1": feats})[0]  # (B, 3)
        for sig_raw, bak_raw, ovr_raw in np.asarray(raw, dtype=np.float64):
            s, b, o = _dnsmos_polyfit(float(sig_raw), float(bak_raw), float(ovr_raw))
            sig_vals.append(s)
            bak_vals.append(b)
            ovr_vals.append(o)
    return {
        "dnsmos_sig": float(np.mean(sig_vals)),
        "dnsmos_bak": float(np.mean(bak_vals)),
        "dnsmos_ovr": float(np.mean(ovr_vals)),
        "dnsmos_n_chunks": len(sig_vals),
    }


def _dnsmos_nan() -> dict:
    return {
        "dnsmos_sig": float("nan"),
        "dnsmos_bak": float("nan"),
        "dnsmos_ovr": float("nan"),
        "dnsmos_n_chunks": 0,
    }


# ---------------------------------------------------------------------------
# Brouhaha (isolated venv via subprocess)
# ---------------------------------------------------------------------------

# The worker script run inside the isolated venv. BATCH worker: loads the
# model ONCE (on GPU when the isolated venv's torch sees CUDA), then scores
# every wav listed in the manifest file — one ``RES``/``FAILED`` line per wav,
# flushed as it goes. Kept inline (not a repo file) because it must import the
# isolated venv's pinned pyannote.audio, never the main env's. Per-wav
# failures are reported per-line, never silently dropped (SCOPE spirit).
_BROUHAHA_WORKER = r'''
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
list_file = sys.argv[1]
ckpt = sys.argv[2]
try:
    import torch
    from pyannote.audio import Model, Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.from_pretrained(ckpt, strict=False)
    inference = Inference(model, device=device)
    print(f"META device={device}", flush=True)
except Exception as e:
    print(f"FATAL {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(3)
wavs = [l.strip() for l in open(list_file) if l.strip()]
for wav in wavs:
    try:
        out = inference(wav)
        data = out.data  # (n_frames, 3) = vad, snr, c50
        vad, snr, c50 = data[:, 0], data[:, 1], data[:, 2]
        voiced = vad > 0.5
        mss = float(np.mean(snr[voiced])) if voiced.any() else float("nan")
        print(f"RES {wav} {mss} {float(np.mean(snr))} {float(np.mean(c50))} {len(data)}",
              flush=True)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")
        print(f"FAILED {wav} {msg}", flush=True)
'''


def brouhaha_available() -> bool:
    """True if the isolated brouhaha venv python imports brouhaha AND the
    local checkpoint exists (we load weights from the repo ckpt, not the gated
    HF hub — see BROUHAHA_CKPT)."""
    if not BROUHAHA_VENV_PY.exists() or not BROUHAHA_CKPT.exists():
        return False
    try:
        r = subprocess.run(
            [str(BROUHAHA_VENV_PY), "-c", "import brouhaha"],
            capture_output=True, timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False


def brouhaha_score_batch(wav_paths: list) -> tuple:
    """Run Brouhaha over ALL given wavs in ONE isolated-venv subprocess (model
    loaded once, GPU if the venv's torch sees CUDA — phase-major, mirroring the
    asr_pipeline convention of one model on GPU at a time).

    Returns ``(results, errors)``:
      - ``results``: {str(wav_path): {brouhaha_snr, brouhaha_snr_all,
        brouhaha_c50, brouhaha_n_frames}} for each scored wav;
      - ``errors``: [(str(wav_path), message)] for per-wav worker failures.

    Raises on whole-subprocess failure (the caller turns that into nan cells
    for every fragment + a loud fallback note, never silent).
    """
    if not wav_paths:
        return {}, []
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_BROUHAHA_WORKER)
        worker = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(p) for p in wav_paths))
        list_file = f.name
    try:
        r = subprocess.run(
            [str(BROUHAHA_VENV_PY), worker, list_file, str(BROUHAHA_CKPT)],
            capture_output=True, text=True,
            timeout=120 + 60 * len(wav_paths),
            env={**os.environ},
        )
    finally:
        os.unlink(worker)
        os.unlink(list_file)
    if r.returncode != 0:
        raise RuntimeError(
            f"brouhaha batch subprocess rc={r.returncode}: {r.stderr.strip()[:300]}"
        )
    results: dict = {}
    errors: list = []
    for line in r.stdout.splitlines():
        if line.startswith("META "):
            print(f"[brouhaha] {line[5:]}", file=sys.stderr)
        elif line.startswith("RES "):
            _, wav, snr_sp, snr_all, c50, n = line.split()
            results[wav] = {
                "brouhaha_snr": float(snr_sp),
                "brouhaha_snr_all": float(snr_all),
                "brouhaha_c50": float(c50),
                "brouhaha_n_frames": int(n),
            }
        elif line.startswith("FAILED "):
            _, wav, msg = line.split(" ", 2)
            errors.append((wav, msg))
    missing = [str(p) for p in wav_paths
               if str(p) not in results and str(p) not in {w for w, _ in errors}]
    for wav in missing:
        errors.append((wav, "no result line from worker (crashed mid-batch?)"))
    return results, errors


def brouhaha_nan() -> dict:
    return {
        "brouhaha_snr": float("nan"),
        "brouhaha_snr_all": float("nan"),
        "brouhaha_c50": float("nan"),
        "brouhaha_n_frames": 0,
    }


# ---------------------------------------------------------------------------
# Per-fragment scoring
# ---------------------------------------------------------------------------

# Order matters: this is the CSV column order (after frag_id).
METRIC_FIELDS = [
    "squim_stoi", "squim_pesq", "squim_si_sdr", "squim_n_chunks",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_n_chunks",
    "brouhaha_snr", "brouhaha_snr_all", "brouhaha_c50", "brouhaha_n_frames",
    "wada_snr", "lufs", "clip_rate",
]


@dataclass
class FragmentFailure:
    frag_id: str
    metric: str
    msg: str


def score_fragment(
    frag_id: str,
    wav_path: Path,
    squim_model,
    squim_device,
    dnsmos_session,
    brouhaha_row: Optional[dict],
    failures: list,
) -> dict:
    """Score one fragment across all metrics. Each metric is independently
    guarded: a failure NaNs that metric, records a FragmentFailure, prints a
    loud stderr line, and lets the others proceed.

    ``brouhaha_row`` is the precomputed result from the phase-major batch
    subprocess (or None — unavailable/failed, already reported by the batch
    step)."""
    row: dict = {"frag_id": frag_id}
    audio = load_mono_16k(wav_path)

    # SQUIM (reuse layer2 helper).
    try:
        sq = squim_chunked(audio, SR, squim_model, squim_device)
        row["squim_stoi"] = sq["squim_stoi"]
        row["squim_pesq"] = sq["squim_pesq"]
        row["squim_si_sdr"] = sq["squim_si_sdr"]
        row["squim_n_chunks"] = sq["n_chunks"]
    except Exception as e:  # noqa: BLE001
        _fail(failures, frag_id, "squim", e)
        row.update({"squim_stoi": float("nan"), "squim_pesq": float("nan"),
                    "squim_si_sdr": float("nan"), "squim_n_chunks": 0})

    # DNSMOS.
    try:
        row.update(dnsmos_windowed(audio, dnsmos_session, SR))
    except Exception as e:  # noqa: BLE001
        _fail(failures, frag_id, "dnsmos", e)
        row.update(_dnsmos_nan())

    # Brouhaha — precomputed by the batch subprocess; None means unavailable
    # or failed (the batch step already reported it loudly).
    row.update(brouhaha_row if brouhaha_row is not None else brouhaha_nan())

    # WADA-SNR (always).
    try:
        row["wada_snr"] = wada_snr(audio)
    except Exception as e:  # noqa: BLE001
        _fail(failures, frag_id, "wada_snr", e)
        row["wada_snr"] = float("nan")

    # LUFS.
    try:
        row["lufs"] = integrated_lufs(audio, SR)
    except Exception as e:  # noqa: BLE001
        _fail(failures, frag_id, "lufs", e)
        row["lufs"] = float("nan")

    # Clipping rate.
    try:
        row["clip_rate"] = clipping_rate(audio)
    except Exception as e:  # noqa: BLE001
        _fail(failures, frag_id, "clip_rate", e)
        row["clip_rate"] = float("nan")

    return row


def _fail(failures: list, frag_id: str, metric: str, exc: Exception) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    print(f"[FAIL] {frag_id} metric={metric}: {msg}", file=sys.stderr)
    failures.append(FragmentFailure(frag_id, metric, msg))


# ---------------------------------------------------------------------------
# Manifest + fragment discovery
# ---------------------------------------------------------------------------


def discover_fragments() -> list:
    """Return sorted frag_ids = on-disk dirs (excluding the bwe-benchmark dir)
    that contain ``<frag_id>.wav``."""
    out = []
    for d in sorted(FRAGMENTS_ROOT.iterdir()):
        if not d.is_dir() or d.name == _EXCLUDE_DIR or d.name.startswith("."):
            continue
        if (d / f"{d.name}.wav").exists():
            out.append(d.name)
    return out


def load_manifest() -> dict:
    """frag_id -> {noise, overlap_bin, Autor}. Rows whose frag_id has no dir
    are kept (harmless); the scorer joins on disk dirs."""
    out = {}
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r.get("frag_id", "")
            if not fid:
                continue
            out[fid] = {
                "noise": r.get("noise", ""),
                "overlap_bin": r.get("overlap_bin", ""),
                "Autor": r.get("Autor", ""),
            }
    return out


def is_dev(frag_id: str, manifest: dict) -> bool:
    autor = manifest.get(frag_id, {}).get("Autor", "")
    return autor.startswith(DEV_AUTOR_PREFIXES)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def write_csv(rows: list, path: Path) -> None:
    fields = ["frag_id"] + METRIC_FIELDS
    rows_sorted = sorted(rows, key=lambda r: r["frag_id"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> dict:
    """frag_id -> row dict (metric values as float, n_* as int, '' -> nan)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            parsed = {"frag_id": r["frag_id"]}
            for k in METRIC_FIELDS:
                v = r.get(k, "")
                if v == "" or v is None:
                    parsed[k] = float("nan")
                elif k.endswith(("_chunks", "_frames")):
                    try:
                        parsed[k] = int(float(v))
                    except ValueError:
                        parsed[k] = 0
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = float("nan")
            out[r["frag_id"]] = parsed
    return out


# ===========================================================================
# Calibration + report (pure analysis on the scored CSV)
# ===========================================================================

# Metrics that go into the correlation analysis (n_chunks/n_frames excluded).
ANALYSIS_METRICS = [
    "squim_stoi", "squim_pesq", "squim_si_sdr",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr",
    "brouhaha_snr", "brouhaha_c50",
    "wada_snr", "lufs", "clip_rate",
]


def spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    """Spearman rho + two-sided p-value on the pairwise-complete rows. Returns
    ``(rho, p, n)``; ``(nan, nan, n)`` when fewer than 3 finite pairs exist or
    a side is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(x[mask], y[mask])
        return float(rho), float(p), n
    except Exception:
        # Pearson-on-ranks fallback (no p-value).
        xr = _rankdata(x[mask])
        yr = _rankdata(y[mask])
        if np.std(xr) == 0 or np.std(yr) == 0:
            return float("nan"), float("nan"), n
        rho = float(np.corrcoef(xr, yr)[0, 1])
        return rho, float("nan"), n


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of `a` (ties shared), matching scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple:
    """Mann-Whitney U two-sided p + the common-language effect size
    (P(A>B), the AUC). Returns ``(auc, p, n_a, n_b)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), na, nb
    try:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = float(u / (na * nb))
        return auc, float(p), na, nb
    except Exception:
        # AUC via rank sum, no p-value.
        allv = np.concatenate([a, b])
        ranks = _rankdata(allv)
        r_a = ranks[:na].sum()
        u = r_a - na * (na + 1) / 2.0
        return float(u / (na * nb)), float("nan"), na, nb


@dataclass
class MetricCorr:
    metric: str
    rho_grade: float
    p_grade: float
    n_grade: int
    auc_wysoki: float  # P(Wysoki harder); >0.5 if metric ranks Wysoki harder
    p_wysoki: float
    direction: int  # +1 if higher==harder, -1 if higher==easier (from grade rho sign)


def _column(scores: dict, frag_ids: list, metric: str) -> np.ndarray:
    return np.array([scores.get(f, {}).get(metric, float("nan")) for f in frag_ids])


def analyse(scores: dict, manifest: dict) -> dict:
    """Run the full calibration analysis. Returns a dict consumed by the
    report writer (and inspectable from a notebook)."""
    graded_ids = [f for f in AUTHOR_GRADES if f in scores]
    grades = np.array([AUTHOR_GRADES[f] for f in graded_ids])

    all_ids = sorted(scores)
    wysoki_ids = [f for f in all_ids if manifest.get(f, {}).get("noise") == "Wysoki"]
    rest_ids = [f for f in all_ids if f not in wysoki_ids]

    corrs: list = []
    for m in ANALYSIS_METRICS:
        col_graded = _column(scores, graded_ids, m)
        rho, p, n = spearman(col_graded, grades)
        # Wysoki-vs-rest: AUC of "metric value is higher in Wysoki".
        auc, pw, _, _ = mann_whitney(
            _column(scores, wysoki_ids, m), _column(scores, rest_ids, m)
        )
        # Direction higher==harder iff grade-rho positive (more grade = harder).
        direction = 1 if (np.isfinite(rho) and rho >= 0) else -1
        corrs.append(MetricCorr(m, rho, p, n, auc, pw, direction))

    # Composite: metrics with |rho| >= 0.5 vs grades, sign-aligned so larger ==
    # harder, z-scored over ALL fragments, averaged.
    strong = [c for c in corrs if np.isfinite(c.rho_grade) and abs(c.rho_grade) >= 0.5]
    composite = None
    if len(strong) >= 2:
        composite = build_composite(scores, all_ids, strong)

    return {
        "graded_ids": graded_ids,
        "grades": grades,
        "all_ids": all_ids,
        "wysoki_ids": wysoki_ids,
        "rest_ids": rest_ids,
        "corrs": corrs,
        "strong": strong,
        "composite": composite,  # None or {frag_id: z-score-mean}
    }


def build_composite(scores: dict, all_ids: list, strong: list) -> dict:
    """z-score each strong metric over all fragments, flip sign so higher ==
    harder (using the grade-rho sign), average across metrics per fragment.

    A metric whose grade-rho is POSITIVE already means higher==harder → keep.
    Negative → flip. NaN cells are mean-imputed per metric before averaging so
    one missing metric doesn't drop a whole fragment.
    """
    z_cols = {}
    for c in strong:
        col = _column(scores, all_ids, c.metric)
        finite = col[np.isfinite(col)]
        mu = float(np.mean(finite)) if finite.size else float("nan")
        sd = float(np.std(finite)) if finite.size else float("nan")
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (np.where(np.isfinite(col), col, mu) - mu) / sd
        # Sign-align: grade-rho > 0 means the raw metric rises with hardness.
        if c.rho_grade < 0:
            z = -z
        z_cols[c.metric] = z
    if not z_cols:
        return {}
    stacked = np.vstack(list(z_cols.values()))
    comp = np.nanmean(stacked, axis=0)
    return {fid: float(comp[i]) for i, fid in enumerate(all_ids)}


# ---------------------------------------------------------------------------
# robione (annotated) parsing — exclude annotated from the "hardest" list
# ---------------------------------------------------------------------------


def load_annotated_ids() -> tuple:
    """Return (set_of_done_ids, note). Missing robione.txt → empty set + note
    (surfaced in the report, never silent)."""
    if not ROBIONE_PATH.exists():
        return set(), f"robione.txt not found at {ROBIONE_PATH} — no fragments excluded as annotated"
    text = ROBIONE_PATH.read_text(encoding="utf-8", errors="replace")
    done = set(parse_robione(text))
    return done, f"parsed {len(done)} annotated (done) fragments from {ROBIONE_PATH}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def build_report(
    scores: dict,
    manifest: dict,
    analysis: dict,
    failures: list,
    fallbacks: list,
    annotated: set,
    annotated_note: str,
) -> str:
    lines: list = []
    L = lines.append
    L("# CLARIN fragment acoustic-complexity report")
    L("")
    L(f"- Fragments scored: **{len(scores)}**")
    L(f"- Graded (calibration target): **{len(analysis['graded_ids'])}** "
      f"(n=16 by-ear grades — small-n; correlations are indicative, not "
      f"confirmatory).")
    L(f"- Wysoki anchors: **{len(analysis['wysoki_ids'])}**")
    L("")

    # --- Fallbacks / failures (surfaced first, per SCOPE no-silent rule) ---
    L("## Metric provenance: what ran vs fell back")
    L("")
    if fallbacks:
        for fb in fallbacks:
            L(f"- {fb}")
    else:
        L("- All configured metrics ran with their primary backend.")
    L("")
    if failures:
        from collections import Counter

        by_metric = Counter(f.metric for f in failures)
        L(f"**Per-fragment failures: {len(failures)}** "
          f"({', '.join(f'{m}={n}' for m, n in sorted(by_metric.items()))}).")
        L("")
        L("| frag_id | metric | error |")
        L("|---|---|---|")
        for f in failures:
            L(f"| {f.frag_id} | {f.metric} | {f.msg[:80]} |")
    else:
        L("**Per-fragment failures: 0.**")
    L("")

    # --- Correlation table ---
    L("## Correlation vs author grades & Wysoki separation")
    L("")
    L("Spearman rho of each metric vs the 16 by-ear grades (higher grade = "
      "harder), and the Mann-Whitney AUC = P(metric value higher on a Wysoki "
      "fragment than on a non-Wysoki one). A metric that tracks hardness should "
      "show |rho| not near 0 and an AUC far from 0.5 (the direction depending "
      "on whether the metric rises or falls with hardness).")
    L("")
    L("| metric | rho vs grade | p | n | AUC Wysoki>rest | p |")
    L("|---|---|---|---|---|---|")
    for c in sorted(analysis["corrs"], key=lambda c: -abs_or_zero(c.rho_grade)):
        L(f"| {c.metric} | {_fmt(c.rho_grade)} | {_fmt(c.p_grade)} | {c.n_grade} "
          f"| {_fmt(c.auc_wysoki)} | {_fmt(c.p_wysoki)} |")
    L("")

    # --- Composite ---
    strong = analysis["strong"]
    composite = analysis["composite"]
    L("## Composite complexity score")
    L("")
    if composite:
        names = ", ".join(f"{c.metric} (rho={_fmt(c.rho_grade,2)})" for c in strong)
        L(f"Composite built from **{len(strong)}** metric(s) with |rho| >= 0.5 "
          f"vs grades: {names}. Each is z-scored over all fragments and "
          f"sign-aligned so **higher = harder**, then averaged.")
        L("")
        # Show composite vs grade on the graded set.
        L("Composite on the 16 graded fragments (sorted hardest first):")
        L("")
        L("| frag_id | grade | composite |")
        L("|---|---|---|")
        graded_sorted = sorted(
            analysis["graded_ids"], key=lambda f: -composite.get(f, float("-inf"))
        )
        for f in graded_sorted:
            L(f"| {f} | {_fmt(AUTHOR_GRADES[f],1)} | {_fmt(composite[f],2)} |")
        # Composite-vs-grade rho.
        g = np.array([AUTHOR_GRADES[f] for f in analysis["graded_ids"]])
        cvals = np.array([composite[f] for f in analysis["graded_ids"]])
        rho, p, n = spearman(cvals, g)
        L("")
        L(f"Composite vs grade: **rho={_fmt(rho)}** (p={_fmt(p)}, n={n}).")
    else:
        L("**Omitted.** Fewer than 2 metrics reach |rho| >= 0.5 against the 16 "
          "grades — no defensible composite. This is itself a finding: at this "
          "n, no single objective metric (or pair) cleanly reproduces the "
          "author's ear, so the objective scores characterise the set but do "
          "not replace by-ear grading.")
    L("")

    # --- Dev vs test + corpus-label cross-tab ---
    L("## Distribution: dev vs test, and corpus noise labels")
    L("")
    key_metric = pick_key_metric(analysis)
    L(f"Key axis below = **{key_metric}** "
      f"({'composite' if key_metric == 'composite' else 'best single metric by |rho|'}).")
    L("")
    dev_ids = [f for f in analysis["all_ids"] if is_dev(f, manifest)]
    test_ids = [f for f in analysis["all_ids"] if not is_dev(f, manifest)]
    keyvals = _key_values(analysis, scores, key_metric)
    L("| split | n | mean | median | min | max |")
    L("|---|---|---|---|---|---|")
    for label, ids in (("dev", dev_ids), ("test", test_ids), ("all", analysis["all_ids"])):
        vals = np.array([keyvals[f] for f in ids if np.isfinite(keyvals.get(f, float("nan")))])
        L(f"| {label} | {len(vals)} | {_fmt(np.mean(vals)) if vals.size else 'nan'} "
          f"| {_fmt(np.median(vals)) if vals.size else 'nan'} "
          f"| {_fmt(np.min(vals)) if vals.size else 'nan'} "
          f"| {_fmt(np.max(vals)) if vals.size else 'nan'} |")
    L("")
    L("Corpus noise label cross-tab (does Niski/Średni actually separate on "
      "the key axis?):")
    L("")
    L("| corpus label | n | mean | median |")
    L("|---|---|---|---|")
    for lab in ["Niski", "Średni", "Sredni", "Wysoki", ""]:
        ids = [f for f in analysis["all_ids"]
               if manifest.get(f, {}).get("noise", "") == lab]
        if not ids:
            continue
        vals = np.array([keyvals[f] for f in ids if np.isfinite(keyvals.get(f, float("nan")))])
        show = lab if lab else "(blank)"
        L(f"| {show} | {len(vals)} | {_fmt(np.mean(vals)) if vals.size else 'nan'} "
          f"| {_fmt(np.median(vals)) if vals.size else 'nan'} |")
    L("")

    # --- 10 hardest un-annotated test fragments ---
    L("## 10 hardest un-annotated TEST fragments")
    L("")
    L(f"_{annotated_note}._")
    L("")
    L("Candidates for a possible step-2 swap: TEST fragments (not in the dev "
      "set) that are NOT already annotated, ranked hardest-first on the key "
      "axis.")
    L("")
    candidates = [
        f for f in test_ids
        if f not in annotated and np.isfinite(keyvals.get(f, float("nan")))
    ]
    candidates.sort(key=lambda f: -keyvals[f])
    L("| rank | frag_id | corpus label | key axis | wada_snr | dnsmos_ovr | brouhaha_c50 |")
    L("|---|---|---|---|---|---|---|")
    for i, f in enumerate(candidates[:10], 1):
        lab = manifest.get(f, {}).get("noise", "") or "(blank)"
        r = scores.get(f, {})
        L(f"| {i} | {f} | {lab} | {_fmt(keyvals[f])} | {_fmt(r.get('wada_snr'))} "
          f"| {_fmt(r.get('dnsmos_ovr'))} | {_fmt(r.get('brouhaha_c50'))} |")
    L("")

    # --- Graded spot-check ---
    L("## Graded fragments vs their metrics (spot-check)")
    L("")
    L("| frag_id | grade | squim_pesq | dnsmos_ovr | dnsmos_bak | wada_snr | brouhaha_snr | brouhaha_c50 | clip_rate |")
    L("|---|---|---|---|---|---|---|---|---|")
    for f in sorted(analysis["graded_ids"], key=lambda f: AUTHOR_GRADES[f]):
        r = scores.get(f, {})
        L(f"| {f} | {_fmt(AUTHOR_GRADES[f],1)} | {_fmt(r.get('squim_pesq'))} "
          f"| {_fmt(r.get('dnsmos_ovr'))} | {_fmt(r.get('dnsmos_bak'))} "
          f"| {_fmt(r.get('wada_snr'))} | {_fmt(r.get('brouhaha_snr'))} "
          f"| {_fmt(r.get('brouhaha_c50'))} | {_fmt(r.get('clip_rate'),4)} |")
    L("")
    return "\n".join(lines)


def abs_or_zero(v: float) -> float:
    return abs(v) if (v is not None and math.isfinite(v)) else 0.0


def pick_key_metric(analysis: dict) -> str:
    """The 'key axis' for the distribution tables: the composite if it exists,
    else the single metric with the largest |rho| vs grades."""
    if analysis["composite"]:
        return "composite"
    best = max(analysis["corrs"], key=lambda c: abs_or_zero(c.rho_grade))
    return best.metric


def _key_values(analysis: dict, scores: dict, key_metric: str) -> dict:
    """Per-fragment value on the key axis, sign-aligned so higher == harder."""
    if key_metric == "composite":
        return dict(analysis["composite"])
    corr = next(c for c in analysis["corrs"] if c.metric == key_metric)
    sign = 1.0 if corr.rho_grade >= 0 else -1.0
    return {
        f: sign * scores.get(f, {}).get(key_metric, float("nan"))
        for f in analysis["all_ids"]
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_scoring(force: bool, do_brouhaha: bool, limit: Optional[int]) -> tuple:
    """Score fragments → write CSV. Returns (rows, failures, fallbacks)."""
    frag_ids = discover_fragments()
    if limit:
        frag_ids = frag_ids[:limit]
    # NB: the manifest is NOT loaded here — scoring keys off frag_id alone and
    # never touched it. It is loaded only in run_report (calibration). This lets
    # --root point at a tree that has no manifest.csv (e.g. a candidate-mining
    # staging tree scored with --no-report).

    # Idempotency: keep existing rows unless --force.
    existing = read_csv(SCORES_CSV) if (SCORES_CSV.exists() and not force) else {}
    todo = [f for f in frag_ids if f not in existing]
    print(f"[score] {len(frag_ids)} fragments; {len(existing)} cached, "
          f"{len(todo)} to compute (force={force}).", file=sys.stderr)

    fallbacks: list = []
    failures: list = []

    # DNSMOS session.
    try:
        dnsmos_session = load_dnsmos_session()
    except Exception as e:  # noqa: BLE001
        dnsmos_session = None
        fallbacks.append(f"DNSMOS unavailable ({type(e).__name__}: {e}) — "
                         f"dnsmos_* cells will be nan.")
        print(f"[WARN] DNSMOS load failed: {e}", file=sys.stderr)

    # Brouhaha availability.
    brouhaha_on = do_brouhaha and brouhaha_available()
    if do_brouhaha and not brouhaha_on:
        fallbacks.append(
            f"Brouhaha unavailable (isolated venv {BROUHAHA_VENV_PY} or ckpt "
            f"{BROUHAHA_CKPT} missing / import failed) — brouhaha_* cells nan; "
            f"WADA-SNR is the SNR axis."
        )
        print("[WARN] brouhaha isolated venv unavailable — falling back to WADA-SNR.",
              file=sys.stderr)
    elif not do_brouhaha:
        fallbacks.append("Brouhaha skipped by --no-brouhaha — brouhaha_* cells nan.")

    # Phase-major: Brouhaha batch FIRST (its own subprocess owns the GPU, model
    # loaded once for all fragments), then SQUIM + CPU metrics per fragment.
    brouhaha_results: dict = {}
    if brouhaha_on and todo:
        wavs = [FRAGMENTS_ROOT / fid / f"{fid}.wav" for fid in todo]
        print(f"[brouhaha] batch-scoring {len(wavs)} wavs in one subprocess...",
              file=sys.stderr)
        try:
            brouhaha_results, brouhaha_errs = brouhaha_score_batch(wavs)
            for wav, msg in brouhaha_errs:
                fid = Path(wav).stem
                _fail(failures, fid, "brouhaha", RuntimeError(msg))
        except Exception as e:  # noqa: BLE001 — whole-batch failure
            fallbacks.append(
                f"Brouhaha batch subprocess failed ({type(e).__name__}: "
                f"{str(e)[:200]}) — brouhaha_* cells nan for this run."
            )
            print(f"[WARN] brouhaha batch failed: {e}", file=sys.stderr)

    rows: list = list(existing.values())
    if todo:
        squim_model, squim_device = load_squim_model()
        try:
            for i, fid in enumerate(todo, 1):
                wav = FRAGMENTS_ROOT / fid / f"{fid}.wav"
                try:
                    row = score_fragment(
                        fid, wav, squim_model, squim_device, dnsmos_session,
                        brouhaha_results.get(str(wav)), failures,
                    )
                    rows.append(row)
                except Exception as e:  # noqa: BLE001 — whole-fragment failure
                    _fail(failures, fid, "fragment", e)
                    rows.append({"frag_id": fid,
                                 **{k: float("nan") for k in METRIC_FIELDS}})
                # Incremental write: a killed run loses at most one fragment
                # (cached rows are skipped on the next run unless --force).
                write_csv(rows, SCORES_CSV)
                if i % 10 == 0 or i == len(todo):
                    print(f"[score] {i}/{len(todo)}", file=sys.stderr)
        finally:
            unload_squim_model(squim_model)

    write_csv(rows, SCORES_CSV)
    print(f"[score] wrote {SCORES_CSV} ({len(rows)} rows).", file=sys.stderr)
    return rows, failures, fallbacks


def run_report(failures: list, fallbacks: list) -> None:
    scores = read_csv(SCORES_CSV)
    manifest = load_manifest()
    annotated, annotated_note = load_annotated_ids()
    analysis = analyse(scores, manifest)
    report = build_report(
        scores, manifest, analysis, failures, fallbacks, annotated, annotated_note
    )
    REPORT_MD.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] wrote {REPORT_MD}", file=sys.stderr)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="recompute all fragments (ignore cached CSV rows)")
    ap.add_argument("--report", action="store_true",
                    help="only (re)build the report from the cached CSV")
    ap.add_argument("--no-brouhaha", action="store_true",
                    help="skip the slow brouhaha subprocess stage")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N fragments (debug)")
    ap.add_argument("--root", type=Path, default=None,
                    help="score a DIFFERENT fragment tree than the default eval "
                         "set (e.g. a candidate-mining staging tree). The tree "
                         "must still have the <root>/<frag_id>/<frag_id>.wav "
                         "layout. SCORES_CSV/REPORT_MD default under it unless "
                         "--csv-out overrides. Default behaviour (no flag) is "
                         "byte-identical to before.")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="write the scores CSV here instead of "
                         "<root>/acoustic_scores.csv. Use with --root for a "
                         "candidate run whose CSV must not collide with the "
                         "eval set's.")
    ap.add_argument("--no-report", action="store_true",
                    help="score only — skip the calibration report. Use for a "
                         "non-eval tree (e.g. candidate mining) where the "
                         "grade/manifest calibration assumptions do not hold; "
                         "the report's composite is re-normalised over whatever "
                         "set it is given, so it is NOT comparable across trees.")
    args = ap.parse_args(argv)

    # --root / --csv-out override the module path globals BEFORE any scoring or
    # discovery runs (every function reads these at call time). With no flag,
    # the globals keep their default values and behaviour is unchanged.
    global FRAGMENTS_ROOT, MANIFEST_PATH, SCORES_CSV, REPORT_MD
    if args.root is not None:
        FRAGMENTS_ROOT = args.root
        MANIFEST_PATH = FRAGMENTS_ROOT / "manifest.csv"
        SCORES_CSV = FRAGMENTS_ROOT / "acoustic_scores.csv"
        REPORT_MD = FRAGMENTS_ROOT / "ACOUSTIC_SCORES_REPORT.md"
    if args.csv_out is not None:
        SCORES_CSV = args.csv_out

    if args.report:
        run_report([], ["(report-only run — fallbacks from the scoring run not "
                         "re-derived here)"])
        return 0

    rows, failures, fallbacks = run_scoring(
        force=args.force, do_brouhaha=not args.no_brouhaha, limit=args.limit
    )
    if args.no_report:
        print(f"[score] --no-report: wrote {SCORES_CSV} ({len(rows)} rows), "
              f"skipped calibration report.", file=sys.stderr)
        return 0
    run_report(failures, fallbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
