"""Layer 2 — audio quality metrics.

Two complementary halves, both run on the pipeline's per-speaker streams:

- **Intrusive** (requires oracle audio): SI-SDR (closed-form on the whole
  stream, NaN-skipped when the oracle target is essentially silent) +
  PESQ-WB / STOI (chunked over 8-second windows, median aggregate, with a
  speech-presence filter so silence-vs-silence frames don't inflate the
  score).
- **Non-intrusive** (no reference needed): SQUIM_OBJECTIVE on 30-second
  windows, mean aggregate. Always available — used on the mixture and
  on each pipeline stream.

The chunking is load-bearing: ITU P.862 (PESQ) is undefined on long
clips, and SQUIM's transformer attention is O(N²) — a 10-minute recording
in one shot OOMs the GPU. The notebook's previous version of this code
hit both walls on real CLARIN recordings.

Alignment between pipeline streams and oracle audio is the caller's
problem. The intrusive metrics assume the arrays are time-aligned; the
est-vs-target arm truncates to the shorter of estimate/target, and the
mixture-baseline arm additionally truncates to the mix length.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from torchmetrics.functional.audio import (
    perceptual_evaluation_speech_quality,
    scale_invariant_signal_distortion_ratio,
    short_time_objective_intelligibility,
)

from asr_pipeline.eval.recordings import Recording


# |sample| at/below this counts as silence — the one FLOOR shared by every
# silence check here (the per-chunk speech-presence filters and the whole-stream
# SI-SDR guard). The fraction THRESHOLDS below differ; the floor is one value.
_SILENCE_FLOOR = 1e-4

# Per-chunk speech-presence fraction: a PESQ/STOI/SQUIM chunk is scored only if
# at least this fraction of its samples exceed _SILENCE_FLOOR, so a
# silence-vs-silence window doesn't drag the median.
_MIN_SPEECH_FRAC = 0.30

# Whole-stream SI-SDR guard: an oracle TARGET with fewer than this fraction of
# non-silent samples is unusable as a reference, so SI-SDR is NaN-skipped rather
# than reported as a fabricated floor value. Deliberately far below
# _MIN_SPEECH_FRAC (see _target_essentially_silent for the rationale).
_ESSENTIALLY_SILENT_FRAC = 0.01

# Minimum chunk length to score. PESQ/STOI tolerate sub-second windows; SQUIM's
# transformer needs >= 1 s to be defined.
_MIN_INTRUSIVE_CHUNK_S = 0.5
_MIN_SQUIM_CHUNK_S = 1.0


# ---------------------------------------------------------------------------
# Shared chunk iterator
# ---------------------------------------------------------------------------


def _speech_fraction(chunk) -> float:
    """Fraction of `chunk` samples above `_SILENCE_FLOOR`. Accepts a torch
    tensor or a numpy array (the three metric loops mix both)."""
    if isinstance(chunk, torch.Tensor):
        return (chunk.abs() > _SILENCE_FLOOR).float().mean().item()
    return float((np.abs(chunk) > _SILENCE_FLOOR).mean())


def _iter_speech_chunks(presence, payloads, sr, chunk_s, min_chunk_s, min_speech_frac):
    """Window `presence` (+ aligned `payloads`) and yield only chunks that are
    long enough AND speech-present.

    The single windowing/speech-presence iterator shared by the three chunked
    metric loops (PESQ, STOI, SQUIM) so they make byte-identical chunk
    decisions (E2). `presence` drives both the length gate and the
    speech-presence gate (it is the reference channel for the intrusive
    metrics, the audio itself for SQUIM). `payloads` are arrays sliced on the
    same boundaries (e.g. the estimate channel); they are NOT tested for
    presence — they ride along with the presence signal's decision.

    Yields ``(reason, presence_chunk, payload_chunks)`` for every window:
      - ``reason`` is ``None`` for a kept chunk, else ``"short"`` / ``"silent"``
        so a caller can tally skips without re-deriving the predicate.
      - skipped chunks still yield (with empty payload list) so the counts stay
        exact; callers filter on ``reason is None``.

    Chunk boundaries, the ``len < min_n`` short-gate, and the
    ``fraction < min_speech_frac`` silent-gate reproduce the pre-E2 per-loop
    logic exactly.
    """
    chunk_n = int(chunk_s * sr)
    min_n = max(int(min_chunk_s * sr), 1)
    for i in range(0, len(presence), chunk_n):
        p_c = presence[i:i + chunk_n]
        if len(p_c) < min_n:
            yield "short", p_c, []
            continue
        if _speech_fraction(p_c) < min_speech_frac:
            yield "silent", p_c, []
            continue
        yield None, p_c, [pl[i:i + chunk_n] for pl in payloads]


# ---------------------------------------------------------------------------
# Chunked intrusive metrics
# ---------------------------------------------------------------------------


def pesq_wb_chunked(
    est_t: torch.Tensor,
    ref_t: torch.Tensor,
    sr: int,
    chunk_s: float = 8.0,
    min_speech_frac: float = _MIN_SPEECH_FRAC,
) -> dict:
    """PESQ-WB on `chunk_s` windows where the ref is ≥ `min_speech_frac` non-silent.

    Aggregates by median. Returns `{median, n_scored, n_skipped_silent,
    n_errored}` (the chunker tallies; `n_skipped_short` was trimmed — read by
    nobody, E3).
    """
    vals: list[float] = []
    skipped_silent = errored = 0
    for reason, _r_c, payloads in _iter_speech_chunks(
        ref_t, [est_t], sr, chunk_s, _MIN_INTRUSIVE_CHUNK_S, min_speech_frac,
    ):
        if reason == "silent":
            skipped_silent += 1
            continue
        if reason is not None:        # "short"
            continue
        r_c, (e_c,) = _r_c, payloads
        try:
            v = perceptual_evaluation_speech_quality(e_c, r_c, sr, "wb").item()
            vals.append(v)
        except Exception:
            errored += 1
            continue
    return {
        "median": float(np.median(vals)) if vals else float("nan"),
        "n_scored": len(vals),
        "n_skipped_silent": skipped_silent,
        "n_errored": errored,
    }


def stoi_chunked(
    est_t: torch.Tensor,
    ref_t: torch.Tensor,
    sr: int,
    chunk_s: float = 8.0,
    min_speech_frac: float = _MIN_SPEECH_FRAC,
) -> float:
    """STOI on `chunk_s` windows. Same speech-presence filter as PESQ (same
    shared chunk iterator) so the two metrics are scored over the same
    regions. Aggregates by median.
    """
    vals: list[float] = []
    for reason, r_c, payloads in _iter_speech_chunks(
        ref_t, [est_t], sr, chunk_s, _MIN_INTRUSIVE_CHUNK_S, min_speech_frac,
    ):
        if reason is not None:
            continue
        (e_c,) = payloads
        try:
            vals.append(short_time_objective_intelligibility(e_c, r_c, sr).item())
        except Exception:
            continue
    return float(np.median(vals)) if vals else float("nan")


def _align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _essentially_silent(x: torch.Tensor) -> bool:
    """True when fewer than `_ESSENTIALLY_SILENT_FRAC` of `x`'s samples exceed
    `_SILENCE_FLOOR` (empty counts as silent). Deliberately far below the
    per-chunk `_MIN_SPEECH_FRAC`: this catches a degenerate ~all-silent signal,
    not a legitimately low-energy one. Shared by the silent-target and
    silent-estimate guards in `compute_intrusive`.
    """
    if len(x) == 0:
        return True
    return float((x.abs() > _SILENCE_FLOOR).float().mean()) < _ESSENTIALLY_SILENT_FRAC


def _target_essentially_silent(t: torch.Tensor) -> bool:
    """The oracle target is unusable as an SI-SDR reference (empty / near-silent
    debleed channel for a speaker who barely talks). See `_essentially_silent`.
    """
    return _essentially_silent(t)


def _dlog_silent_estimate(label: Optional[str]) -> None:
    """Visibly flag a silent estimate against a non-silent target (E4 /
    SCOPE §4.3) — a dropped speaker. Routed through `dlog` (stdout + the debug
    log), naming the stream when the caller passed a label."""
    from asr_pipeline.debug_log import dlog

    where = f" [{label}]" if label else ""
    dlog(
        "eval-layer2",
        f"silent estimate vs non-silent target{where}: si_sdr floored to "
        f"{_SILENCE_FLOOR_DB} (dropped speaker, not NaN-skipped)",
    )


def _dlog_nonfinite_estimate(label: Optional[str]) -> None:
    """Visibly flag a non-finite estimate (NaN/Inf from an upstream separator/BWE
    output) before it is NaN-skipped (E5 / SCOPE §4.3). Routed through `dlog`
    (stdout + the debug log) so the drop is no longer silent."""
    from asr_pipeline.debug_log import dlog

    where = f" [{label}]" if label else ""
    dlog(
        "eval-layer2",
        f"non-finite estimate (NaN/Inf) vs target{where}: intrusive metrics "
        "NaN-skipped for this stream (corrupt input, not silently dropped)",
    )


_SILENCE_FLOOR_DB = 0.0  # si_sdr score charged to a silent estimate (E4)


def compute_intrusive(
    estimate: np.ndarray, target: np.ndarray, mix: np.ndarray, sr: int,
    label: Optional[str] = None,
) -> dict:
    """SI-SDR (whole stream) + PESQ-WB / STOI (chunked) + improvement vs
    the mono-mix baseline.

    Caller is responsible for ensuring `estimate`, `target`, `mix` are 1-D
    float32 numpy arrays at the same sample rate. The est-vs-target arm
    truncates to the shorter of estimate/target; the mixture-baseline arm
    additionally truncates to the mix length (the mix is irrelevant to the
    est-vs-target metric, so its length must not clip it).

    `label` (e.g. ``"<rec_id>/A"``) only names the stream in the silent-estimate
    warning (E4); scoring is identical with or without it.

    Two distinct silence guards (do NOT conflate them):
      - **silent TARGET** → SI-SDR NaN-skipped: the oracle reference is unusable,
        so the row carries no information and must not be aggregated.
      - **silent ESTIMATE** against a non-silent target → SI-SDR floored to
        ``_SILENCE_FLOOR_DB`` (0.0), NOT NaN: the separator dropping a speaker
        who actually spoke is a *real, bad* result L2 exists to catch; NaN-ing
        it would silently hide a separation failure. Flagged visibly (SCOPE
        §4.3) via ``dlog`` naming the stream.
    """
    # Est-vs-target arm: mixture-independent — do NOT truncate to the mix.
    n_et = min(len(estimate), len(target))
    e = torch.from_numpy(estimate[:n_et].astype(np.float32))
    t = torch.from_numpy(target[:n_et].astype(np.float32))
    # Baseline arm: mono mix vs target, truncated to the mix length here only.
    n_base = min(n_et, len(mix))
    t_base = t[:n_base]
    m = torch.from_numpy(mix[:n_base].astype(np.float32))

    # E5 (resolved): a non-finite sample in `e` (upstream NaN/Inf from a
    # separator/BWE output) would flow into SI-SDR + the PESQ/STOI arms and emit
    # NaN/garbage that nan-aware aggregation then dropped SILENTLY (SCOPE §4). We
    # now DETECT it, flag it visibly, and NaN-skip the whole intrusive row (a
    # corrupt estimate carries no usable metric) — the drop is logged, not silent.
    # Finite inputs are untouched. We do not hard-crash eval on one bad sample.
    if not torch.isfinite(e).all():
        _dlog_nonfinite_estimate(label)
        nan = float("nan")
        return {
            "si_sdr": nan, "si_sdr_baseline": nan, "si_sdri": nan,
            "pesq": nan, "pesq_baseline": nan, "pesqi": nan,
            "stoi": nan, "stoi_baseline": nan, "stoii": nan,
            "pesq_n_scored": 0,
        }
    if _target_essentially_silent(t):
        # A near-silent target sends both SI-SDR arms to the EPS floor; their
        # difference is a fabricated, stable, positive si_sdri (~+15 dB) that
        # would inflate the reported mean. NaN-skip, matching the chunked arms.
        si = si0 = float("nan")
    elif _essentially_silent(e):
        # Silent estimate vs a real (non-silent) target: a dropped speaker.
        # Floor SI-SDR rather than NaN-skip so the failure is scored, and warn
        # so it is not silent in the logs (SCOPE §4.3).
        _dlog_silent_estimate(label)
        si = _SILENCE_FLOOR_DB
        si0 = scale_invariant_signal_distortion_ratio(m, t_base).item()
    else:
        si = scale_invariant_signal_distortion_ratio(e, t).item()
        si0 = scale_invariant_signal_distortion_ratio(m, t_base).item()
    pq = pesq_wb_chunked(e, t, sr)
    pq0 = pesq_wb_chunked(m, t_base, sr)
    st = stoi_chunked(e, t, sr)
    st0 = stoi_chunked(m, t_base, sr)

    # E3: `pesq_n_skipped_silent` / `pesq_n_errored` were forwarded here but
    # read by nobody (summary.py uses only `pesq_n_scored`) — dropped. The
    # chunker still tallies them on its own dict for the layer2 test.
    return {
        "si_sdr": si, "si_sdr_baseline": si0, "si_sdri": si - si0,
        "pesq": pq["median"], "pesq_baseline": pq0["median"],
        "pesqi": pq["median"] - pq0["median"],
        "stoi": st, "stoi_baseline": st0, "stoii": st - st0,
        "pesq_n_scored": pq["n_scored"],
    }


# ---------------------------------------------------------------------------
# Non-intrusive SQUIM
# ---------------------------------------------------------------------------


def squim_chunked(
    audio: np.ndarray,
    sr: int,
    squim_model,
    device,
    chunk_s: float = 30.0,
    min_speech_frac: float = _MIN_SPEECH_FRAC,
) -> dict:
    """SQUIM_OBJECTIVE on `chunk_s` windows, mean aggregate.

    Caller passes a loaded ``squim_model`` and target ``device`` (eval-side
    keeps the model alive across multiple recordings — loading is
    expensive). Near-silent chunks are skipped because SQUIM is undefined
    on silence.
    """
    stoi_vals: list[float] = []
    pesq_vals: list[float] = []
    sisdr_vals: list[float] = []
    with torch.no_grad():
        for reason, c, _ in _iter_speech_chunks(
            audio, [], sr, chunk_s, _MIN_SQUIM_CHUNK_S, min_speech_frac,
        ):
            if reason is not None:        # too short (< 1 s) or silent
                continue
            x = torch.from_numpy(c.astype(np.float32)).unsqueeze(0).to(device)
            stoi_p, pesq_p, sisdr_p = squim_model(x)
            stoi_vals.append(float(stoi_p.item()))
            pesq_vals.append(float(pesq_p.item()))
            sisdr_vals.append(float(sisdr_p.item()))
    if not stoi_vals:
        return {
            "squim_stoi": float("nan"), "squim_pesq": float("nan"),
            "squim_si_sdr": float("nan"), "n_chunks": 0,
        }
    return {
        "squim_stoi": float(np.mean(stoi_vals)),
        "squim_pesq": float(np.mean(pesq_vals)),
        "squim_si_sdr": float(np.mean(sisdr_vals)),
        "n_chunks": len(stoi_vals),
    }


def load_squim_model(device: Optional[str] = None):
    """Load SQUIM_OBJECTIVE on the requested device. Caller is responsible
    for freeing it via :func:`unload_squim_model`.
    """
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SQUIM_OBJECTIVE.get_model().eval().to(device)
    return model, torch.device(device)


def unload_squim_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------------


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    """Read a WAV and return mono float32 at `target_sr` (assume already
    at that rate — we only check; resample is a separate concern)."""
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        raise ValueError(
            f"{path}: sample rate {sr} != expected {target_sr}; "
            "use load_audio_as_mono if resampling is needed."
        )
    return arr


def compute_layer2(
    rec: Recording,
    sr: int = 16_000,
    squim_model=None,
    squim_device=None,
) -> Optional[dict]:
    """Run Layer 2 for one recording.

    ``squim_model`` / ``squim_device`` should be supplied by the caller
    when batching across recordings (load once, reuse). When None, this
    function loads + unloads the model itself — fine for one-shot use,
    wasteful for a sweep.

    Returns ``{"intrusive": {…}|None, "squim": {…}}``:
      - ``intrusive`` is per-speaker: ``{"A": {...}, "B": {...}}`` only
        when ``rec.reference_audio`` is populated.
      - ``squim`` always present: ``{"mixture": {...}, "stream_A": {...},
        "stream_B": {...}}``.

    Returns None when the pipeline hasn't been run for this recording.
    """
    if rec.pipeline_dir is None:
        return None

    stream_paths = {
        "A": rec.pipeline_dir / "stream_A.wav",
        "B": rec.pipeline_dir / "stream_B.wav",
    }
    for label, p in stream_paths.items():
        if not p.exists():
            return None

    streams = {label: _load_mono(p, sr) for label, p in stream_paths.items()}
    mixture = _load_mono(rec.mixture_path, sr)

    # Intrusive only when oracles are present.
    intrusive_out: Optional[dict] = None
    if rec.reference_audio:
        intrusive_out = {}
        for label, est in streams.items():
            ref_path = rec.reference_audio.get(label)
            if ref_path is None or not ref_path.exists():
                continue
            target = _load_mono(ref_path, sr)
            est_n, tgt_n = _align_lengths(est, target)
            _, mix_n = _align_lengths(target, mixture)
            intrusive_out[label] = compute_intrusive(
                est_n, tgt_n, mix_n[:len(tgt_n)], sr, label=f"{rec.id}/{label}",
            )

    # SQUIM (non-intrusive) — always.
    owned_model = squim_model is None
    if owned_model:
        squim_model, squim_device = load_squim_model()
    try:
        squim_out = {
            "mixture": squim_chunked(mixture, sr, squim_model, squim_device),
            "stream_A": squim_chunked(streams["A"], sr, squim_model, squim_device),
            "stream_B": squim_chunked(streams["B"], sr, squim_model, squim_device),
        }
    finally:
        if owned_model:
            unload_squim_model(squim_model)

    return {"intrusive": intrusive_out, "squim": squim_out}
