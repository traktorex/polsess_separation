"""Tests for Layer-2 audio-quality scoring (asr_pipeline/eval/layer2.py).

Covers compute_intrusive's silence/empty guard (the headline fix: a near-silent
oracle target must NOT fabricate a positive si_sdri), the mix-length-independence
of the est-vs-target arm, the per-chunk speech-presence filter, and the SQUIM
chunker's aggregation — all on CPU. The heavy SQUIM model is never loaded (a fake
callable stands in); the PESQ/STOI backends (pesq, pystoi) are guarded with
importorskip, matching tests/test_pipeline_eval_metrics.py. SI-SDR-only tests
need no guard — SI-SDR has no extra backend and degrades gracefully if PESQ/STOI
are absent.
"""

import numpy as np
import pytest
import torch

from asr_pipeline.eval.layer2 import (
    _ESSENTIALLY_SILENT_FRAC,
    _MIN_INTRUSIVE_CHUNK_S,
    _MIN_SPEECH_FRAC,
    _SILENCE_FLOOR,
    _iter_speech_chunks,
    compute_intrusive,
    pesq_wb_chunked,
    squim_chunked,
    stoi_chunked,
)

SR = 16_000


def _noise(n: int, seed: int = 0, scale: float = 0.1) -> np.ndarray:
    return (np.random.RandomState(seed).standard_normal(n) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_intrusive — the silence guard (headline fix)
# ---------------------------------------------------------------------------


def test_compute_intrusive_silent_target_yields_nan_sisdri():
    # Pre-fix, a silent oracle target sent both SI-SDR arms to the EPS floor and
    # produced a fabricated, stable si_sdri ~= +15.56 dB that inflated the
    # Chapter-5 mean. It must now be NaN so nan-aware aggregation skips the row.
    loud = _noise(SR * 30)
    out = compute_intrusive(loud, np.zeros(SR * 30, np.float32), loud, SR)
    assert np.isnan(out["si_sdr"])
    assert np.isnan(out["si_sdr_baseline"])
    assert np.isnan(out["si_sdri"])


def test_compute_intrusive_near_silent_target_below_floor_is_nan():
    # Every sample below _SILENCE_FLOOR (amplitude 1e-6), not literal zeros.
    loud = _noise(SR * 30)
    near_silent = _noise(SR * 30, seed=1, scale=1e-6)
    assert np.isnan(compute_intrusive(loud, near_silent, loud, SR)["si_sdri"])


def test_compute_intrusive_low_talk_target_still_scored():
    # Anti-over-skip lock: a speaker who talks ~25% of a recording (whole-stream
    # non-silent fraction well above _ESSENTIALLY_SILENT_FRAC) must STILL be
    # scored — NOT NaN-skipped like the degenerate ~all-silent channel. Guards
    # against a too-aggressive whole-stream threshold silently regressing in.
    loud = _noise(SR * 30)
    low_talk = np.zeros(SR * 30, np.float32)
    low_talk[: int(SR * 30 * 0.25)] = _noise(int(SR * 30 * 0.25), seed=2)
    frac = float((np.abs(low_talk) > 1e-4).mean())
    assert frac > _ESSENTIALLY_SILENT_FRAC               # precondition: not "silent"
    assert np.isfinite(compute_intrusive(loud, low_talk, loud, SR)["si_sdri"])


def test_compute_intrusive_empty_input_all_nan():
    z = np.zeros(0, np.float32)
    out = compute_intrusive(z, z, z, SR)
    assert np.isnan(out["si_sdr"])
    assert np.isnan(out["si_sdri"])


def test_compute_intrusive_est_vs_target_arm_is_mix_length_independent():
    # Bug #3: the est-vs-target metric is mixture-independent, so a shorter mix
    # must NOT clip it. si_sdr is identical for a short vs full mix; only the
    # baseline arm (mix vs target) may differ.
    target = _noise(SR * 4, seed=1)
    estimate = target + _noise(SR * 4, seed=3, scale=0.02)   # imperfect → finite si_sdr
    mix_full = _noise(SR * 4, seed=4)
    out_full = compute_intrusive(estimate, target, mix_full, SR)
    out_short = compute_intrusive(estimate, target, mix_full[: SR * 2], SR)
    assert out_full["si_sdr"] == out_short["si_sdr"]                    # arm not clipped to mix
    assert out_full["si_sdr_baseline"] != out_short["si_sdr_baseline"]  # baseline uses mix


def test_compute_intrusive_identical_streams_zero_improvement():
    # est == target == mix is the true no-op: every improvement delta is 0 and
    # STOI is perfect. The absolute SI-SDR is an EPS-clamp ceiling that is
    # length-dependent (~91/94/101 dB @ 1/2/10 s), so only BOUND it, never pin.
    pytest.importorskip("pesq")
    pytest.importorskip("pystoi")
    one = _noise(SR * 10, seed=5)
    out = compute_intrusive(one, one, one, SR)
    assert out["si_sdri"] == pytest.approx(0.0, abs=1e-6)
    assert out["pesqi"] == pytest.approx(0.0, abs=1e-6)
    assert out["stoii"] == pytest.approx(0.0, abs=1e-6)
    assert out["stoi"] == pytest.approx(1.0, abs=1e-6)
    assert out["si_sdr"] > 100                            # bounded, not pinned (EPS ceiling)


# ---------------------------------------------------------------------------
# Per-chunk speech-presence filter
# ---------------------------------------------------------------------------


def _chunk_frac_loud(frac: float, n_s: int = 8) -> torch.Tensor:
    a = np.zeros(SR * n_s, np.float32)
    a[: int(SR * n_s * frac)] = 0.1        # constant, all clearly above the floor
    return torch.from_numpy(a)


def test_pesq_chunked_skips_mostly_silent_chunk():
    # The filter is active: a 50%-speech chunk is classified (scored/errored,
    # not skipped); a 10%-speech chunk is skipped as silent.
    pytest.importorskip("pesq")
    assert pesq_wb_chunked(_chunk_frac_loud(0.50), _chunk_frac_loud(0.50), SR)["n_skipped_silent"] == 0
    assert pesq_wb_chunked(_chunk_frac_loud(0.10), _chunk_frac_loud(0.10), SR)["n_skipped_silent"] == 1


def test_stoi_chunked_all_silent_is_nan():
    pytest.importorskip("pystoi")
    sil = torch.zeros(SR * 8)
    assert np.isnan(stoi_chunked(sil, sil, SR))


# ---------------------------------------------------------------------------
# E2 — shared chunk iterator reproduces the old per-loop decisions exactly
# ---------------------------------------------------------------------------


def _old_kept_boundaries(presence, sr, chunk_s, min_chunk_s, min_speech_frac):
    """The pre-E2 inline windowing/filter, reimplemented here as the oracle:
    returns the start indices of the chunks the old loops would have SCORED."""
    chunk_n = int(chunk_s * sr)
    min_n = max(int(min_chunk_s * sr), 1)
    kept = []
    for i in range(0, len(presence), chunk_n):
        c = presence[i:i + chunk_n]
        if len(c) < min_n:
            continue
        if isinstance(c, torch.Tensor):
            frac = (c.abs() > _SILENCE_FLOOR).float().mean().item()
        else:
            frac = float((np.abs(c) > _SILENCE_FLOOR).mean())
        if frac < min_speech_frac:
            continue
        kept.append(i)
    return kept


def test_iter_speech_chunks_matches_old_decisions_on_mixed_signal():
    # A signal with loud, silent, and a final short ragged chunk — the three
    # branches (kept / silent / short). New iterator's kept boundaries AND its
    # payload slices must equal the old inline logic's, byte-for-byte.
    rng = np.random.RandomState(7)
    n = int(SR * 30.5)                       # 3 full 8 s chunks + part + ragged tail
    presence = np.zeros(n, np.float32)
    presence[: SR * 8] = rng.standard_normal(SR * 8) * 0.1      # chunk 0: loud
    # chunk 1 (8..16 s): left silent
    presence[SR * 16: SR * 24] = rng.standard_normal(SR * 8) * 0.1   # chunk 2: loud
    presence[SR * 24: SR * 30] = (rng.standard_normal(SR * 6) * 0.1).astype(np.float32)  # chunk 3 loud-ish
    payload = (presence * 2.0).astype(np.float32)               # arbitrary aligned rider
    pt, payt = torch.from_numpy(presence), torch.from_numpy(payload)

    expected = _old_kept_boundaries(pt, SR, 8.0, _MIN_INTRUSIVE_CHUNK_S, _MIN_SPEECH_FRAC)
    kept_starts = []
    cursor = 0
    chunk_n = int(8.0 * SR)
    for reason, p_c, payloads in _iter_speech_chunks(
        pt, [payt], SR, 8.0, _MIN_INTRUSIVE_CHUNK_S, _MIN_SPEECH_FRAC,
    ):
        if reason is None:
            kept_starts.append(cursor)
            # payload slice must be exactly payload[cursor:cursor+chunk_n]
            assert torch.equal(payloads[0], payt[cursor:cursor + chunk_n])
            # presence slice identity too
            assert torch.equal(p_c, pt[cursor:cursor + chunk_n])
        cursor += chunk_n
    assert kept_starts == expected


# ---------------------------------------------------------------------------
# E4 — silent-estimate guard (distinct from the silent-target NaN-skip)
# ---------------------------------------------------------------------------


def test_compute_intrusive_silent_estimate_scored_not_nan():
    # A real (non-silent) target but a ~silent estimate = a dropped speaker.
    # MUST be SCORED (si_sdr floored to 0.0), NOT NaN-skipped like a silent
    # target — NaN-ing it would hide a separation failure.
    target = _noise(SR * 30)
    silent_est = np.zeros(SR * 30, np.float32)
    out = compute_intrusive(silent_est, target, target, SR)
    assert out["si_sdr"] == 0.0
    assert not np.isnan(out["si_sdr"])
    assert np.isfinite(out["si_sdr_baseline"])   # baseline arm still real


def test_compute_intrusive_silent_estimate_is_flagged(capsys):
    # SCOPE §4.3: the dropped-speaker case must be VISIBLE (dlog → stdout),
    # naming the stream.
    target = _noise(SR * 30)
    silent_est = np.zeros(SR * 30, np.float32)
    compute_intrusive(silent_est, target, target, SR, label="rec42/A")
    captured = capsys.readouterr()
    blob = captured.out + captured.err
    assert "rec42/A" in blob
    assert "silent estimate" in blob.lower()


# ---------------------------------------------------------------------------
# SQUIM chunker (fake model — never loads the real SQUIM)
# ---------------------------------------------------------------------------


def test_squim_chunked_averages_over_speech_chunks():
    # Fake SQUIM returns constant (stoi, pesq, sisdr) per chunk; assert the mean
    # aggregate and that it ran once per non-silent 30 s chunk.
    def fake_squim(x):
        return (torch.tensor(0.9), torch.tensor(2.5), torch.tensor(12.0))

    out = squim_chunked(_noise(SR * 90), SR, fake_squim, torch.device("cpu"))
    assert out["n_chunks"] == 3
    assert out["squim_stoi"] == pytest.approx(0.9)
    assert out["squim_pesq"] == pytest.approx(2.5)
    assert out["squim_si_sdr"] == pytest.approx(12.0)


def test_squim_chunked_all_silent_returns_nan_without_calling_model():
    calls = []

    def fake_squim(x):
        calls.append(1)
        return (torch.tensor(0.9), torch.tensor(2.5), torch.tensor(12.0))

    out = squim_chunked(np.zeros(SR * 90, np.float32), SR, fake_squim, torch.device("cpu"))
    assert out["n_chunks"] == 0
    assert np.isnan(out["squim_stoi"])
    assert calls == []                     # silent chunks never hit the model


# ---------------------------------------------------------------------------
# E6 — real SQUIM load/unload smoke (weights/network-dependent → optional)
# ---------------------------------------------------------------------------


def _load_real_squim_or_skip():
    """Load the real SQUIM_OBJECTIVE on CPU, or skip when the weights aren't
    cached / no network — so a cold `pytest -k pipeline` never fails here."""
    pytest.importorskip("torchaudio")
    from asr_pipeline.eval.layer2 import load_squim_model
    try:
        return load_squim_model(device="cpu")
    except Exception as exc:                 # download failure, offline, etc.
        pytest.skip(f"SQUIM weights unavailable ({type(exc).__name__}): {exc}")


def test_load_squim_model_real_forward_and_unload():
    # The fake-model tests cover control flow; this one exercises the actual
    # weight load + a forward + clean unload. Optional: skips without weights.
    from asr_pipeline.eval.layer2 import unload_squim_model

    model, device = _load_real_squim_or_skip()
    try:
        out = squim_chunked(_noise(SR * 5), SR, model, device)
        assert out["n_chunks"] >= 1
        for k in ("squim_stoi", "squim_pesq", "squim_si_sdr"):
            assert np.isfinite(out[k])
    finally:
        unload_squim_model(model)            # must not raise
