"""Unit tests for the shared Hann overlap-add helper in Stage 3a.

`_hann_overlap_add` replaced two near-identical chunking loops (MP-SENet's
`_enhance_chunked` and ClearerVoice's `_enhance_native`). The regression
tests at the bottom keep verbatim copies of both *original* loops and assert
the helper reproduces them exactly — so any future change to the COLA math
that would alter the audio fails loudly here.
"""

import numpy as np
import pytest

from asr_pipeline.stages.enhancement import _hann_overlap_add

RNG = np.random.default_rng(0)


def _noise(n: int) -> np.ndarray:
    return RNG.standard_normal(n).astype(np.float32)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_single_chunk_is_one_call():
    x = _noise(100)
    out = _hann_overlap_add(x, window_n=200, process_chunk=lambda s: s * 2.0)
    assert np.allclose(out, x * 2.0, atol=1e-6)


def test_exactly_window_length_is_one_call():
    calls = []

    def proc(seg):
        calls.append(len(seg))
        return seg

    x = _noise(200)
    _hann_overlap_add(x, window_n=200, process_chunk=proc)
    assert calls == [200]


def test_identity_process_reconstructs_interior():
    # out/weights cancels the window wherever coverage is non-zero, so an
    # identity process must reproduce the input exactly — except the first
    # and last sample, where the Hann endpoint weight is 0.
    x = _noise(1000)
    out = _hann_overlap_add(x, window_n=256, process_chunk=lambda s: s)
    assert out.shape == x.shape
    assert np.allclose(out[1:-1], x[1:-1], atol=1e-5)
    assert out[0] == pytest.approx(0.0, abs=1e-6)


def test_truncates_overlong_chunk_output():
    # process_chunk may return more samples than it was given (e.g. a padded
    # forward); the helper must truncate to the chunk's real length.
    x = _noise(100)
    out = _hann_overlap_add(
        x, window_n=200, process_chunk=lambda s: np.concatenate([s, s])
    )
    assert out.shape == x.shape
    assert np.allclose(out, x, atol=1e-6)


def test_output_is_float32():
    out = _hann_overlap_add(_noise(500), 200, lambda s: s)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Regression: byte-for-byte equivalence with the two original loops
# ---------------------------------------------------------------------------
# Verbatim copies of the pre-refactor implementations (only the model call is
# parameterised). If `_hann_overlap_add` ever drifts from these, the audio
# output of Stage 3a changes — these tests are the tripwire.


def _old_clearvoice_loop(x, window, cv_call):
    if len(x) <= window:
        return cv_call(x)[: len(x)]
    hop = window // 2
    win = np.hanning(window).astype(np.float32)
    out = np.zeros(len(x), dtype=np.float32)
    norm = np.zeros(len(x), dtype=np.float32)
    idx = 0
    while idx < len(x):
        seg = x[idx: idx + window]
        seg_out = cv_call(seg)[: len(seg)]
        w = win[: len(seg)]
        out[idx: idx + len(seg)] += seg_out * w
        norm[idx: idx + len(seg)] += w
        if idx + window >= len(x):
            break
        idx += hop
    norm[norm < 1e-8] = 1.0
    return out / norm


def _old_mpsenet_loop(audio_np, chunk_n, win_size, one_shot):
    hop_n = chunk_n // 2
    n = len(audio_np)
    if n <= chunk_n:
        return one_shot(audio_np)
    out = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    window = np.hanning(chunk_n).astype(np.float32)
    start = 0
    while start < n:
        end = min(start + chunk_n, n)
        seg = audio_np[start:end]
        if len(seg) < win_size:
            chunk_out = seg.astype(np.float32)
        else:
            seg_padded = (
                np.pad(seg, (0, chunk_n - len(seg))) if len(seg) < chunk_n else seg
            )
            chunk_out = one_shot(seg_padded)[: len(seg)]
        w = window[: len(seg)]
        out[start:end] += chunk_out * w
        weights[start:end] += w
        if end == n:
            break
        start += hop_n
    weights = np.maximum(weights, 1e-6)
    return (out / weights).astype(np.float32)


def _fake_model(s):
    """Deterministic non-trivial stand-in for an enhancement forward."""
    return (np.asarray(s, np.float32) * 0.7 + 0.05).astype(np.float32)


@pytest.mark.parametrize("n", [1, 500, 999, 1000, 1001, 1500, 2500, 3001])
def test_matches_original_clearvoice_loop(n):
    x = _noise(n)
    old = _old_clearvoice_loop(x, 1000, _fake_model)
    new = _hann_overlap_add(x, 1000, _fake_model)
    assert old.shape == new.shape
    assert np.allclose(old, new, atol=1e-6, rtol=0)


@pytest.mark.parametrize("n", [1250, 1500, 2500, 3001, 4096])
def test_matches_original_mpsenet_loop(n):
    chunk_n, win_size = 1000, 300
    x = _noise(n)
    old = _old_mpsenet_loop(x, chunk_n, win_size, _fake_model)

    # Mirror the refactored MPSENet enhance(): pad each chunk to chunk_n.
    def _process(seg):
        if len(seg) < win_size:
            return seg.astype(np.float32)
        sp = np.pad(seg, (0, chunk_n - len(seg))) if len(seg) < chunk_n else seg
        return _fake_model(sp)

    new = _hann_overlap_add(x, chunk_n, _process)
    assert old.shape == new.shape
    assert np.allclose(old, new, atol=1e-6, rtol=0)
