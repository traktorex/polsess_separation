"""Unit tests for Stage 3a enhancement.

Covers the shared Hann overlap-add helper, the MP-SENet STFT/ISTFT seam, the
backend dispatcher + its error guards, and the ClearVoice resample/length
contract — all on CPU, no model weights.

`_hann_overlap_add` replaced two near-identical chunking loops (MP-SENet's
`_enhance_chunked` and ClearerVoice's `_enhance_native`). The regression
tests at the bottom keep verbatim copies of both *original* loops and assert
the helper reproduces them exactly — so any future change to the COLA math
that would alter the audio fails loudly here.
"""

import numpy as np
import pytest
import torch

from asr_pipeline.config import EnhancementConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.enhancement import (
    _CLEARVOICE_BACKENDS,
    EnhancementStage,
    _ClearVoiceBackend,
    _MPSENetBackend,
    _hann_overlap_add,
    _mpsenet_istft,
    _mpsenet_stft,
)

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
# MP-SENet STFT/ISTFT seam (default backend hot path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cf", [0.3, 0.5, 1.0])
def test_mpsenet_stft_istft_roundtrip(cf):
    # Default backend hot path: the STFT compresses magnitude (^cf) and the
    # ISTFT inverts it (^1/cf). The interior (away from the reflect-pad edges)
    # must reconstruct to within float tolerance (measured clean err ~7e-7 at
    # cf=0.3). Looping cf pins the compress/decompress *pairing*, not the
    # literal 0.3: a swapped pow() would silently corrupt every MP-SENet output
    # and fail here for cf != 1.0 (cf == 1.0 is the degenerate no-op case).
    h = {"n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": cf}
    torch.manual_seed(0)
    x = torch.randn(1, 4000)
    mag, pha = _mpsenet_stft(x, h)
    rec = _mpsenet_istft(mag, pha, h)
    assert torch.allclose(rec[..., 200:-200], x[..., 200:-200], atol=1e-5)


# ---------------------------------------------------------------------------
# Backend dispatch + error guards
# ---------------------------------------------------------------------------


def test_load_dispatches_to_correct_backend(monkeypatch):
    # Lock the table that decides which model runs and at which sample rate —
    # a wrong SR pairing degrades output with no crash. Backend load() is
    # stubbed so no weights are read.
    monkeypatch.setattr(_MPSENetBackend, "load", lambda self, device: None)
    monkeypatch.setattr(_ClearVoiceBackend, "load", lambda self, device: None)
    dev = torch.device("cpu")

    expected = {
        "frcrn_se_16k": ("FRCRN_SE_16K", 16_000),
        "mossformer_gan_se_16k": ("MossFormerGAN_SE_16K", 16_000),
        "mossformer2_se_48k": ("MossFormer2_SE_48K", 48_000),
    }
    # Literal pin above is the SR-pairing tripwire; this guard makes a newly
    # added backend row fail here until it is pinned too.
    assert set(expected) == set(_CLEARVOICE_BACKENDS)
    for backend, (model_name, sr) in expected.items():
        stage = EnhancementStage(EnhancementConfig(backend=backend))
        stage.load(dev)
        assert isinstance(stage._backend, _ClearVoiceBackend)
        assert stage._backend.model_name == model_name
        assert stage._backend.native_sample_rate == sr

    stage = EnhancementStage(EnhancementConfig(backend="mpsenet"))
    stage.load(dev)
    assert isinstance(stage._backend, _MPSENetBackend)


def test_unknown_backend_raises():
    stage = EnhancementStage(EnhancementConfig(backend="does_not_exist"))
    with pytest.raises(ValueError, match="Unknown enhancement backend"):
        stage.load(torch.device("cpu"))


def test_run_before_load_raises():
    stage = EnhancementStage(EnhancementConfig(backend="mpsenet"))
    ctx = PipelineContext()
    ctx.audio = _noise(1000)
    with pytest.raises(RuntimeError, match="called before load"):
        stage.run(ctx)


def test_run_audio_none_raises():
    stage = EnhancementStage(EnhancementConfig(backend="mpsenet"))
    stage._backend = object()  # bypass load(); exercise the audio guard
    ctx = PipelineContext()
    ctx.audio = None
    with pytest.raises(RuntimeError, match="audio is None"):
        stage.run(ctx)


def test_clearvoice_enhance_before_load_raises():
    backend = _ClearVoiceBackend("FRCRN_SE_16K", 16_000)
    with pytest.raises(RuntimeError, match="called before load"):
        backend.enhance(_noise(1000), 16_000)


# ---------------------------------------------------------------------------
# ClearVoice resample / length contract (stubbed forward — no weights)
# ---------------------------------------------------------------------------


def _stub_clearvoice(cv=None):
    # MossFormer2_SE_48K: native 48 kHz, so a 16 kHz input exercises the
    # up/down resample round-trip. _decode_window_s is normally set in load().
    backend = _ClearVoiceBackend("MossFormer2_SE_48K", 48_000)
    backend._device = torch.device("cpu")
    backend._decode_window_s = 20.0
    backend._cv = cv if cv is not None else (lambda arr: arr)  # arr is (1, T)
    return backend


def test_clearvoice_enhance_preserves_length_identity():
    # The assembler slices per-speaker streams on this length, so enhance()
    # must return exactly the input length regardless of resample / chunking.
    out = _stub_clearvoice().enhance(_noise(32_000), 16_000)
    assert len(out) == 32_000
    assert out.dtype == np.float32


def test_clearvoice_enhance_truncates_long_forward():
    backend = _stub_clearvoice(cv=lambda arr: np.concatenate([arr, arr], axis=1))
    assert len(backend.enhance(_noise(32_000), 16_000)) == 32_000


def test_clearvoice_enhance_pads_short_forward():
    backend = _stub_clearvoice(cv=lambda arr: arr[:, : arr.shape[1] // 2])
    assert len(backend.enhance(_noise(32_000), 16_000)) == 32_000


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
