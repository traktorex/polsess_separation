"""Unit tests for Stage 3a enhancement.

Covers the shared Hann overlap-add helper, the backend dispatcher + its error
guards, and the ClearVoice resample/length contract — all on CPU, no model
weights.

`_hann_overlap_add` is the COLA reconstruction the ClearerVoice backend uses
for long audio. The regression test at the bottom keeps a verbatim copy of the
*original* ClearerVoice loop and asserts the helper reproduces it exactly — so
any future change to the COLA math that would alter the audio fails loudly here.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from asr_pipeline.config import EnhancementConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.enhancement import (
    _CLEARVOICE_BACKENDS,
    EnhancementStage,
    _ClearVoiceBackend,
    _hann_overlap_add,
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


def test_zero_coverage_sample_uses_floor_not_nan():
    # The Hann window is 0 at its endpoints, so the very first output sample
    # accumulates zero weight (only the first chunk touches it, with win[0]=0).
    # Without the `weights = np.maximum(weights, 1e-8)` floor that sample would
    # be 0/0 = NaN; the floor turns it into a finite 0.0. Assert no NaN/Inf
    # leaks into the enhanced audio anywhere.
    x = _noise(1000)
    out = _hann_overlap_add(x, window_n=256, process_chunk=lambda s: s)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-6)  # zero-coverage edge → floored, not NaN


# ---------------------------------------------------------------------------
# Stage load_signature + spill
# ---------------------------------------------------------------------------


def test_load_signature_is_backend_key():
    # The phase-major scheduler keys model (un)loading on this signature; the
    # ClearerVoice backend's whole identity is its backend name (self-download
    # by name), so the signature must be exactly that — change it and the
    # scheduler reloads/reuses the wrong model.
    stage = EnhancementStage(EnhancementConfig(backend="frcrn_se_16k"))
    assert stage.load_signature() == ("frcrn_se_16k",)
    other = EnhancementStage(EnhancementConfig(backend="mossformer_gan_se_16k"))
    assert other.load_signature() != stage.load_signature()


def test_spill_writes_enhanced_full(tmp_path):
    stage = EnhancementStage(EnhancementConfig(backend="frcrn_se_16k"))
    ctx = PipelineContext()
    ctx.sample_rate = 16_000
    ctx.enhanced_full = _noise(8000)
    stage.spill(ctx, tmp_path)
    out = tmp_path / "enhanced_full.wav"
    assert out.exists()
    audio, sr = sf.read(out)
    assert sr == 16_000
    assert len(audio) == 8000


def test_spill_noop_when_nothing_enhanced(tmp_path):
    stage = EnhancementStage(EnhancementConfig(backend="frcrn_se_16k"))
    ctx = PipelineContext()
    ctx.sample_rate = 16_000
    ctx.enhanced_full = None
    stage.spill(ctx, tmp_path)
    assert not (tmp_path / "enhanced_full.wav").exists()


# ---------------------------------------------------------------------------
# Backend dispatch + error guards
# ---------------------------------------------------------------------------


def test_load_dispatches_to_correct_backend(monkeypatch):
    # Lock the table that decides which model runs and at which sample rate —
    # a wrong SR pairing degrades output with no crash. Backend load() is
    # stubbed so no weights are read.
    monkeypatch.setattr(_ClearVoiceBackend, "load", lambda self, device: None)
    dev = torch.device("cpu")

    expected = {
        "frcrn_se_16k": ("FRCRN_SE_16K", 16_000),
        "mossformer_gan_se_16k": ("MossFormerGAN_SE_16K", 16_000),
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


def test_unknown_backend_raises():
    stage = EnhancementStage(EnhancementConfig(backend="does_not_exist"))
    with pytest.raises(ValueError, match="Unknown enhancement backend"):
        stage.load(torch.device("cpu"))


def test_run_before_load_raises():
    stage = EnhancementStage(EnhancementConfig(backend="frcrn_se_16k"))
    ctx = PipelineContext()
    ctx.audio = _noise(1000)
    with pytest.raises(RuntimeError, match="called before load"):
        stage.run(ctx)


def test_run_audio_none_raises():
    stage = EnhancementStage(EnhancementConfig(backend="frcrn_se_16k"))
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
    # A native-48 kHz backend, so a 16 kHz input exercises the generic up/down
    # resample round-trip of _ClearVoiceBackend (independent of any registered
    # backend — the registry has no 48 kHz entry, but the resample capability is
    # generic and must stay tested). _decode_window_s is normally set in load().
    backend = _ClearVoiceBackend("_stub_48k", 48_000)
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
# Regression: byte-for-byte equivalence with the original ClearerVoice loop
# ---------------------------------------------------------------------------
# Verbatim copy of the pre-refactor implementation (only the model call is
# parameterised). If `_hann_overlap_add` ever drifts from this, the audio
# output of Stage 3a changes — this test is the tripwire.


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


# ---------------------------------------------------------------------------
# Observation Adding (OA) / dry-wet mix in EnhancementStage.run
# ---------------------------------------------------------------------------
# A stub backend returns a known constant array regardless of input, so the
# blend `out = (1-r)*enhanced + r*observed` is checkable in closed form.


class _ConstBackend:
    """Stub enhancement backend that ignores its input and returns a fixed
    constant array of the input length — so OA's convex blend with the observed
    signal is exactly computable."""

    def __init__(self, value: float) -> None:
        self.value = value

    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.full(len(audio_np), self.value, dtype=np.float32)


def _stage_with_const_backend(ratio: float, value: float = 2.0) -> EnhancementStage:
    stage = EnhancementStage(
        EnhancementConfig(backend="frcrn_se_16k", observation_mix_ratio=ratio)
    )
    stage._backend = _ConstBackend(value)
    return stage


def test_oa_blend_is_convex_mix_of_enhanced_and_observed():
    stage = _stage_with_const_backend(ratio=0.5, value=2.0)
    ctx = PipelineContext()
    ctx.sample_rate = 16_000
    ctx.audio = _noise(4000)  # observed (dry)
    stage.run(ctx)
    enhanced = np.full(4000, 2.0, dtype=np.float32)
    expected = 0.5 * enhanced + 0.5 * ctx.audio.astype(np.float32)
    assert np.allclose(ctx.enhanced_full, expected, atol=1e-6)
    assert ctx.enhanced_full.dtype == np.float32


def test_oa_ratio_zero_is_byte_identical_to_no_oa():
    # r=0 must be a strict no-op: the stored array equals the backend output
    # byte-for-byte (the observed signal never enters the arithmetic).
    observed = _noise(4000)
    stage_oa0 = _stage_with_const_backend(ratio=0.0, value=2.0)
    ctx0 = PipelineContext()
    ctx0.sample_rate = 16_000
    ctx0.audio = observed.copy()
    stage_oa0.run(ctx0)

    raw_enhanced = _ConstBackend(2.0).enhance(observed, 16_000)
    assert np.array_equal(ctx0.enhanced_full, raw_enhanced)


def test_oa_ratio_one_returns_observed():
    stage = _stage_with_const_backend(ratio=1.0, value=2.0)
    ctx = PipelineContext()
    ctx.sample_rate = 16_000
    ctx.audio = _noise(4000)
    stage.run(ctx)
    assert np.allclose(ctx.enhanced_full, ctx.audio.astype(np.float32), atol=1e-6)


class _ShorterBackend:
    """Stub backend that returns FEWER samples than its input — exercises the
    OA blend's `n = min(len(enhanced), len(observed))` length guard, which the
    const/identity backends (equal length) never reach."""

    def __init__(self, out_len: int, value: float) -> None:
        self.out_len = out_len
        self.value = value

    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.full(self.out_len, self.value, dtype=np.float32)


def test_oa_blend_length_guard_truncates_to_shorter_stream():
    # Backend returns 3000 samples for a 4000-sample input; the blend must clip
    # both sides to n = min(3000, 4000) = 3000 and stay finite (a mismatched
    # backend would otherwise broadcast-crash).
    stage = EnhancementStage(
        EnhancementConfig(backend="frcrn_se_16k", observation_mix_ratio=0.5)
    )
    stage._backend = _ShorterBackend(out_len=3000, value=2.0)
    ctx = PipelineContext()
    ctx.sample_rate = 16_000
    ctx.audio = _noise(4000)
    stage.run(ctx)
    assert len(ctx.enhanced_full) == 3000
    assert np.all(np.isfinite(ctx.enhanced_full))
    expected = 0.5 * 2.0 + 0.5 * ctx.audio[:3000].astype(np.float32)
    assert np.allclose(ctx.enhanced_full, expected, atol=1e-6)
