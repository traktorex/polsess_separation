"""Tests for Stage 3c (PostSeparationProcessingStage).

The naive backend is identity, so the stage's own logic — VAD-mask
application, mask/output length alignment — is observable without a model.
But identity hides the stage's *reason to exist*: that BWE runs on the
UNMASKED raw and the mask is applied AFTER, and that fully-masked streams
skip the backend entirely. Those are pinned with a non-identity spy backend.
The neural backends' load/dispatch/guard/level-restore seams are covered
with stub models (no weights); the neural forwards themselves are a
third-party boundary and are not exercised here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from asr_pipeline.config import PostSeparationProcessingConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.post_separation_processing import (
    PostSeparationProcessingStage,
    _APBWEBackend,
    _FlowHighBackend,
    _NaiveBackend,
    _match_length,
)

SR = 16_000


def _stage() -> PostSeparationProcessingStage:
    stage = PostSeparationProcessingStage(
        PostSeparationProcessingConfig(backend="naive")
    )
    stage.load(torch.device("cpu"))
    return stage


def _entry(s1_raw, mask1, s2_raw, mask2, idx=0) -> dict:
    return {
        "idx": idx,
        "s1_raw": np.asarray(s1_raw, dtype=np.float32),
        "s2_raw": np.asarray(s2_raw, dtype=np.float32),
        "mask1": np.asarray(mask1, dtype=np.float32),
        "mask2": np.asarray(mask2, dtype=np.float32),
    }


def _ctx_with(entries) -> PipelineContext:
    ctx = PipelineContext(sample_rate=SR)
    ctx.overlap_separated = entries
    return ctx


def test_run_before_load_raises():
    stage = PostSeparationProcessingStage(
        PostSeparationProcessingConfig(backend="naive")
    )
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx_with([_entry([1.0], [1.0], [1.0], [1.0])]))


def test_no_overlaps_is_noop():
    ctx = _ctx_with([])
    _stage().run(ctx)
    assert ctx.overlap_separated == []


def test_naive_gated_is_raw_times_mask():
    raw1 = np.linspace(-1.0, 1.0, 1024)
    mask1 = np.concatenate([np.ones(512), np.zeros(512)])
    raw2 = np.full(1024, 0.5)
    mask2 = np.ones(1024)
    entry = _entry(raw1, mask1, raw2, mask2)
    _stage().run(_ctx_with([entry]))
    np.testing.assert_allclose(
        entry["s1_gated"], raw1.astype(np.float32) * mask1.astype(np.float32)
    )
    np.testing.assert_allclose(entry["s2_gated"], raw2.astype(np.float32))
    assert entry["s1_gated"].dtype == np.float32


class _SpyBackend:
    """Non-identity backend that records what it was asked to extend.

    Lets the tests observe the stage's two load-bearing behaviours that an
    identity backend cannot show: extend() runs on the UNMASKED raw, and a
    fully-masked stream never reaches extend() at all.
    """

    def __init__(self, gain: float = 2.0) -> None:
        self.calls: list[np.ndarray] = []
        self.gain = gain

    def extend(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls.append(np.asarray(audio_np).copy())
        return (audio_np * self.gain).astype(np.float32)


def _spy_stage(spy: _SpyBackend) -> PostSeparationProcessingStage:
    stage = PostSeparationProcessingStage(
        PostSeparationProcessingConfig(backend="naive")
    )
    stage._backend = spy   # bypass load(); inject the spy directly
    return stage


def test_extend_runs_on_unmasked_raw_then_mask_applied_after():
    # raw is non-zero everywhere; mask zeros the second half. If extend ran on
    # the unmasked raw, the spy sees 0.3 in the masked-out region; the gate is
    # applied AFTER, so the output's second half is zero.
    spy = _SpyBackend(gain=2.0)
    raw = np.full(1024, 0.3, dtype=np.float32)
    mask = np.concatenate([np.ones(512), np.zeros(512)]).astype(np.float32)
    entry = _entry(raw, mask, raw, np.ones(1024))
    _spy_stage(spy).run(_ctx_with([entry]))

    assert len(spy.calls) == 2                          # s1 and s2 both run
    np.testing.assert_array_equal(spy.calls[0], raw)    # saw the UNMASKED raw
    np.testing.assert_allclose(entry["s1_gated"][:512], 0.6)   # (raw*2)*1
    np.testing.assert_array_equal(entry["s1_gated"][512:], 0.0)  # masked after


def test_silent_stream_does_not_invoke_backend():
    # Distinguishes "skipped extend" from "ran extend then masked to zero":
    # under the skip, the spy is called only for the non-silent s2.
    spy = _SpyBackend()
    raw = np.ones(1024, dtype=np.float32)
    entry = _entry(raw, np.zeros(1024), raw, np.ones(1024))   # s1 fully masked
    _spy_stage(spy).run(_ctx_with([entry]))

    assert len(spy.calls) == 1                          # s1 never reached extend
    np.testing.assert_array_equal(spy.calls[0], raw)    # the call was s2's raw
    np.testing.assert_array_equal(entry["s1_gated"], np.zeros(1024, np.float32))
    assert entry["s1_gated"].dtype == np.float32


def test_mask_shorter_than_output_pads_with_silence():
    # Backend output longer than the mask: the un-covered tail must be
    # gated out (padded mask = 0), not passed through.
    raw = np.ones(1000)
    mask = np.ones(900)
    entry = _entry(raw, mask, raw, np.ones(1000))
    _stage().run(_ctx_with([entry]))
    np.testing.assert_allclose(entry["s1_gated"][:900], 1.0)
    np.testing.assert_array_equal(entry["s1_gated"][900:], 0.0)


def test_mask_longer_than_output_is_truncated():
    raw = np.ones(900)
    mask = np.ones(1000)
    entry = _entry(raw, mask, raw, np.ones(900))
    _stage().run(_ctx_with([entry]))
    assert len(entry["s1_gated"]) == 900
    np.testing.assert_allclose(entry["s1_gated"], 1.0)


def test_raw_streams_left_untouched():
    raw1 = np.linspace(-1.0, 1.0, 1024).astype(np.float32)
    entry = _entry(raw1.copy(), np.zeros(1024), raw1.copy(), np.ones(1024))
    _stage().run(_ctx_with([entry]))
    np.testing.assert_array_equal(entry["s1_raw"], raw1)
    np.testing.assert_array_equal(entry["s2_raw"], raw1)


# ---------------------------------------------------------------------------
# _match_length helper
# ---------------------------------------------------------------------------


def test_match_length_trims_and_pads():
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(_match_length(x, 5), x[:5])         # trim
    np.testing.assert_array_equal(_match_length(x, 10), x)            # exact
    padded = _match_length(x, 13)                                    # pad
    assert len(padded) == 13
    np.testing.assert_array_equal(padded[:10], x)
    np.testing.assert_array_equal(padded[10:], 0.0)
    assert _match_length(x, 5).dtype == np.float32


# ---------------------------------------------------------------------------
# Backend dispatch + load_signature
# ---------------------------------------------------------------------------


def _cfg(backend, **kw):
    return PostSeparationProcessingConfig(backend=backend, **kw)


def test_load_dispatches_to_correct_backend(monkeypatch):
    monkeypatch.setattr(_APBWEBackend, "load", lambda self, device: None)
    monkeypatch.setattr(_FlowHighBackend, "load", lambda self, device: None)
    dev = torch.device("cpu")

    s = PostSeparationProcessingStage(_cfg("naive"))
    s.load(dev)
    assert isinstance(s._backend, _NaiveBackend)

    s = PostSeparationProcessingStage(_cfg("ap_bwe", checkpoint_path="/x/y.zip"))
    s.load(dev)
    assert isinstance(s._backend, _APBWEBackend)
    assert s._backend.checkpoint_path == "/x/y.zip"     # threaded through

    s = PostSeparationProcessingStage(_cfg("flowhigh", flowhigh_input_sr=8000))
    s.load(dev)
    assert isinstance(s._backend, _FlowHighBackend)
    assert s._backend.input_sr == 8000


def test_unknown_backend_raises():
    # PostSeparationProcessingConfig has no __post_init__, so load()'s guard is
    # the only thing rejecting a bad backend name when built standalone.
    s = PostSeparationProcessingStage(_cfg("nonsense"))
    with pytest.raises(ValueError, match="Unknown post_separation_processing backend"):
        s.load(torch.device("cpu"))


def test_load_signature_shapes():
    sig = lambda **kw: PostSeparationProcessingStage(_cfg(**kw)).load_signature()
    assert sig(backend="naive") == ("naive",)
    assert sig(backend="ap_bwe", checkpoint_path="/x/y") == ("ap_bwe", "/x/y")
    assert sig(backend="flowhigh", flowhigh_input_sr=8000) == ("flowhigh", 8000)


# ---------------------------------------------------------------------------
# AP-BWE load error paths + short-input guard (the latent crash regression)
# ---------------------------------------------------------------------------


def test_apbwe_load_missing_checkpoint_raises(tmp_path):
    backend = _APBWEBackend(str(tmp_path / "does_not_exist.zip"))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        backend.load(torch.device("cpu"))


def test_apbwe_load_dir_checkpoint_raises(tmp_path):
    ckpt_dir = tmp_path / "extracted_ckpt"
    ckpt_dir.mkdir()
    backend = _APBWEBackend(str(ckpt_dir))
    with pytest.raises(IsADirectoryError, match="directory"):
        backend.load(torch.device("cpu"))


def test_apbwe_extend_before_load_raises():
    with pytest.raises(RuntimeError, match="before load"):
        _APBWEBackend("/x").extend(np.ones(2048, np.float32), SR)


def test_apbwe_short_input_passes_through_without_crashing():
    # Regression: pre-fix the guard used win_size (320), but the STFT
    # reflect-pad needs len > n_fft//2 = 512, so a 400-sample non-silent input
    # crashed inside torch.stft. The guard now gates at n_fft, so it passes
    # through cleanly (and as float32) instead of reaching the STFT.
    backend = _APBWEBackend("/x")
    backend._model = object()            # never reached — guard short-circuits
    backend._h = {"n_fft": 1024, "win_size": 320, "hop_size": 80}
    backend._device = torch.device("cpu")
    out = backend.extend(np.full(400, 0.3, dtype=np.float64), SR)
    assert out.dtype == np.float32
    assert len(out) == 400


# ---------------------------------------------------------------------------
# FlowHigh extend: guard, level restoration, near-silent
# ---------------------------------------------------------------------------


def _flowhigh_with_const_model(const: float) -> _FlowHighBackend:
    """A FlowHigh backend whose generate() returns a constant 48 kHz tensor."""
    backend = _FlowHighBackend(SR)
    backend._device = torch.device("cpu")

    def fake_generate(audio, sr_in, sr_out):
        n48 = int(len(audio) * sr_out / sr_in)
        return torch.full((1, n48), const)

    backend._model = SimpleNamespace(generate=fake_generate)
    return backend


def test_flowhigh_extend_before_load_raises():
    with pytest.raises(RuntimeError, match="before load"):
        _FlowHighBackend(SR).extend(np.ones(1024, np.float32), SR)


def test_flowhigh_short_input_passes_through_as_float32():
    backend = _flowhigh_with_const_model(0.9)
    out = backend.extend(np.full(256, 0.3, dtype=np.float64), SR)   # < 512
    assert out.dtype == np.float32
    assert len(out) == 256


def test_flowhigh_restores_input_rms():
    backend = _flowhigh_with_const_model(0.9)
    x = np.full(1024, 0.2, dtype=np.float32)            # in-RMS = 0.2
    out = backend.extend(x, SR)
    assert len(out) == 1024
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    assert abs(out_rms - 0.2) < 1e-3                    # level preserved


def test_flowhigh_near_silent_input_not_amplified():
    # The fix: a near-silent input must map to a near-silent output, not be
    # left at FlowHigh's ~0.9 internal peak (pre-fix the in_rms>1e-8 guard
    # skipped the rescale, leaking full-scale content).
    backend = _flowhigh_with_const_model(0.9)
    x = np.full(1024, 1e-9, dtype=np.float32)
    out = backend.extend(x, SR)
    assert float(np.max(np.abs(out))) < 1e-3


# ---------------------------------------------------------------------------
# spill
# ---------------------------------------------------------------------------


def test_spill_naive_has_no_backend_suffix(tmp_path):
    stage = _stage()
    entry = _entry(np.ones(8), np.ones(8), np.ones(8), np.ones(8))
    ctx = _ctx_with([entry])
    stage.run(ctx)
    stage.spill(ctx, tmp_path)
    assert (tmp_path / "overlap_0_s1_gated.wav").exists()
    assert (tmp_path / "overlap_0_s2_gated.wav").exists()


def test_spill_bwe_backend_tags_filename(tmp_path):
    stage = PostSeparationProcessingStage(_cfg("ap_bwe"))
    entry = _entry(np.ones(8), np.ones(8), np.ones(8), np.ones(8))
    entry["s1_gated"] = np.ones(8, np.float32)
    entry["s2_gated"] = np.ones(8, np.float32)
    stage.spill(_ctx_with([entry]), tmp_path)
    assert (tmp_path / "overlap_0_s1_gated_bwe_ap_bwe.wav").exists()
    assert (tmp_path / "overlap_0_s2_gated_bwe_ap_bwe.wav").exists()


def test_spill_no_overlaps_is_noop(tmp_path):
    _stage().spill(_ctx_with([]), tmp_path)
    assert list(tmp_path.iterdir()) == []
