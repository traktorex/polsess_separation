"""Tests for the custom speaker-embedding wrappers
(asr_pipeline/stages/custom_embeddings.py) and their wiring into the
diarization stage.

Two tiers:
  - Interface tests on a fake subclass (no model download): exercise the
    BaseCustomSpeakerEmbedding mask handling, NaN-on-short-signal bookkeeping,
    shape contract, and the diarization-stage injection path with the real
    model build stubbed out.
  - Real-model tests (ecapa2 / eres2netv2): SKIPPED when the model can't be
    loaded (offline / not downloaded). They assert the verified interface
    constants and a basic discriminativeness sanity check.

All CPU-only.
"""

import numpy as np
import pytest
import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.stages.custom_embeddings import (
    CUSTOM_EMBEDDING_NAMES,
    BaseCustomSpeakerEmbedding,
    build_custom_embedding,
)
from asr_pipeline.stages.diarization import DiarizationStage

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tier 1 — interface tests on a fake embedder (no downloads)
# ---------------------------------------------------------------------------


class _FakeEmbedding(BaseCustomSpeakerEmbedding):
    """Deterministic stand-in: embeds a 1-D signal as [mean, std, len-proxy].

    Discriminative enough for the interface tests and needs no model. Min
    usable length is 4 samples (below it the std term is degenerate), letting us
    exercise the short-signal NaN path with tiny tensors.
    """

    _DIM = 3

    def __init__(self, device=None):
        super().__init__(device=device)
        self.moved_to = None

    def _move_to(self, device):
        self.moved_to = device

    @property
    def dimension(self):
        return self._DIM

    @property
    def min_num_samples(self):
        return 4

    def _embed_one(self, signal):
        s = signal.detach().cpu().numpy().astype(np.float64)
        return np.array([s.mean(), s.std(), float(len(s)) / 16000.0])


def test_call_no_mask_shape_and_values():
    emb = _FakeEmbedding(CPU)
    # two distinct constant signals
    a = torch.full((1, 1, 100), 0.5)
    b = torch.full((1, 1, 100), -0.5)
    batch = torch.cat([a, b], dim=0)        # (2, 1, 100)
    out = emb(batch)
    assert out.shape == (2, emb.dimension)
    assert np.isfinite(out).all()
    # means recovered
    assert out[0, 0] == pytest.approx(0.5)
    assert out[1, 0] == pytest.approx(-0.5)


def test_call_too_short_row_is_nan():
    emb = _FakeEmbedding(CPU)
    short = torch.zeros(1, 1, emb.min_num_samples - 1)
    out = emb(short)
    assert out.shape == (1, emb.dimension)
    assert np.isnan(out).all()


def test_call_mask_selects_subset():
    # 8 frames, keep only the first half -> only the first half of samples feed
    # the embedder. With a step signal (first half = 1.0, second half = 0.0) the
    # masked mean must be ~1.0, the unmasked mean ~0.5.
    emb = _FakeEmbedding(CPU)
    n = 800
    sig = torch.zeros(1, 1, n)
    sig[..., : n // 2] = 1.0
    nframes = 8
    mask = torch.zeros(1, nframes)
    mask[:, : nframes // 2] = 1.0            # keep first half of frames
    masked = emb(sig, masks=mask)
    unmasked = emb(sig)
    assert masked[0, 0] == pytest.approx(1.0)
    assert unmasked[0, 0] == pytest.approx(0.5)


def test_call_mask_all_zero_row_is_nan():
    # A fully-masked-out item has zero usable samples (< min_num_samples) -> NaN.
    emb = _FakeEmbedding(CPU)
    sig = torch.ones(1, 1, 800)
    mask = torch.zeros(1, 8)
    out = emb(sig, masks=mask)
    assert np.isnan(out).all()


def test_call_rejects_multichannel():
    emb = _FakeEmbedding(CPU)
    with pytest.raises(AssertionError):
        emb(torch.zeros(1, 2, 100))          # num_channels == 2


def test_to_returns_self_and_moves():
    emb = _FakeEmbedding(CPU)
    ret = emb.to(CPU)
    assert ret is emb
    assert emb.moved_to == CPU
    with pytest.raises(TypeError):
        emb.to("cpu")                        # must be a torch.device


def test_metric_and_sample_rate_constants():
    emb = _FakeEmbedding(CPU)
    assert emb.metric == "cosine"
    assert emb.sample_rate == 16000


def test_build_custom_embedding_unknown_returns_none():
    # A pyannote-format id (or any non-custom string) is NOT a custom name.
    assert build_custom_embedding("eek/wespeaker-voxceleb-resnet293-LM", CPU) is None
    assert build_custom_embedding("pyannote/embedding", CPU) is None


def test_custom_names_inventory():
    assert set(CUSTOM_EMBEDDING_NAMES) == {"ecapa2", "eres2netv2"}


# ---------------------------------------------------------------------------
# Tier 1 — diarization-stage injection wiring (model build stubbed)
# ---------------------------------------------------------------------------


class _FakeSpeakerDiarization:
    """Minimal SpeakerDiarization stand-in to exercise the injection path."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self._embedding = object()          # the placeholder pyannote embedder
        self.instantiated = None
        self.device = None

    def instantiate(self, params):
        self.instantiated = params

    def to(self, device):
        self.device = device
        return self


def test_load_injects_custom_embedder(monkeypatch):
    """A custom name builds with a pyannote placeholder, then `_embedding` is
    replaced by the wrapper from build_custom_embedding."""
    import asr_pipeline.stages.diarization as diar_mod

    # Stub the pyannote SpeakerDiarization import (it lives behind a local
    # `from pyannote.audio.pipelines import SpeakerDiarization`).
    import pyannote.audio.pipelines as pa_pipelines
    monkeypatch.setattr(
        pa_pipelines, "SpeakerDiarization", _FakeSpeakerDiarization, raising=True
    )

    sentinel = _FakeEmbedding(CPU)
    captured = {}

    def fake_build(name, device):
        captured["name"] = name
        captured["device"] = device
        return sentinel

    monkeypatch.setattr(diar_mod, "build_custom_embedding", fake_build)

    stage = DiarizationStage(
        DiarizationConfig(embedding="ecapa2", hf_token="dummy")
    )
    stage.load(CPU)

    # Built with a placeholder, NOT the custom name (pyannote can't dispatch it).
    # The placeholder is resnet34-LM (the stock 3.1 embedder) rather than the
    # gated "pyannote/embedding" — see DiarizationStage.load's comment.
    assert (
        _FakeSpeakerDiarization.last_kwargs["embedding"]
        == "pyannote/wespeaker-voxceleb-resnet34-LM"
    )
    # The custom wrapper was injected.
    assert stage._pipeline._embedding is sentinel
    assert captured["name"] == "ecapa2"
    assert captured["device"] == CPU


def test_load_pyannote_id_not_treated_as_custom(monkeypatch):
    """A pyannote-format model id keeps the existing path: passed straight to
    SpeakerDiarization, no custom injection."""
    import asr_pipeline.stages.diarization as diar_mod
    import pyannote.audio.pipelines as pa_pipelines

    monkeypatch.setattr(
        pa_pipelines, "SpeakerDiarization", _FakeSpeakerDiarization, raising=True
    )

    def fail_build(name, device):  # must never be called for a pyannote id
        raise AssertionError("build_custom_embedding called for a pyannote id")

    monkeypatch.setattr(diar_mod, "build_custom_embedding", fail_build)

    model_id = "eek/wespeaker-voxceleb-resnet293-LM"
    stage = DiarizationStage(DiarizationConfig(embedding=model_id, hf_token="dummy"))
    stage.load(CPU)

    assert _FakeSpeakerDiarization.last_kwargs["embedding"] == model_id


# ---------------------------------------------------------------------------
# Tier 2 — real-model tests (skipped if the model can't load)
# ---------------------------------------------------------------------------


def _try_build(name):
    try:
        return build_custom_embedding(name, CPU)
    except Exception as exc:  # offline / not downloaded / dep missing
        pytest.skip(f"{name} unavailable on this machine: {type(exc).__name__}: {exc}")


def _two_speaker_signals(n=16000, sr=16000):
    """Two synthetic harmonic 'voices' + a duplicate of the first."""
    t = np.arange(n) / sr

    def voice(f0, seed):
        rng = np.random.default_rng(seed)
        sig = np.zeros(n, dtype=np.float32)
        for k in range(1, 6):
            sig += (0.3 / k) * np.sin(2 * np.pi * f0 * k * t)
        sig += 0.01 * rng.standard_normal(n).astype(np.float32)
        return sig

    a = voice(110.0, 1)
    b = voice(220.0, 2)
    return a, b


@pytest.mark.parametrize("name", ["ecapa2", "eres2netv2"])
def test_real_model_interface_and_discriminativeness(name):
    emb = _try_build(name)

    # Verified interface constants.
    assert emb.dimension == 192
    assert emb.sample_rate == 16000
    assert emb.metric == "cosine"
    assert 0 < emb.min_num_samples < 16000

    # dummy (1,1,16000) -> (1, dimension), finite.
    out = emb(torch.zeros(1, 1, 16000))
    assert out.shape == (1, emb.dimension)
    assert np.isfinite(out).all()

    # Discriminativeness: a duplicate of speaker A is more cosine-similar to A
    # than a different speaker B is.
    a, b = _two_speaker_signals()
    batch = torch.from_numpy(np.stack([a, b, a.copy()])).unsqueeze(1)  # (3,1,N)
    embs = emb(batch)
    assert embs.shape == (3, emb.dimension)
    assert np.isfinite(embs).all()

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    sim_same = cos(embs[0], embs[2])
    sim_diff = cos(embs[0], embs[1])
    assert sim_same > sim_diff
