"""Unit tests for Stage 5 transcription (asr_pipeline/stages/transcription.py).

Covers the model-free surface — the silence/short-stream gate, the empty-result
contract, `_normalise_result`, backend dispatch + `load_signature`, the
`_ensure_ct2_model` resolution branches (incl. the subprocess-failure cleanup),
and `spill` — all on CPU with a stub backend. The `self._backend.transcribe(...)`
delegation into faster-whisper / wav2vec2 is a third-party boundary and is NOT
exercised here (CLAUDE.md: trust internal code, validate at boundaries).
"""

import json

import numpy as np
import pytest

from asr_pipeline.config import TranscriptionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.transcription import (
    _SILENCE_FLOOR,
    TranscriptionStage,
    _WhisperBackend,
    _WhisperXBackend,
    _empty_result,
    _ensure_ct2_model,
    _finite_or_zero,
    _normalise_result,
)


class _StubBackend:
    """Records every transcribe() call; returns a recognisable sentinel."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def transcribe(self, audio: np.ndarray) -> dict:
        self.calls.append(audio)
        return {"text": "STUB", "segments": [{"text": "STUB"}], "language": "pl"}


def _stage(backend: _StubBackend | None = None, **cfg_kwargs) -> TranscriptionStage:
    stage = TranscriptionStage(TranscriptionConfig(**cfg_kwargs))
    stage._backend = backend if backend is not None else _StubBackend()
    return stage


# ---------------------------------------------------------------------------
# _empty_result — the stage's output contract
# ---------------------------------------------------------------------------


def test_empty_result_shape():
    assert _empty_result("pl") == {"text": "", "segments": [], "language": "pl"}


def test_empty_result_is_fresh_each_call():
    a = _empty_result("pl")
    b = _empty_result("pl")
    # No aliased mutable segments list shared across speakers.
    assert a is not b
    assert a["segments"] is not b["segments"]
    a["segments"].append("x")
    assert b["segments"] == []


# ---------------------------------------------------------------------------
# _normalise_result — cross-backend top-level contract
# ---------------------------------------------------------------------------


def test_normalise_fills_text_from_segments():
    out = _normalise_result(
        {"segments": [{"text": " hello "}, {"text": "world"}]}, "pl"
    )
    assert out["text"] == "hello world"


def test_normalise_fills_language_when_absent():
    out = _normalise_result({"segments": []}, "pl")
    assert out["language"] == "pl"
    assert out["text"] == ""


def test_normalise_preserves_existing_text_and_language():
    out = _normalise_result(
        {"text": "kept", "language": "en", "segments": [{"text": "ignored"}]}, "pl"
    )
    assert out["text"] == "kept"
    assert out["language"] == "en"


def test_normalise_handles_missing_and_none_segments():
    assert _normalise_result({}, "pl")["text"] == ""
    assert _normalise_result({"segments": None}, "pl")["text"] == ""


def test_normalise_does_not_mutate_input_top_level():
    src = {"segments": [{"text": "a"}]}
    _normalise_result(src, "pl")
    assert "text" not in src        # shallow copy: caller's dict untouched
    assert "language" not in src


def test_normalise_sanitises_nonfinite_segment_timestamps():
    # WhisperX's interpolate_nans can ffill/bfill an unalignable segment to
    # all-NaN; the boundary must coerce None/NaN/inf to 0.0 before any writer.
    out = _normalise_result(
        {"segments": [
            {"start": None, "end": float("nan"), "text": "a"},
            {"start": float("inf"), "end": 2.0, "text": "b"},
        ]},
        "pl",
    )
    s0, s1 = out["segments"]
    assert s0["start"] == 0.0 and s0["end"] == 0.0
    assert s1["start"] == 0.0 and s1["end"] == 2.0


def test_normalise_does_not_mutate_input_segments():
    src = {"segments": [{"start": float("nan"), "end": 1.0, "text": "a"}]}
    _normalise_result(src, "pl")
    # Segment dicts are copied, not mutated in place.
    assert src["segments"][0]["start"] != src["segments"][0]["start"]   # still NaN


# ---------------------------------------------------------------------------
# _finite_or_zero — the time-sanitisation contract (single owner)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_finite_or_zero_coerces_nonfinite_to_zero(bad):
    assert _finite_or_zero(bad) == 0.0


def test_finite_or_zero_clamps_negative_to_zero():
    # WhisperX align() emits small negatives; written as `[ -0.30 → ...]` they
    # are silently dropped by the eval reader. Clamp here, the single owner.
    assert _finite_or_zero(-0.30) == 0.0
    assert _finite_or_zero(-1e-9) == 0.0


def test_finite_or_zero_passes_through_nonnegative():
    assert _finite_or_zero(0.0) == 0.0
    assert _finite_or_zero(2.5) == 2.5


def test_normalise_clamps_negative_segment_start():
    out = _normalise_result(
        {"segments": [{"start": -0.30, "end": 1.5, "text": "a"}]}, "pl"
    )
    assert out["segments"][0]["start"] == 0.0
    assert out["segments"][0]["end"] == 1.5


# ---------------------------------------------------------------------------
# The silence / short-stream gate in run()
# ---------------------------------------------------------------------------


def _run(stage: TranscriptionStage, assembled: dict, sample_rate: int = 16_000):
    ctx = PipelineContext(sample_rate=sample_rate)
    ctx.assembled = assembled
    stage.run(ctx)
    return ctx


def test_silent_stream_skips_backend():
    # The assembler's no-event sentinel: all-zeros, 1.0 s (> the 0.5 s floor).
    backend = _StubBackend()
    ctx = _run(_stage(backend), {"A": np.zeros(16_000, dtype=np.float32)})
    assert backend.calls == []                       # never reached Whisper
    assert ctx.transcripts["A"] == {"text": "", "segments": [], "language": "pl"}


def test_quiet_real_stream_reaches_backend():
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(16_000) * 1e-3).astype(np.float32)
    assert float(np.max(np.abs(quiet))) > _SILENCE_FLOOR    # genuinely audible
    backend = _StubBackend()
    ctx = _run(_stage(backend), {"A": quiet})
    assert len(backend.calls) == 1
    assert ctx.transcripts["A"]["text"] == "STUB"


def test_silence_floor_boundary_is_strict():
    n = 16_000
    just_below = np.full(n, _SILENCE_FLOOR * 0.5, dtype=np.float32)
    just_above = np.full(n, _SILENCE_FLOOR * 2.0, dtype=np.float32)
    backend = _StubBackend()
    ctx = _run(_stage(backend), {"below": just_below, "above": just_above})
    assert ctx.transcripts["below"]["text"] == ""        # skipped
    assert ctx.transcripts["above"]["text"] == "STUB"     # transcribed
    assert len(backend.calls) == 1


def test_gate_inspects_whole_array_not_just_head():
    # Silent head, loud tail — peak over the WHOLE array must clear the floor.
    audio = np.concatenate(
        [np.zeros(16_000, dtype=np.float32), np.full(200, 0.5, dtype=np.float32)]
    )
    backend = _StubBackend()
    ctx = _run(_stage(backend), {"A": audio})
    assert len(backend.calls) == 1
    assert ctx.transcripts["A"]["text"] == "STUB"


def test_short_stream_gate_tracks_sample_rate():
    # At SR=8000 the floor is 0.5 s = 4000 samples. A 3999-sample loud stream is
    # too short; a 4001-sample one is long enough. Proves the gate is SR-relative
    # (with the default 16 kHz, 4001 samples would be under the 8000 threshold).
    loud = lambda n: np.full(n, 0.5, dtype=np.float32)
    backend = _StubBackend()
    ctx = _run(_stage(backend), {"short": loud(3999), "ok": loud(4001)}, sample_rate=8000)
    assert ctx.transcripts["short"]["text"] == ""
    assert ctx.transcripts["ok"]["text"] == "STUB"


def test_empty_assembled_produces_no_transcripts():
    backend = _StubBackend()
    ctx = _run(_stage(backend), {})
    assert backend.calls == []
    assert ctx.transcripts == {}


# ---------------------------------------------------------------------------
# Mixture-baseline path
# ---------------------------------------------------------------------------


def test_mixture_transcribed_when_enabled():
    backend = _StubBackend()
    stage = _stage(backend, transcribe_mixture=True)
    ctx = PipelineContext(sample_rate=16_000)
    ctx.audio = np.full(16_000, 0.5, dtype=np.float32)
    stage.run(ctx)
    assert ctx.mixture_transcript["text"] == "STUB"


def test_silent_mixture_skipped():
    backend = _StubBackend()
    stage = _stage(backend, transcribe_mixture=True)
    ctx = PipelineContext(sample_rate=16_000)
    ctx.audio = np.zeros(16_000, dtype=np.float32)
    stage.run(ctx)
    assert ctx.mixture_transcript == {"text": "", "segments": [], "language": "pl"}


def test_mixture_not_touched_when_disabled():
    backend = _StubBackend()
    stage = _stage(backend, transcribe_mixture=False)
    ctx = PipelineContext(sample_rate=16_000)
    ctx.audio = np.full(16_000, 0.5, dtype=np.float32)
    stage.run(ctx)
    assert ctx.mixture_transcript is None


# ---------------------------------------------------------------------------
# Lifecycle: run-before-load, dispatch, load_signature
# ---------------------------------------------------------------------------


def test_run_before_load_raises():
    stage = TranscriptionStage(TranscriptionConfig())   # no _backend set
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(PipelineContext())


def test_load_dispatches_to_whisper_backend(monkeypatch):
    monkeypatch.setattr(_WhisperBackend, "load", lambda self, device: None)
    stage = TranscriptionStage(TranscriptionConfig(backend="whisper"))
    stage.load(torch_cpu())
    assert isinstance(stage._backend, _WhisperBackend)


def test_load_dispatches_to_whisperx_backend(monkeypatch):
    monkeypatch.setattr(_WhisperXBackend, "load", lambda self, device: None)
    stage = TranscriptionStage(TranscriptionConfig(backend="whisperx"))
    stage.load(torch_cpu())
    assert isinstance(stage._backend, _WhisperXBackend)


def test_load_unknown_backend_raises():
    stage = TranscriptionStage(TranscriptionConfig(backend="nonsense"))
    with pytest.raises(ValueError, match="Unknown transcription backend"):
        stage.load(torch_cpu())


def test_load_signature_whisper_excludes_align_model():
    stage = TranscriptionStage(TranscriptionConfig(backend="whisper", model_name="large-v2"))
    assert stage.load_signature() == ("whisper", "large-v2")


def test_load_signature_whisperx_includes_align_model():
    cfg = TranscriptionConfig(backend="whisperx", model_name="large-v2",
                              align_model_name="some/aligner")
    stage = TranscriptionStage(cfg)
    assert stage.load_signature() == ("whisperx", "large-v2", "some/aligner")


def torch_cpu():
    import torch
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# _ensure_ct2_model — id resolution and cold-path cleanup
# ---------------------------------------------------------------------------


def test_ensure_ct2_passes_through_openai_short_name():
    assert _ensure_ct2_model("large-v3") == "large-v3"


def test_ensure_ct2_passes_through_existing_local_path(tmp_path):
    # A path that contains "/" but exists on disk is returned unchanged.
    local = tmp_path / "my-ct2-model"
    local.mkdir()
    assert _ensure_ct2_model(str(local)) == str(local)


def test_ensure_ct2_returns_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr("asr_pipeline.stages.transcription._CT2_CACHE_ROOT", tmp_path)
    cache_dir = tmp_path / "org-model"
    cache_dir.mkdir()
    (cache_dir / "model.bin").write_bytes(b"x")     # warm cache marker
    assert _ensure_ct2_model("org/model") == str(cache_dir)


def test_ensure_ct2_subprocess_failure_cleans_cache(monkeypatch, tmp_path):
    """A failed converter must remove the half-built cache dir and raise — never
    leave a directory that the cache-hit probe would later trust."""
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr("asr_pipeline.stages.transcription._CT2_CACHE_ROOT", tmp_path)

    class _Loaded:
        def save_pretrained(self, path):
            pass

    # Patch from_pretrained on the real HF classes so the re-materialise step is
    # a no-op (no download), regardless of how the function resolves the names.
    for cls_name in ("WhisperForConditionalGeneration", "WhisperProcessor",
                     "WhisperTokenizerFast"):
        monkeypatch.setattr(getattr(transformers, cls_name), "from_pretrained",
                            staticmethod(lambda name: _Loaded()))

    class _Failed:
        returncode = 1
        stdout = "out"
        stderr = "boom"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Failed())

    cache_dir = tmp_path / "org-model"
    with pytest.raises(RuntimeError, match="ct2-transformers-converter failed"):
        _ensure_ct2_model("org/model")
    assert not cache_dir.exists()      # poisoned cache removed


# ---------------------------------------------------------------------------
# spill
# ---------------------------------------------------------------------------


def test_spill_writes_per_speaker_files_with_unicode(tmp_path):
    stage = _stage()
    ctx = PipelineContext()
    ctx.transcripts = {
        "spk0": {"text": "zażółć gęślą jaźń", "segments": [], "language": "pl"},
    }
    ctx.spk_to_label = {"spk0": "A"}
    stage.spill(ctx, tmp_path)

    txt = (tmp_path / "transcript_A.txt").read_text(encoding="utf-8")
    assert "zażółć gęślą jaźń" in txt          # diacritics preserved
    data = json.loads((tmp_path / "transcript_A.json").read_text(encoding="utf-8"))
    assert data["language"] == "pl"


def test_spill_uses_raw_speaker_key_when_no_label(tmp_path):
    stage = _stage()
    ctx = PipelineContext()
    ctx.transcripts = {"spk0": {"text": "x", "segments": [], "language": "pl"}}
    ctx.spk_to_label = {}       # no label mapping → fall back to raw key
    stage.spill(ctx, tmp_path)
    assert (tmp_path / "transcript_spk0.txt").exists()


def test_spill_noop_when_no_transcripts(tmp_path):
    stage = _stage()
    stage.spill(PipelineContext(), tmp_path)
    assert list(tmp_path.iterdir()) == []
