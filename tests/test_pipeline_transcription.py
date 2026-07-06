"""Unit tests for Stage 5 transcription (asr_pipeline/stages/transcription.py).

Covers the model-free surface — the silence/short-stream gate, the empty-result
contract, `_normalise_result`, backend dispatch + `load_signature`, the
`_ensure_ct2_model` resolution branches (incl. the subprocess-failure cleanup),
and `spill` — all on CPU with a stub backend. The `self._backend.transcribe(...)`
delegation into faster-whisper / wav2vec2 is a third-party boundary and is NOT
exercised here (CLAUDE.md: trust internal code, validate at boundaries).
"""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from asr_pipeline.config import TranscriptionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.transcription import (
    TranscriptionStage,
    _WhisperBackend,
    _WhisperXBackend,
    _CohereXBackend,
    _empty_result,
    _ensure_ct2_model,
    _finite_or_zero,
    _normalise_result,
    _temperature_schedule,
)
from asr_pipeline.text_metrics import (
    LOOP_SCORE_THRESHOLD,
    PHRASE_RUN_MIN,
    PhraseRun,
    find_phrase_runs,
    max_phrase_run,
    phrase_run_score,
    repetition_loop_score,
)

# The silence floor is now a config field (TranscriptionConfig.silence_floor),
# not a module constant; the stage gate uses whatever value the config carries.
# The tests below exercise the DEFAULT, so anchor on the dataclass default.
_SILENCE_FLOOR = TranscriptionConfig().silence_floor


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


def test_load_dispatches_to_coherex_backend(monkeypatch):
    monkeypatch.setattr(_CohereXBackend, "load", lambda self, device: None)
    stage = TranscriptionStage(TranscriptionConfig(backend="coherex"))
    stage.load(torch_cpu())
    assert isinstance(stage._backend, _CohereXBackend)


def test_coherex_backend_fail_loud_without_venv_env(monkeypatch):
    # SCOPE §4: a missing isolated venv is a loud crash at load, never a silent
    # fall-back to WhisperX.
    monkeypatch.delenv("COHEREX_VENV_PY", raising=False)
    stage = TranscriptionStage(TranscriptionConfig(backend="coherex"))
    with pytest.raises(RuntimeError, match="COHEREX_VENV_PY"):
        stage.load(torch_cpu())


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


def test_load_signature_coherex_includes_align_model():
    cfg = TranscriptionConfig(backend="coherex", model_name="CohereLabs/x",
                              align_model_name="some/aligner")
    stage = TranscriptionStage(cfg)
    assert stage.load_signature() == ("coherex", "CohereLabs/x", "some/aligner")


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
# Decode knobs reach the backends (the sweep relies on this)
# ---------------------------------------------------------------------------


def test_temperature_schedule_normalisation():
    """Scalar → single-element list (no-fallback); list/tuple → list."""
    assert _temperature_schedule(0.3) == [0.3]
    assert _temperature_schedule([0.0, 0.5]) == [0.0, 0.5]
    assert _temperature_schedule((0.0, 0.5)) == [0.0, 0.5]


class _FakeWhisperModel:
    """Captures the kwargs the whisper backend passes to transcribe()."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return {"text": "x", "segments": [], "language": "pl"}


def test_whisper_backend_passes_decode_knobs():
    """openai-whisper backend forwards every decode knob to model.transcribe()
    with the configured values."""
    cfg = TranscriptionConfig(
        backend="whisper", beam_size=3, temperature=[0.0, 0.4],
        condition_on_previous_text=True, no_speech_threshold=0.5,
        compression_ratio_threshold=2.0, patience=1.5,
    )
    backend = _WhisperBackend(cfg)
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))

    kw = fake.kwargs
    assert kw["beam_size"] == 3
    assert kw["patience"] == 1.5
    assert kw["temperature"] == (0.0, 0.4)        # scheduled as a tuple
    assert kw["condition_on_previous_text"] is True
    assert kw["no_speech_threshold"] == 0.5
    assert kw["compression_ratio_threshold"] == 2.0


def test_whisper_backend_default_knobs_reproduce_current_behaviour():
    """With default config the whisper backend forwards the WhisperX-matched
    defaults — the values that make a baseline run byte-identical."""
    backend = _WhisperBackend(TranscriptionConfig(backend="whisper"))
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))
    kw = fake.kwargs
    assert kw["beam_size"] == 5
    assert kw["patience"] == 1.0
    assert kw["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert kw["condition_on_previous_text"] is False
    assert kw["no_speech_threshold"] == 0.6
    assert kw["compression_ratio_threshold"] == 2.4


# ---------------------------------------------------------------------------
# Anti-hallucination knobs: faster-whisper-only, guarded on the whisper backend
# ---------------------------------------------------------------------------


def test_whisper_backend_default_antihallucination_knobs_are_noop():
    """Default config (0 / 1.0 / None) must NOT forward the faster-whisper-only
    knobs to openai-whisper — they have no equivalent there, so passing them
    would crash or silently substitute behaviour. Default = byte-identical."""
    backend = _WhisperBackend(TranscriptionConfig(backend="whisper"))
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))   # must not raise
    kw = fake.kwargs
    assert "no_repeat_ngram_size" not in kw
    assert "repetition_penalty" not in kw
    assert "hallucination_silence_threshold" not in kw


@pytest.mark.parametrize("field,value,token", [
    ("no_repeat_ngram_size", 3, "no_repeat_ngram_size"),
    ("repetition_penalty", 1.2, "repetition_penalty"),
    ("hallucination_silence_threshold", 2.0, "hallucination_silence_threshold"),
    ("chunk_size", 15, "chunk_size"),   # WhisperX-only VAD knob; no-op for whisper
])
def test_whisper_backend_rejects_nondefault_antihallucination_knob(field, value, token):
    """A WhisperX-only knob set to a non-default value with the openai-whisper
    backend fails loud (SCOPE §4.1: no silent substitution) — and the model's
    transcribe() is never reached."""
    cfg = TranscriptionConfig(backend="whisper", **{field: value})
    backend = _WhisperBackend(cfg)
    fake = _FakeWhisperModel()
    backend._model = fake
    with pytest.raises(ValueError, match=token):
        backend.transcribe(np.zeros(16_000, dtype=np.float32))
    assert fake.kwargs is None       # never reached openai-whisper


def test_whisper_backend_default_length_penalty_not_forwarded():
    """length_penalty defaults to 1.0, but openai-whisper's own default is None
    and 1.0 there is NOT equivalent (different ranker formula). So at the default
    the backend must OMIT the kwarg entirely, leaving openai-whisper's None →
    byte-identical baseline."""
    backend = _WhisperBackend(TranscriptionConfig(backend="whisper"))
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))
    assert "length_penalty" not in fake.kwargs


def test_whisper_backend_forwards_nondefault_length_penalty():
    """A swept length_penalty (!= 1.0) IS forwarded to openai-whisper — it's a
    shared knob (not WhisperX-only), so no reject, just a pass-through."""
    cfg = TranscriptionConfig(backend="whisper", length_penalty=0.8)
    backend = _WhisperBackend(cfg)
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))   # must not raise
    assert fake.kwargs["length_penalty"] == 0.8


@pytest.mark.parametrize("field,value,token", [
    ("suppress_numerals", True, "suppress_numerals"),
    ("vad_onset", 0.4, "vad_onset"),
    ("vad_offset", 0.2, "vad_offset"),
])
def test_whisper_backend_rejects_nondefault_tier2_whisperx_knob(field, value, token):
    """suppress_numerals / vad_onset / vad_offset are WhisperX-only; a non-default
    value with backend=whisper fails loud (SCOPE §4.1) before reaching the model."""
    cfg = TranscriptionConfig(backend="whisper", **{field: value})
    backend = _WhisperBackend(cfg)
    fake = _FakeWhisperModel()
    backend._model = fake
    with pytest.raises(ValueError, match=token):
        backend.transcribe(np.zeros(16_000, dtype=np.float32))
    assert fake.kwargs is None       # never reached openai-whisper


def test_whisper_backend_default_tier2_whisperx_knobs_are_noop():
    """At their defaults the WhisperX-only Tier-2 knobs neither raise nor reach
    openai-whisper — byte-identical baseline on the whisper backend."""
    backend = _WhisperBackend(TranscriptionConfig(backend="whisper"))
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))   # must not raise
    assert "suppress_numerals" not in fake.kwargs
    assert "vad_onset" not in fake.kwargs
    assert "vad_offset" not in fake.kwargs


def test_whisper_backend_retry_knob_does_not_raise_and_is_ignored():
    """Unlike chunk_size (a hard reject), retry_collapsed_chunk_size defaults to
    8 (ON), so the openai-whisper backend must NOT reject it — it logs that the
    knob is WhisperX-only and ignored, then transcribes normally (SCOPE §4.1: a
    visible no-op, not a silent one). Contrast with chunk_size, which raises."""
    cfg = TranscriptionConfig(backend="whisper")     # default retry=8
    assert cfg.retry_collapsed_chunk_size == 8
    backend = _WhisperBackend(cfg)
    fake = _FakeWhisperModel()
    backend._model = fake
    backend.transcribe(np.zeros(16_000, dtype=np.float32))   # must not raise
    assert fake.kwargs is not None                    # reached openai-whisper
    # The retry knob is never forwarded to openai-whisper (no equivalent).
    assert "retry_collapsed_chunk_size" not in fake.kwargs


def test_whisperx_backend_passes_chunk_size_to_transcribe(monkeypatch):
    """The WhisperX backend forwards transcription.chunk_size to the faster-
    whisper pipeline's transcribe() — the knob that bounds the max merged VAD
    segment length (default 30; lower splits over-long segments)."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(backend="whisperx", chunk_size=15, word_timestamps=False)
    backend = _WhisperXBackend(cfg)

    captured = {}

    class _FakeASR:
        def transcribe(self, audio, language, chunk_size):
            captured["chunk_size"] = chunk_size
            captured["language"] = language
            return {"segments": [], "language": language}

    backend._asr = _FakeASR()
    backend.transcribe(np.zeros(16_000, dtype=np.float32))
    assert captured["chunk_size"] == 15
    assert captured["language"] == cfg.language


# ---------------------------------------------------------------------------
# Detect-and-retry for collapsed WhisperX windows
# ---------------------------------------------------------------------------


class _RetryFakeASR:
    """Fake faster-whisper pipeline for the WhisperX retry path.

    The first ``transcribe`` (chunk_size = the configured cs30 pass) returns
    ``first_segments``; every subsequent call (the per-window retry at the small
    chunk) returns ``retry_segments`` and records the sub-clip length so the test
    can confirm the right audio span was sliced.
    """

    def __init__(self, first_segments, retry_segments) -> None:
        self.first_segments = first_segments
        self.retry_segments = retry_segments
        self.calls: list[dict] = []

    def transcribe(self, audio, language, chunk_size):
        self.calls.append({"chunk_size": chunk_size, "n_samples": len(audio)})
        segs = self.first_segments if len(self.calls) == 1 else self.retry_segments
        return {"segments": [dict(s) for s in segs], "language": language}


def _retry_backend(monkeypatch, first_segments, retry_segments, **cfg_kwargs):
    """A _WhisperXBackend wired to a _RetryFakeASR, whisperx import stubbed."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(backend="whisperx", word_timestamps=False, **cfg_kwargs)
    backend = _WhisperXBackend(cfg)
    backend._asr = _RetryFakeASR(first_segments, retry_segments)
    return backend


def test_retry_does_not_fire_on_normal_segment(monkeypatch):
    """A normal multi-word segment is not collapse-eligible: the retry never
    runs (only the one cs30 pass) and the segment passes through unchanged."""
    normal = [{"start": 0.0, "end": 5.0, "text": "to jest zwykłe zdanie po polsku"}]
    backend = _retry_backend(monkeypatch, normal, retry_segments=[])
    out = backend.transcribe(np.zeros(30 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 1                  # no retry pass
    assert out["segments"][0]["text"] == "to jest zwykłe zdanie po polsku"
    assert out["segments"][0]["start"] == 0.0 and out["segments"][0]["end"] == 5.0


def test_retry_fires_and_splices_with_offset_timestamps(monkeypatch):
    """A long near-empty window (25 s, 1 word → 0.04 w/s) collapses: the retry
    fires, its richer output is spliced in, and the recovered timestamps are
    offset by the collapsed window's start."""
    # Window 10.0-35.0 s (dur=25 s) with a single word → collapsed.
    collapsed = [{"start": 10.0, "end": 35.0, "text": "x"}]
    # Retry returns local timestamps (relative to the sub-clip start).
    recovered = [
        {"start": 0.0, "end": 3.0, "text": "odzyskane słowa jeden"},
        {"start": 3.0, "end": 6.0, "text": "odzyskane słowa dwa"},
    ]
    backend = _retry_backend(monkeypatch, collapsed, recovered)
    out = backend.transcribe(np.zeros(40 * 16_000, dtype=np.float32))

    # One cs30 pass + one retry pass.
    assert len(backend._asr.calls) == 2
    assert backend._asr.calls[0]["chunk_size"] == 30          # initial cs30
    assert backend._asr.calls[1]["chunk_size"] == 8           # retry chunk
    # The retry was fed exactly the collapsed window's audio span (10-35 s).
    assert backend._asr.calls[1]["n_samples"] == int(35.0 * 16_000) - int(10.0 * 16_000)

    # Spliced result carries the recovered words, with timestamps offset by 10 s.
    texts = [s["text"] for s in out["segments"]]
    assert texts == ["odzyskane słowa jeden", "odzyskane słowa dwa"]
    assert out["segments"][0]["start"] == 10.0 and out["segments"][0]["end"] == 13.0
    assert out["segments"][1]["start"] == 13.0 and out["segments"][1]["end"] == 16.0


def test_retry_guard_keeps_original_when_not_more_words(monkeypatch):
    """The guard: if the retry yields FEWER/equal words than the collapsed
    original, the original window is kept (a window can't get emptier)."""
    # 4 words over 25 s → 0.16 w/s, still below 0.7 → collapse-eligible.
    collapsed = [{"start": 0.0, "end": 25.0, "text": "cztery słowa oryginalne tu"}]
    # Retry recovers only 2 words (fewer) → guard keeps the original.
    fewer = [{"start": 0.0, "end": 2.0, "text": "dwa słowa"}]
    backend = _retry_backend(monkeypatch, collapsed, fewer)
    out = backend.transcribe(np.zeros(30 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                  # retry ran
    # ...but its output was rejected; the original survives unchanged.
    assert out["segments"] == [
        {"start": 0.0, "end": 25.0, "text": "cztery słowa oryginalne tu"}
    ]


def test_retry_disabled_when_chunk_size_zero(monkeypatch):
    """retry_collapsed_chunk_size=0 disables the retry entirely: even a collapsed
    window passes straight through (no second transcribe call)."""
    collapsed = [{"start": 0.0, "end": 25.0, "text": "x"}]
    backend = _retry_backend(
        monkeypatch, collapsed, retry_segments=[], retry_collapsed_chunk_size=0
    )
    out = backend.transcribe(np.zeros(30 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 1                  # no retry pass
    assert out["segments"][0]["text"] == "x"


def test_retry_resorts_spliced_segments_by_start(monkeypatch):
    """After splicing, segments are re-sorted by start time so a later-window
    splice can't leave the list out of order."""
    # Two windows: a normal one at 30-33 s and a collapsed one at 0-25 s. The
    # collapsed (earlier) window is listed second to force a re-sort.
    first = [
        {"start": 30.0, "end": 33.0, "text": "późne zwykłe zdanie tutaj"},
        {"start": 0.0, "end": 25.0, "text": "x"},
    ]
    recovered = [{"start": 0.0, "end": 4.0, "text": "wcześnie odzyskane słowa pięć sześć"}]
    backend = _retry_backend(monkeypatch, first, recovered)
    out = backend.transcribe(np.zeros(40 * 16_000, dtype=np.float32))
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts)
    assert out["segments"][0]["text"] == "wcześnie odzyskane słowa pięć sześć"
    assert out["segments"][-1]["text"] == "późne zwykłe zdanie tutaj"


def test_retry_runs_before_alignment_so_align_sees_recovered_segments(monkeypatch):
    """Ordering contract: the collapse retry runs on the RAW (pre-alignment)
    segments, then `whisperx.align` aligns the SPLICED result. If the retry ran
    after alignment, the alignment would re-segment the collapsed (near-empty)
    window and the recovered words would never get word timestamps. Stub
    `whisperx.align` to record exactly what it's handed and assert it's the
    recovered segments, not the collapsed original.
    """
    import sys
    import types

    fake_whisperx = types.ModuleType("whisperx")
    seen = {}

    def fake_align(segments, model, metadata, audio, device, return_char_alignments):
        seen["segments"] = segments
        return {"segments": segments, "word_segments": []}

    fake_whisperx.align = fake_align
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    collapsed = [{"start": 10.0, "end": 35.0, "text": "x"}]      # 25 s, 1 word
    recovered = [
        {"start": 0.0, "end": 3.0, "text": "odzyskane słowa jeden"},
        {"start": 3.0, "end": 6.0, "text": "odzyskane słowa dwa"},
    ]
    cfg = TranscriptionConfig(backend="whisperx", word_timestamps=True)
    backend = _WhisperXBackend(cfg)
    backend._asr = _RetryFakeASR(collapsed, recovered)
    backend._align_model = object()
    backend._align_metadata = {"meta": True}
    backend._device_str = "cpu"

    backend.transcribe(np.zeros(40 * 16_000, dtype=np.float32))

    # align must have been handed the recovered (spliced) segments — with the
    # collapsed window's start offset applied — not the original "x" placeholder.
    texts = [s["text"] for s in seen["segments"]]
    assert texts == ["odzyskane słowa jeden", "odzyskane słowa dwa"]
    assert seen["segments"][0]["start"] == 10.0      # offset by the window start
    assert all(s["text"] != "x" for s in seen["segments"])


def test_whisperx_backend_builds_asr_options(monkeypatch):
    """WhisperX backend merges the decode knobs into asr_options, mapping the
    temperature schedule onto the `temperatures` (plural) key WhisperX/
    faster-whisper expect. Captures the dict passed to whisperx.load_model."""
    import sys
    import types

    captured = {}

    fake_whisperx = types.ModuleType("whisperx")

    def fake_load_model(model_path, device, compute_type, language, asr_options,
                        vad_options):
        captured["asr_options"] = asr_options
        captured["vad_options"] = vad_options
        return object()    # stand-in ASR pipeline; load() doesn't call it

    def fake_load_align_model(language_code, device, model_name):
        return object(), {"meta": True}

    fake_whisperx.load_model = fake_load_model
    fake_whisperx.load_align_model = fake_load_align_model
    # _WhisperXBackend.load reads DEFAULT_ALIGN_MODELS_* from this submodule
    # when align_model_name is None (to log the resolved aligner).
    fake_alignment = types.ModuleType("whisperx.alignment")
    fake_alignment.DEFAULT_ALIGN_MODELS_TORCH = {}
    fake_alignment.DEFAULT_ALIGN_MODELS_HF = {"pl": "jonatasgrosman/x"}
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", fake_alignment)

    cfg = TranscriptionConfig(
        backend="whisperx", beam_size=2, temperature=0.0,
        condition_on_previous_text=True, no_speech_threshold=0.4,
        compression_ratio_threshold=1.8, patience=1.2,
        no_repeat_ngram_size=3, repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
        suppress_numerals=True, length_penalty=1.1,
        vad_onset=0.4, vad_offset=0.2,
    )
    backend = _WhisperXBackend(cfg)
    backend.load(torch_cpu())

    opts = captured["asr_options"]
    assert opts["beam_size"] == 2
    assert opts["patience"] == 1.2
    assert opts["temperatures"] == [0.0]          # plural key, scalar→[scalar]
    assert "temperature" not in opts              # never the singular key
    assert opts["condition_on_previous_text"] is True
    assert opts["no_speech_threshold"] == 0.4
    assert opts["compression_ratio_threshold"] == 1.8
    assert opts["initial_prompt"] == cfg.initial_prompt
    # Anti-hallucination knobs reach the dict under their faster-whisper names.
    assert opts["no_repeat_ngram_size"] == 3
    assert opts["repetition_penalty"] == 1.2
    assert opts["hallucination_silence_threshold"] == 2.0
    # Tier-2: suppress_numerals + length_penalty ride in asr_options; vad_onset/
    # vad_offset go in a separate vad_options dict load_model merges over its own.
    assert opts["suppress_numerals"] is True
    assert opts["length_penalty"] == 1.1
    assert captured["vad_options"] == {"vad_onset": 0.4, "vad_offset": 0.2}


def test_whisperx_backend_default_asr_options_match_whisperx_defaults(monkeypatch):
    """Default config → asr_options whose values equal WhisperX's own
    default_asr_options, so merging them changes nothing (byte-identical
    baseline). Evidence: whisperx/asr.py load_model default_asr_options."""
    import sys
    import types

    captured = {}
    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_model = (
        lambda model_path, device, compute_type, language, asr_options, vad_options: (
            captured.__setitem__("asr_options", asr_options)
            or captured.__setitem__("vad_options", vad_options)
            or object()
        )
    )
    fake_whisperx.load_align_model = lambda language_code, device, model_name: (object(), {})
    fake_alignment = types.ModuleType("whisperx.alignment")
    fake_alignment.DEFAULT_ALIGN_MODELS_TORCH = {}
    fake_alignment.DEFAULT_ALIGN_MODELS_HF = {"pl": "jonatasgrosman/x"}
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", fake_alignment)

    backend = _WhisperXBackend(TranscriptionConfig(backend="whisperx"))
    backend.load(torch_cpu())

    opts = captured["asr_options"]
    assert opts["beam_size"] == 5
    assert opts["patience"] == 1.0
    assert opts["temperatures"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert opts["condition_on_previous_text"] is False
    assert opts["no_speech_threshold"] == 0.6
    assert opts["compression_ratio_threshold"] == 2.4
    # Anti-hallucination knobs at their defaults equal WhisperX's own
    # default_asr_options values, so merging them is a no-op (byte-identical).
    assert opts["no_repeat_ngram_size"] == 0
    assert opts["repetition_penalty"] == 1.0
    assert opts["hallucination_silence_threshold"] is None
    # Tier-2 defaults also equal WhisperX's own defaults → no-op merge.
    assert opts["suppress_numerals"] is False
    assert opts["length_penalty"] == 1
    assert captured["vad_options"] == {"vad_onset": 0.500, "vad_offset": 0.363}


# ---------------------------------------------------------------------------
# Conditional repetition-loop retry (WhisperX)
# ---------------------------------------------------------------------------


# A merged window transcribed as one token repeated — the hallucination the
# collapse detector cannot catch. >= LOOP_MIN_TOKENS tokens, top token >= 8×.
_LOOP_TEXT = "No " + "tak " * 20        # "tak" ×20 among 21 tokens → score ~0.95


def test_loop_score_flags_synthetic_loop():
    """A short phrase repeated 20+ times scores above the threshold."""
    ls = repetition_loop_score("No " + "tak, " * 25)
    assert ls.top_token == "tak"
    assert ls.top_count == 25
    assert ls.score >= LOOP_SCORE_THRESHOLD
    assert ls.score > 0.9


@pytest.mark.parametrize("text", [
    "zawsze w tej tej koszuli chodzi",          # mild disfluency, short
    "no tak, no tak",                            # repeated once
    # long but genuinely varied — exercises the non-looping path, not just the
    # min-token gate:
    "to jest zupełnie normalne zdanie po polsku bez żadnych powtórzeń wcale naprawdę",
])
def test_loop_score_below_threshold_for_natural_speech(text):
    assert repetition_loop_score(text).score < LOOP_SCORE_THRESHOLD


def test_loop_score_gated_by_min_tokens():
    """A pure repeat that is still short (< LOOP_MIN_TOKENS) is gated to 0.0, so
    natural short repeats never fire the detector."""
    ls = repetition_loop_score("tak tak tak tak tak")
    assert ls.n_tokens == 5 and ls.top_count == 5
    assert ls.score == 0.0


def test_loop_score_gate_sits_between_disfluency_and_real_loops():
    """The min-top-count gate splits the corpus' observed gap: the longest
    GENUINE repeat WhisperX transcribes is ×8-9 (dev 4f9251fe "Tak, tak, ...",
    which a ×8 gate clipped into real deletions), while the tightest REAL loop
    is `takie`×14 (test 150d1ccc). ×9 must be gated to 0.0; ×14 must fire."""
    disfluency = "Tak jak ten, na przykład. " + "Tak, " * 9   # 'tak' ×10/14ish
    ls = repetition_loop_score(disfluency)
    assert ls.top_token == "tak" and ls.top_count < 12
    assert ls.score == 0.0
    loop = "No bo wiesz, to jest " + "takie " * 14 + "no i tyle wlasnie"
    ls = repetition_loop_score(loop)
    assert ls.top_token == "takie" and ls.top_count == 14
    assert ls.score >= LOOP_SCORE_THRESHOLD


@dataclass
class _FakeOptions:
    """Stand-in for faster-whisper's TranscriptionOptions carrying only the field
    the loop retry overrides. A dataclass so `dataclasses.replace` works — the
    exact route _retry_loops (and WhisperX itself, for suppress_numerals) uses."""

    no_repeat_ngram_size: int = 0


class _LoopFakeASR:
    """Fake faster-whisper pipeline for the loop-retry path.

    First transcribe() (the main pass) returns first_segments; each later call
    (a per-window loop retry) returns retry_segments and records the
    no_repeat_ngram_size live on self.options at call time — proving the override
    reached the decode and was restored afterwards.
    """

    def __init__(self, first_segments, retry_segments) -> None:
        self.first_segments = first_segments
        self.retry_segments = retry_segments
        self.options = _FakeOptions()
        self.calls: list[dict] = []
        self.retry_ngram_seen: list[int] = []

    def transcribe(self, audio, language, chunk_size):
        self.calls.append({"chunk_size": chunk_size, "n_samples": len(audio)})
        if len(self.calls) == 1:
            segs = self.first_segments
        else:
            self.retry_ngram_seen.append(self.options.no_repeat_ngram_size)
            segs = self.retry_segments
        return {"segments": [dict(s) for s in segs], "language": language}


def _loop_backend(monkeypatch, first_segments, retry_segments, **cfg_kwargs):
    """A _WhisperXBackend wired to a _LoopFakeASR, loop_retry ON, collapse OFF.

    retry_collapsed_chunk_size=0 isolates the loop retry so call #2 is
    deterministically the loop-retry pass.
    """
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(
        backend="whisperx", word_timestamps=False, loop_retry=True,
        retry_collapsed_chunk_size=0, **cfg_kwargs,
    )
    backend = _WhisperXBackend(cfg)
    backend._asr = _LoopFakeASR(first_segments, retry_segments)
    return backend


def test_loop_retry_default_off_does_not_run(monkeypatch):
    """loop_retry defaults to False → the loop-retry pass never executes even on a
    clearly-looping window; the segment passes through unchanged (byte-identical
    shipped behaviour). Only the single main transcribe call happens."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(backend="whisperx", word_timestamps=False,
                              retry_collapsed_chunk_size=0)
    assert cfg.loop_retry is False
    backend = _WhisperXBackend(cfg)
    backend._asr = _LoopFakeASR(
        [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}], retry_segments=[]
    )
    out = backend.transcribe(np.zeros(12 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 1                 # no retry pass
    assert out["segments"][0]["text"] == _LOOP_TEXT


def test_loop_retry_accepts_when_retry_breaks_loop(monkeypatch):
    """A looped window is retried with the ngram override; a clean retry is
    spliced in with timestamps offset by the window start, and the override is
    restored afterwards."""
    looped = [{"start": 4.0, "end": 14.0, "text": _LOOP_TEXT}]
    clean = [{"start": 0.0, "end": 5.0,
              "text": "to jest zupełnie normalne zdanie po polsku bez powtórzeń naprawdę wcale"}]
    backend = _loop_backend(monkeypatch, looped, clean, loop_retry_ngram=3)
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                       # retry ran
    assert backend._asr.retry_ngram_seen == [3]               # override reached decode
    assert backend._asr.options.no_repeat_ngram_size == 0     # ...and was restored
    # Retry was fed exactly the looped window's audio span (4-14 s).
    assert backend._asr.calls[1]["n_samples"] == int(14.0 * 16_000) - int(4.0 * 16_000)
    # Clean text spliced in, timestamps offset by the window start (4 s).
    assert out["segments"][0]["text"].startswith("to jest")
    assert out["segments"][0]["start"] == 4.0 and out["segments"][0]["end"] == 9.0


def test_loop_retry_rejects_when_retry_still_loops(monkeypatch):
    """Inverted accept guard: if the retry still loops (score >= threshold) it is
    rejected and the original window kept."""
    looped = [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}]
    still = [{"start": 0.0, "end": 5.0, "text": "nie " * 15}]      # still a loop
    backend = _loop_backend(monkeypatch, looped, still)
    out = backend.transcribe(np.zeros(12 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 2                  # retry ran...
    # ...but was rejected; the original survives unchanged.
    assert out["segments"] == [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}]


def test_loop_retry_keeps_original_when_retry_empty(monkeypatch):
    """A retry that produces no content never empties a window that had content —
    the original is kept."""
    looped = [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}]
    backend = _loop_backend(monkeypatch, looped, retry_segments=[])
    out = backend.transcribe(np.zeros(12 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 2
    assert out["segments"] == [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}]


def test_loop_retry_leaves_normal_segments_untouched(monkeypatch):
    """Only looped windows are retried; a normal window in the same stream is not
    re-decoded and stays in place after the re-sort."""
    segs = [
        {"start": 0.0, "end": 4.0, "text": "zwykłe zdanie które nie jest pętlą wcale"},
        {"start": 5.0, "end": 15.0, "text": _LOOP_TEXT},
    ]
    clean = [{"start": 0.0, "end": 3.0,
              "text": "poprawnie odzyskane słowa bez pętli tutaj naprawdę teraz zdanie"}]
    backend = _loop_backend(monkeypatch, segs, clean)
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 2                  # exactly one retry
    texts = [s["text"] for s in out["segments"]]
    assert texts[0] == "zwykłe zdanie które nie jest pętlą wcale"    # untouched, first
    assert texts[1].startswith("poprawnie odzyskane")                # spliced, offset 5 s
    assert out["segments"][1]["start"] == 5.0


def test_loop_retry_restores_live_global_ngram(monkeypatch):
    """The retry overrides no_repeat_ngram_size only for its own decode and
    restores whatever global value was live before — the global knob and the
    loop-retry ngram stay independent (task interaction rule 5)."""
    looped = [{"start": 0.0, "end": 10.0, "text": _LOOP_TEXT}]
    clean = [{"start": 0.0, "end": 3.0,
              "text": "czyste zdanie bez pętli po polsku naprawdę teraz już koniec"}]
    backend = _loop_backend(monkeypatch, looped, clean, loop_retry_ngram=4)
    backend._asr.options.no_repeat_ngram_size = 2        # a live global setting
    backend.transcribe(np.zeros(12 * 16_000, dtype=np.float32))
    assert backend._asr.retry_ngram_seen == [4]          # retry used loop_retry_ngram
    assert backend._asr.options.no_repeat_ngram_size == 2  # global restored, not 4


@pytest.mark.parametrize("backend_name", ["whisper", "coherex"])
def test_loop_retry_rejected_on_non_whisperx_backend(backend_name):
    """loop_retry re-decodes via faster-whisper's TranscriptionOptions, which only
    the whisperx backend carries; loop_retry=True on any other backend fails loud
    at stage load — before any model is loaded — never a silent no-op (SCOPE
    §4.1). Backend-level (not config-level) so a sweep arm may override loop_retry
    onto a whisperx default.yaml base without re-declaring the backend."""
    stage = TranscriptionStage(
        TranscriptionConfig(backend=backend_name, loop_retry=True)
    )
    with pytest.raises(ValueError, match="loop_retry"):
        stage.load(torch_cpu())


def test_loop_retry_accepted_on_whisperx_backend(monkeypatch):
    """loop_retry=True + whisperx passes the guard and dispatches normally (the
    backend's own load is stubbed so no model is fetched)."""
    monkeypatch.setattr(_WhisperXBackend, "load", lambda self, device: None)
    stage = TranscriptionStage(
        TranscriptionConfig(backend="whisperx", loop_retry=True)
    )
    stage.load(torch_cpu())        # must not raise
    assert isinstance(stage._backend, _WhisperXBackend)


# ---------------------------------------------------------------------------
# Multi-token phrase-loop metric (text_metrics.find_phrase_runs / phrase_run_score)
# ---------------------------------------------------------------------------


def test_phrase_run_fires_on_three_token_phrase_x6():
    """"Tak, to jest..." ×6 — a repeated 3-token phrase the dominant-token metric
    is blind to (each token caps at ~1/3) — is detected: run 6, correct span."""
    pr = phrase_run_score("Tak, to jest... " * 6)
    assert pr.run == 6
    assert pr.phrase == "tak to jest"
    assert (pr.start, pr.end) == (0, 18)          # 6 repeats × 3 tokens
    # And the single-token detector really is blind to it (< threshold).
    assert repetition_loop_score("Tak, to jest... " * 6).score < LOOP_SCORE_THRESHOLD


def test_phrase_run_fires_on_five_token_phrase_x14():
    """A 5-token phrase repeated 14× (the 5bab2c34 shape) → one run of 14."""
    tokens = ["jak", "pojedziemy", "do", "dekathlonu", "o"] * 14
    runs = find_phrase_runs(tokens)
    assert len(runs) == 1
    assert runs[0].run == 14
    assert (runs[0].start, runs[0].end) == (0, 70)


def test_phrase_run_two_repeats_below_min_run():
    """A phrase repeated only twice does not reach the default min_run gate (4)."""
    assert find_phrase_runs(["a", "b", "c", "a", "b", "c"]) == []
    assert phrase_run_score("a b c a b c").run < PHRASE_RUN_MIN


def test_phrase_run_ignores_uniform_token_run():
    """"no no no ..." is a SINGLE-token loop (repetition_loop_score's job) — the
    >=2-distinct-token guard keeps the phrase detector off it."""
    assert find_phrase_runs(["no"] * 12) == []
    assert phrase_run_score("no " * 12) == PhraseRun(1, "", 0, 0)


def test_phrase_run_unique_sentinels_break_runs():
    """A unique sentinel token between repeats forbids a run from crossing it —
    uniqueness alone (no special-casing) stops the n-gram match."""
    tokens = []
    for k in range(6):
        tokens += ["a", "b", "c", f"\x00{k}"]
    assert find_phrase_runs(tokens) == []


def test_phrase_run_merges_period_and_multiple_detections():
    """One loop is detected at its period (n=3) AND at a multiple (n=6); the
    overlapping token spans merge into ONE span carrying the max run."""
    runs = find_phrase_runs(["a", "b", "c"] * 8)      # 24 tokens
    assert len(runs) == 1
    assert runs[0].run == 8                           # the period-3 detection wins
    assert (runs[0].start, runs[0].end) == (0, 24)


def test_max_phrase_run_sentinel_values():
    """The no-repeat / empty sentinels: empty stream → run 0; a stream with no
    qualifying repeat → run 1; both with empty phrase and zero span."""
    assert max_phrase_run([]) == PhraseRun(0, "", 0, 0)
    assert max_phrase_run(["a"]) == PhraseRun(1, "", 0, 0)
    assert phrase_run_score("") == PhraseRun(0, "", 0, 0)
    assert phrase_run_score("zupełnie normalne zdanie bez powtórzeń") == PhraseRun(1, "", 0, 0)


# ---------------------------------------------------------------------------
# Conditional phrase-loop retry (WhisperX)
# ---------------------------------------------------------------------------


def _phrase_backend(monkeypatch, first_segments, retry_segments, **cfg_kwargs):
    """A _WhisperXBackend wired to a _LoopFakeASR, loop_retry_phrase ON.

    Reuses _LoopFakeASR (returns first_segments on call 1, retry_segments after,
    records the live no_repeat_ngram_size). retry_collapsed_chunk_size=0 and
    loop_retry=False isolate the phrase-loop path so call #2 is deterministically
    the phrase-loop retry pass.
    """
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(
        backend="whisperx", word_timestamps=False, loop_retry_phrase=True,
        loop_retry=False, retry_collapsed_chunk_size=0, **cfg_kwargs,
    )
    backend = _WhisperXBackend(cfg)
    backend._asr = _LoopFakeASR(first_segments, retry_segments)
    return backend


# A clean, varied retry output — no phrase run, no dominant token.
_CLEAN_RETRY = [{"start": 0.0, "end": 3.0,
                 "text": "to jest zupełnie normalne zdanie po polsku bez powtórzeń naprawdę"}]


def test_phrase_loop_cross_segment_run_detected_and_spliced(monkeypatch):
    """The 5bab2c34 shape: 14 identical consecutive segments (each one 5-token
    phrase, tiny gaps) form a cross-segment phrase run → the whole span is
    retried and a clean retry spliced in, offset by the window start."""
    seg_text = "jak pojedziemy do dekathlonu o"
    looped = [{"start": float(k), "end": float(k + 1), "text": seg_text}
              for k in range(14)]
    backend = _phrase_backend(monkeypatch, looped, _CLEAN_RETRY, loop_retry_ngram=3)
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                      # retry ran
    assert backend._asr.retry_ngram_seen == [3]              # override reached decode
    assert backend._asr.options.no_repeat_ngram_size == 0    # ...and was restored
    # Window = [seg0.start=0, seg13.end=14]; retry fed exactly that span.
    assert backend._asr.calls[1]["n_samples"] == int(14.0 * 16_000)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"].startswith("to jest")
    assert out["segments"][0]["start"] == 0.0                # offset by window start (0)


def test_phrase_loop_within_one_segment_detected_and_spliced(monkeypatch):
    """The 152ed870 shape: a phrase repeated ×6 INSIDE one segment is detected and
    the single segment retried; the clean retry is spliced offset by its start."""
    looped = [{"start": 41.4, "end": 57.2, "text": "tak to jest " * 6}]
    backend = _phrase_backend(monkeypatch, looped, _CLEAN_RETRY)
    out = backend.transcribe(np.zeros(60 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2
    assert backend._asr.calls[1]["n_samples"] == int(57.2 * 16_000) - int(41.4 * 16_000)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"].startswith("to jest")
    assert out["segments"][0]["start"] == 41.4               # offset by window start


def test_phrase_loop_growth_guard_rejects_counting_evasion(monkeypatch):
    """The 152ed870 failure mode: under the ngram constraint the decoder evades
    BOTH loop detectors by counting ("D1. D2. ... D78.") — every repeat
    textually distinct (no phrase run), every token distinct (no dominant
    token). Only the length bound catches it: a loop repair may never emit
    more tokens than the loop it replaces."""
    logged = []
    monkeypatch.setattr(
        "asr_pipeline.stages.transcription._log", lambda msg: logged.append(msg)
    )
    looped = [{"start": 41.4, "end": 57.2, "text": "tak to jest " * 6}]   # 18 tokens
    counter = [{"start": 0.0, "end": 15.0,
                "text": " ".join(f"D{i}." for i in range(1, 41))}]        # 40 tokens
    backend = _phrase_backend(monkeypatch, looped, counter)
    out = backend.transcribe(np.zeros(60 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                      # retry ran...
    assert out["segments"] == looped                         # ...but was rejected
    assert any("grew the window (18 -> 40 tokens)" in m for m in logged)


def test_loop_retry_growth_guard_rejects_counting_evasion(monkeypatch):
    """The same counting-evasion length bound on the single-token loop path
    (the hazard is latent there too — same ngram-constrained retry decode)."""
    logged = []
    monkeypatch.setattr(
        "asr_pipeline.stages.transcription._log", lambda msg: logged.append(msg)
    )
    looped = [{"start": 0.0, "end": 20.0,
               "text": "no " + "tak " * 14}]                 # 15 tokens, score 14/15
    counter = [{"start": 0.0, "end": 15.0,
                "text": " ".join(f"D{i}." for i in range(1, 41))}]        # 40 tokens
    backend = _loop_backend(monkeypatch, looped, counter)
    out = backend.transcribe(np.zeros(25 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2
    assert out["segments"] == looped
    assert any("grew the window (15 -> 40 tokens)" in m for m in logged)


def test_phrase_loop_two_runs_in_one_segment_merge_to_one_retry(monkeypatch):
    """Two token-disjoint phrase runs living in the SAME segment merge into one
    retry interval — the window is retried once, never spliced twice (a double
    splice would corrupt segment indices and duplicate retry content)."""
    text = ("tak to jest " * 6
            + "zupełnie inne słowa w środku "
            + "raz dwa trzy " * 6)
    looped = [{"start": 0.0, "end": 30.0, "text": text}]
    backend = _phrase_backend(monkeypatch, looped, _CLEAN_RETRY)
    out = backend.transcribe(np.zeros(35 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                      # exactly ONE retry pass
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"].startswith("to jest")


def test_phrase_loop_accept_guard_rejects_when_retry_still_phrase_loops(monkeypatch):
    """If the retry still carries a phrase loop it is rejected, the original span
    is kept, and a REJECTED line is logged."""
    logged = []
    monkeypatch.setattr(
        "asr_pipeline.stages.transcription._log", lambda msg: logged.append(msg)
    )
    looped = [{"start": 0.0, "end": 18.0, "text": "tak to jest " * 6}]
    still = [{"start": 0.0, "end": 10.0, "text": "raz dwa trzy " * 6}]   # still a phrase loop
    backend = _phrase_backend(monkeypatch, looped, still)
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))

    assert len(backend._asr.calls) == 2                      # retry ran...
    assert out["segments"] == [{"start": 0.0, "end": 18.0, "text": "tak to jest " * 6}]
    assert any("phrase-loop-retry REJECTED" in m for m in logged)


def test_phrase_loop_keeps_original_when_retry_empty(monkeypatch):
    """A retry that would empty a window that had content is rejected."""
    looped = [{"start": 0.0, "end": 18.0, "text": "tak to jest " * 6}]
    backend = _phrase_backend(monkeypatch, looped, retry_segments=[])
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 2
    assert out["segments"] == [{"start": 0.0, "end": 18.0, "text": "tak to jest " * 6}]


def test_phrase_loop_long_gap_prevents_cross_segment_join(monkeypatch):
    """Identical segments separated by a >_PHRASE_JOIN_MAX_GAP_S pause do NOT join
    into one run (a genuine phrase re-said after a long pause) — no retry fires."""
    seg_text = "jak pojedziemy do dekathlonu o"
    # 1 s segments spaced 5 s apart → 4 s gaps > 2.0 s → sentinel-separated.
    looped = [{"start": float(k * 5), "end": float(k * 5 + 1), "text": seg_text}
              for k in range(6)]
    backend = _phrase_backend(monkeypatch, looped, _CLEAN_RETRY)
    out = backend.transcribe(np.zeros(30 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 1                      # no retry pass
    assert [s["text"] for s in out["segments"]] == [seg_text] * 6


@pytest.mark.parametrize("backend_name", ["whisper", "coherex"])
def test_phrase_loop_rejected_on_non_whisperx_backend(backend_name):
    """loop_retry_phrase=True on a non-whisperx backend fails loud at stage load —
    before any model loads — never a silent no-op (SCOPE §4.1)."""
    stage = TranscriptionStage(
        TranscriptionConfig(backend=backend_name, loop_retry_phrase=True)
    )
    with pytest.raises(ValueError, match="loop_retry_phrase"):
        stage.load(torch_cpu())


def test_phrase_loop_default_off_does_not_run(monkeypatch):
    """loop_retry_phrase defaults False → the phrase-loop pass never runs even on a
    clearly phrase-looping window; segments pass through untouched."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    cfg = TranscriptionConfig(backend="whisperx", word_timestamps=False,
                              retry_collapsed_chunk_size=0)
    assert cfg.loop_retry_phrase is False
    backend = _WhisperXBackend(cfg)
    looped = [{"start": 0.0, "end": 18.0, "text": "tak to jest " * 6}]
    backend._asr = _LoopFakeASR(looped, retry_segments=[])
    out = backend.transcribe(np.zeros(20 * 16_000, dtype=np.float32))
    assert len(backend._asr.calls) == 1                      # no retry pass
    assert out["segments"][0]["text"] == "tak to jest " * 6


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
