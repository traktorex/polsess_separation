"""Unit tests for Stage 1 diarization (asr_pipeline/stages/diarization.py).

Exercises the stage-owned logic — the run-before-load / audio-None guards, the
segment/overlap DataFrame construction (incl. the empty-input column guard and
3-dp rounding), the pyannote 3.x/4.x result duck-typing, load_signature, the
empty-token guard, and the spill schema — all with a fake pipeline object. No
real pyannote model is loaded; the model forward is a third-party boundary.
"""

import json

import numpy as np
import pytest
import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.diarization import DiarizationStage

MODEL_ID = "pyannote/speaker-diarization-3.1"


class _Seg:
    """A pyannote-Segment stand-in: .start / .end / .duration."""

    def __init__(self, start, end, duration=None):
        self.start = start
        self.end = end
        self.duration = duration if duration is not None else end - start


class _FakeDiar:
    """Bare-Annotation stand-in (pyannote 3.x shape)."""

    def __init__(self, tracks, overlaps):
        self._tracks = tracks       # list of (_Seg, speaker_label)
        self._overlaps = overlaps   # list of _Seg

    def itertracks(self, yield_label=False):
        for seg, label in self._tracks:
            yield (seg, "_", label) if yield_label else (seg, "_")

    def get_overlap(self):
        return self._overlaps


class _FakeWrapper:
    """DiarizeOutput stand-in (pyannote 4.x shape) — wraps the Annotation."""

    def __init__(self, diar):
        self.speaker_diarization = diar


class _FakePipeline:
    """Callable stand-in for a loaded pyannote Pipeline."""

    def __init__(self, diar):
        self.diar = diar
        self.calls = []

    def __call__(self, inputs, num_speakers=None):
        self.calls.append((inputs, num_speakers))
        return self.diar


def _stage(diar, model_id=MODEL_ID, num_speakers=2) -> DiarizationStage:
    stage = DiarizationStage(DiarizationConfig(model_id=model_id, num_speakers=num_speakers))
    stage._pipeline = _FakePipeline(diar)   # inject; bypass load()
    return stage


def _ctx(audio=None, sr=16_000) -> PipelineContext:
    ctx = PipelineContext(sample_rate=sr)
    ctx.audio = np.zeros(sr, dtype=np.float32) if audio is None else audio
    return ctx


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_run_before_load_raises():
    stage = DiarizationStage(DiarizationConfig())   # no _pipeline
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx())


def test_run_audio_none_raises():
    stage = _stage(_FakeDiar([], []))
    ctx = PipelineContext()
    ctx.audio = None
    with pytest.raises(RuntimeError, match="audio is None"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# run() — segment/overlap construction
# ---------------------------------------------------------------------------


def test_run_builds_segments_and_overlaps():
    diar = _FakeDiar(
        tracks=[(_Seg(1.0, 2.0), "SPEAKER_00"), (_Seg(3.5, 4.25), "SPEAKER_01")],
        overlaps=[_Seg(1.5, 1.8)],
    )
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)

    seg = ctx.diarization.segments_df
    assert list(seg.columns) == ["start", "end", "duration", "speaker"]
    assert len(seg) == 2
    assert seg.iloc[0]["speaker"] == "SPEAKER_00"
    assert seg.iloc[1]["end"] == 4.25
    ovl = ctx.diarization.overlaps_df
    assert len(ovl) == 1
    assert ovl.iloc[0]["start"] == 1.5
    assert ctx.diarization.total_duration_s == pytest.approx(1.0)   # 16000 / 16000
    # num_speakers threaded into the pipeline call.
    assert stage._pipeline.calls[0][1] == 2


def test_run_rounds_to_3dp_and_reads_duration_attribute():
    # duration set independent of end-start to prove the stage reads the
    # attribute rather than recomputing it.
    diar = _FakeDiar(tracks=[(_Seg(1.23456, 2.34567, duration=0.98765), "A")],
                     overlaps=[_Seg(1.111111, 1.222222)])
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)
    row = ctx.diarization.segments_df.iloc[0]
    assert row["start"] == 1.235
    assert row["end"] == 2.346
    assert row["duration"] == 0.988
    assert ctx.diarization.overlaps_df.iloc[0]["start"] == 1.111


def test_run_empty_diarization_keeps_columns():
    stage = _stage(_FakeDiar([], []))
    ctx = _ctx()
    stage.run(ctx)
    seg = ctx.diarization.segments_df
    ovl = ctx.diarization.overlaps_df
    assert len(seg) == 0
    assert list(seg.columns) == ["start", "end", "duration", "speaker"]
    assert len(ovl) == 0
    assert list(ovl.columns) == ["start", "end", "duration"]


def test_run_handles_pyannote4_wrapper_result():
    diar = _FakeDiar([(_Seg(0.0, 1.0), "A")], [])
    stage = DiarizationStage(DiarizationConfig())
    stage._pipeline = _FakePipeline(_FakeWrapper(diar))   # .speaker_diarization arm
    ctx = _ctx()
    stage.run(ctx)
    assert len(ctx.diarization.segments_df) == 1
    assert ctx.diarization.segments_df.iloc[0]["speaker"] == "A"


# ---------------------------------------------------------------------------
# load_signature + load guard
# ---------------------------------------------------------------------------


def test_load_signature_is_model_id_only():
    stage = DiarizationStage(DiarizationConfig(model_id=MODEL_ID, num_speakers=3))
    assert stage.load_signature() == (MODEL_ID,)   # num_speakers excluded


def test_load_empty_token_raises(monkeypatch):
    # The guard sits behind `from pyannote.audio import Pipeline`, so the import
    # must succeed first — skip on a pyannote-absent machine rather than fake it.
    pytest.importorskip("pyannote.audio")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    stage = DiarizationStage(DiarizationConfig(hf_token=""))
    with pytest.raises(RuntimeError, match="hf_token"):
        stage.load(torch.device("cpu"))


# ---------------------------------------------------------------------------
# spill — the {segments, overlaps} schema (NOT the eval-facing {turns})
# ---------------------------------------------------------------------------


def test_spill_writes_segments_overlaps_schema(tmp_path):
    diar = _FakeDiar([(_Seg(1.0, 2.0), "A")], [_Seg(1.5, 1.8)])
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)
    stage.spill(ctx, tmp_path)

    payload = json.loads((tmp_path / "diarization.json").read_text())
    assert set(payload) == {"segments", "overlaps", "total_duration_s"}
    assert "turns" not in payload          # deliberately NOT io.py's eval schema
    assert payload["segments"][0]["speaker"] == "A"
    assert len(payload["overlaps"]) == 1


def test_spill_noop_when_no_diarization(tmp_path):
    stage = DiarizationStage(DiarizationConfig())
    stage.spill(PipelineContext(), tmp_path)        # ctx.diarization is None
    assert list(tmp_path.iterdir()) == []
