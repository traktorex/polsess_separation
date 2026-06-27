"""Unit tests for disagreement-aware diarization fusion (option 3,
asr_pipeline/stages/fusion_diarization.py).

The merge rule is exercised as a PURE function (`fuse_labels`) on synthetic
raw (pass-1) + enhanced (pass-2) segment frames — no pyannote download. The
stage guards (run-before-load, single-speaker no-op, enhanced-None assert) are
tested with the pass-2 pipeline stubbed to return a canned Annotation. All
CPU-only.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from asr_pipeline.config import DiarizationConfig, FusionConfig
from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.stages.fusion_diarization import (
    FusionDiarizationStage,
    _align_pass2_to_pass1,
    _overlap_len,
    _segments_by_speaker,
    fuse_labels,
)

SR = 16_000
CPU = torch.device("cpu")


def _seg_df(rows):
    """rows = list of (speaker, start, end)."""
    return pd.DataFrame(
        [{"start": s, "end": e, "duration": e - s, "speaker": spk}
         for spk, s, e in rows],
        columns=["start", "end", "duration", "speaker"],
    )


# ---------------------------------------------------------------------------
# _overlap_len / _segments_by_speaker
# ---------------------------------------------------------------------------


def test_overlap_len_basic():
    assert _overlap_len([(0.0, 10.0)], [(5.0, 8.0)]) == pytest.approx(3.0)
    assert _overlap_len([(0.0, 2.0)], [(5.0, 8.0)]) == 0.0
    # multi-interval union on both sides
    assert _overlap_len(
        [(0.0, 5.0), (10.0, 15.0)], [(3.0, 12.0)]
    ) == pytest.approx(2.0 + 2.0)


def test_segments_by_speaker_groups():
    df = _seg_df([("A", 0, 1), ("B", 1, 2), ("A", 3, 4)])
    by = _segments_by_speaker(df)
    assert by["A"] == [(0.0, 1.0), (3.0, 4.0)]
    assert by["B"] == [(1.0, 2.0)]


# ---------------------------------------------------------------------------
# _align_pass2_to_pass1 (2x2 max-overlap permutation)
# ---------------------------------------------------------------------------


def test_align_swapped_ids_to_pass1_labels():
    # pass-1 labels A,B; pass-2 ids X,Y where X aligns to B, Y to A (swapped).
    p1 = {"A": [(0.0, 10.0)], "B": [(10.0, 20.0)]}
    p2 = {"X": [(10.5, 19.5)], "Y": [(0.5, 9.5)]}
    out = _align_pass2_to_pass1(p1, p2, ["A", "B"])
    assert out == {"X": "B", "Y": "A"}


def test_align_straight_ids():
    p1 = {"A": [(0.0, 10.0)], "B": [(10.0, 20.0)]}
    p2 = {"X": [(0.5, 9.5)], "Y": [(10.5, 19.5)]}
    out = _align_pass2_to_pass1(p1, p2, ["A", "B"])
    assert out == {"X": "A", "Y": "B"}


def test_align_degenerate_single_pass2_speaker():
    # pass 2 found only one speaker → no permutation forced; the single id maps
    # to the pass-1 label it overlaps most (A here).
    p1 = {"A": [(0.0, 10.0)], "B": [(10.0, 20.0)]}
    p2 = {"X": [(1.0, 9.0)]}
    out = _align_pass2_to_pass1(p1, p2, ["A", "B"])
    assert out == {"X": "A"}


# ---------------------------------------------------------------------------
# fuse_labels — the merge rule (the load-bearing logic)
# ---------------------------------------------------------------------------


def test_fuse_overrides_confident_disagreement():
    """db15fc57 in miniature: pass 1 mislabeled a solo turn (29-31 s) as B; pass 2
    on enhanced audio confidently calls it A → fusion flips it, long turns kept."""
    pass1 = _seg_df([
        ("A", 0.0, 10.0),     # long A turn — kept
        ("B", 12.0, 22.0),    # long B turn — kept
        ("B", 29.0, 31.0),    # MISLABEL: actually A (the db15fc57 case)
    ])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 10.0),    # aligns to A
        ("SPEAKER_01", 12.0, 22.0),   # aligns to B
        ("SPEAKER_00", 29.0, 31.0),   # pass 2 calls the mislabel A, confidently
    ])
    new, n = fuse_labels(pass1, pass2, [], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B", "A"]   # the 29-31 s turn flipped to A
    assert n == 1


def test_fuse_keeps_label_below_confidence():
    """Pass 2 disagrees but only over part of the region (conf < min) → kept."""
    pass1 = _seg_df([
        ("A", 0.0, 10.0),
        ("B", 12.0, 22.0),
        ("B", 30.0, 32.0),    # 2 s region
    ])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 10.0),    # A
        ("SPEAKER_01", 12.0, 22.0),   # B
        # pass 2 covers only 1.0 s of the 2 s region as A → conf 0.5 < 0.75
        ("SPEAKER_00", 30.0, 31.0),
        ("SPEAKER_01", 31.0, 32.0),
    ])
    new, n = fuse_labels(pass1, pass2, [], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B", "B"]   # unchanged
    assert n == 0


def test_fuse_keeps_label_below_min_region():
    """A confident disagreement on a region shorter than min_region_s → kept."""
    pass1 = _seg_df([
        ("A", 0.0, 10.0),
        ("B", 12.0, 22.0),
        ("B", 30.0, 30.3),    # 0.3 s region, below min_region_s=0.5
    ])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_01", 12.0, 22.0),
        ("SPEAKER_00", 30.0, 30.3),   # confidently A but too short
    ])
    new, n = fuse_labels(pass1, pass2, [], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B", "B"]
    assert n == 0


def test_fuse_preserves_label_set_and_presence():
    """The label SET is preserved (alignment maps onto pass-1 labels, never fresh
    ids) — the invariant that keeps ctx.speakers / spk_to_label valid (plan §2.4).
    Row count / boundaries are untouched (presence stays with pass 1)."""
    pass1 = _seg_df([("A", 0.0, 5.0), ("B", 5.0, 10.0), ("B", 10.0, 12.0)])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 5.0),     # A
        ("SPEAKER_01", 5.0, 10.0),    # B
        ("SPEAKER_00", 10.0, 12.0),   # flips the 3rd to A
    ])
    new, n = fuse_labels(pass1, pass2, [], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert set(new) <= {"A", "B"}      # never an invented SPEAKER_0x
    assert len(new) == len(pass1)      # presence (row count) unchanged
    assert n == 1


def test_fuse_excludes_overlap_regions_from_solo_span():
    """A segment fully inside the overlap timeline has an empty solo span →
    never relabeled (presence/overlap identity is assembly's job, plan §5.2)."""
    pass1 = _seg_df([
        ("A", 0.0, 10.0),
        ("B", 12.0, 22.0),
        ("B", 30.0, 32.0),    # this whole region is an overlap → not a solo
    ])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_01", 12.0, 22.0),
        ("SPEAKER_00", 30.0, 32.0),   # pass 2 would flip it, but it's overlap
    ])
    overlap_regions = [(30.0, 32.0)]
    new, n = fuse_labels(pass1, pass2, overlap_regions, ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B", "B"]   # overlap segment kept
    assert n == 0


def test_fuse_partial_overlap_uses_remaining_solo_span():
    """A segment partly overlapped: only its solo remainder is judged. Region
    30-34 s, overlap 30-32 s → 2 s solo (32-34). Pass 2 calls 32-34 s A
    confidently → flip."""
    pass1 = _seg_df([
        ("A", 0.0, 10.0),
        ("B", 12.0, 22.0),
        ("B", 30.0, 34.0),
    ])
    pass2 = _seg_df([
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_01", 12.0, 22.0),
        ("SPEAKER_00", 32.0, 34.0),   # covers the solo remainder as A
    ])
    new, n = fuse_labels(pass1, pass2, [(30.0, 32.0)], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B", "A"]
    assert n == 1


def test_fuse_agreement_is_noop():
    """When pass 2 agrees everywhere, nothing changes."""
    pass1 = _seg_df([("A", 0.0, 10.0), ("B", 12.0, 22.0)])
    pass2 = _seg_df([("SPEAKER_00", 0.0, 10.0), ("SPEAKER_01", 12.0, 22.0)])
    new, n = fuse_labels(pass1, pass2, [], ["A", "B"],
                         confidence_min=0.75, min_region_s=0.5)
    assert new == ["A", "B"]
    assert n == 0


# ---------------------------------------------------------------------------
# Stage lifecycle + guards (pass-2 pipeline stubbed — no pyannote download)
# ---------------------------------------------------------------------------


class _Seg:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.duration = end - start


class _FakeDiar:
    """Bare-Annotation stand-in (pyannote 3.x shape)."""

    def __init__(self, tracks):
        self._tracks = tracks   # list of (_Seg, label)

    def itertracks(self, yield_label=False):
        for seg, label in self._tracks:
            yield (seg, "_", label) if yield_label else (seg, "_")


class _FakePipeline:
    def __init__(self, diar):
        self.diar = diar

    def __call__(self, inputs, num_speakers=None):
        return self.diar


def _cfg(enabled=True):
    return DiarizationConfig(
        model_id="pyannote/speaker-diarization-3.1",
        hf_token="tok",
        fusion=FusionConfig(enabled=enabled, embedding="ecapa2"),
    )


def _ctx(pass1_rows, overlap_regions=None, enhanced=True, speakers=None):
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = np.zeros(SR, dtype=np.float32)
    ctx.enhanced_full = np.zeros(SR, dtype=np.float32) if enhanced else None
    ctx.diarization = DiarizationResult(
        segments_df=_seg_df(pass1_rows),
        overlaps_df=pd.DataFrame(columns=["start", "end", "duration"]),
        total_duration_s=1.0,
    )
    ctx.overlap_regions = overlap_regions if overlap_regions is not None else []
    ctx.speakers = (
        speakers if speakers is not None
        else sorted(ctx.diarization.segments_df["speaker"].unique())
    )
    return ctx


def test_stage_enabled_tracks_fusion_flag():
    assert FusionDiarizationStage(_cfg(enabled=True)).enabled is True
    assert FusionDiarizationStage(_cfg(enabled=False)).enabled is False


def test_stage_run_before_load_raises():
    stage = FusionDiarizationStage(_cfg())
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx([("A", 0.0, 10.0), ("B", 10.0, 20.0)]))


def test_stage_enhanced_none_asserts():
    """SCOPE §4 defence-in-depth: fusion with no enhanced audio → crash, never a
    silent single-pass fall-back."""
    stage = FusionDiarizationStage(_cfg())
    stage._pipeline = _FakePipeline(_FakeDiar([]))
    ctx = _ctx([("A", 0.0, 10.0), ("B", 10.0, 20.0)], enhanced=False)
    with pytest.raises(AssertionError, match="enhanced_full is None"):
        stage.run(ctx)


def test_stage_single_speaker_is_noop():
    """One pass-1 speaker → keep labels, don't invent identity (SCOPE §3)."""
    stage = FusionDiarizationStage(_cfg())
    stage._pipeline = _FakePipeline(_FakeDiar([(_Seg(0.0, 1.0), "SPEAKER_00")]))
    ctx = _ctx([("A", 0.0, 10.0)])
    stage.run(ctx)
    assert ctx.diarization.segments_df["speaker"].tolist() == ["A"]


def test_stage_run_applies_merge_in_place():
    """End-to-end (stubbed pass 2): the stage overrides the speaker column in
    place and leaves boundaries / row count untouched."""
    stage = FusionDiarizationStage(_cfg())
    # pass 2 flips the 3rd (mislabeled) segment to A.
    stage._pipeline = _FakePipeline(_FakeDiar([
        (_Seg(0.0, 10.0), "SPEAKER_00"),
        (_Seg(12.0, 22.0), "SPEAKER_01"),
        (_Seg(29.0, 31.0), "SPEAKER_00"),
    ]))
    ctx = _ctx([("A", 0.0, 10.0), ("B", 12.0, 22.0), ("B", 29.0, 31.0)])
    before_bounds = ctx.diarization.segments_df[["start", "end"]].values.tolist()
    stage.run(ctx)
    seg = ctx.diarization.segments_df
    assert seg["speaker"].tolist() == ["A", "B", "A"]
    assert seg[["start", "end"]].values.tolist() == before_bounds  # presence kept


def test_stage_load_signature_tracks_fusion_embedding():
    """The pass-2 embedder is a model identity → in the signature; flipping it
    triggers a reload. Disabling fusion also changes the signature."""
    base = FusionDiarizationStage(_cfg(enabled=True))
    other_emb = FusionDiarizationStage(DiarizationConfig(
        model_id="pyannote/speaker-diarization-3.1", hf_token="tok",
        fusion=FusionConfig(enabled=True, embedding="eres2netv2"),
    ))
    off = FusionDiarizationStage(_cfg(enabled=False))
    assert base.load_signature() != other_emb.load_signature()
    assert base.load_signature() != off.load_signature()
