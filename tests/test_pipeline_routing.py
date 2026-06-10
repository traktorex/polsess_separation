"""Unit tests for Stage 2 routing helpers (overlap selection + merging)."""

import pandas as pd

from asr_pipeline.stages.routing import _merge_close, _select_overlap_regions


def _ovl_df(rows):
    """Build an overlaps_df like DiarizationStage produces."""
    if not rows:
        return pd.DataFrame(columns=["start", "end", "duration"])
    return pd.DataFrame(
        [{"start": s, "end": e, "duration": e - s} for s, e in rows]
    )


# ---------------------------------------------------------------------------
# _merge_close
# ---------------------------------------------------------------------------


def test_merge_close_empty():
    assert _merge_close([], 0.5) == []


def test_merge_close_single():
    assert _merge_close([(1.0, 2.0)], 0.5) == [(1.0, 2.0)]


def test_merge_close_merges_within_gap():
    # Gap 0.3 < merge_gap 0.5 -> one region.
    assert _merge_close([(0.0, 1.0), (1.3, 2.0)], 0.5) == [(0.0, 2.0)]


def test_merge_close_keeps_beyond_gap():
    # Gap 0.6 >= merge_gap 0.5 -> stays two regions.
    assert _merge_close([(0.0, 1.0), (1.6, 2.0)], 0.5) == [(0.0, 1.0), (1.6, 2.0)]


def test_merge_close_gap_exactly_threshold_not_merged():
    # The condition is strict `<`, so gap == merge_gap does NOT merge.
    assert _merge_close([(0.0, 1.0), (1.5, 2.0)], 0.5) == [(0.0, 1.0), (1.5, 2.0)]


def test_merge_close_sorts_input():
    assert _merge_close([(3.0, 4.0), (0.0, 1.0)], 0.5) == [(0.0, 1.0), (3.0, 4.0)]


def test_merge_close_overlapping_takes_max_end():
    # Second interval nested inside the first: end stays at the max.
    assert _merge_close([(0.0, 2.0), (1.0, 1.5)], 0.5) == [(0.0, 2.0)]


def test_merge_close_chain_merge():
    # Three regions, each within gap of the previous -> single region.
    segs = [(0.0, 1.0), (1.2, 2.0), (2.3, 3.0)]
    assert _merge_close(segs, 0.5) == [(0.0, 3.0)]


# ---------------------------------------------------------------------------
# _select_overlap_regions
# ---------------------------------------------------------------------------


def test_select_filters_short_overlaps():
    df = _ovl_df([(0.0, 0.1), (1.0, 1.5)])
    # 0.1 s overlap < min_overlap_dur 0.2 -> dropped.
    assert _select_overlap_regions(df, min_overlap_dur=0.2, merge_gap=0.5) == [
        (1.0, 1.5)
    ]


def test_select_min_duration_is_inclusive():
    df = _ovl_df([(0.0, 0.2)])
    # duration == min_overlap_dur is kept (>= comparison).
    assert _select_overlap_regions(df, min_overlap_dur=0.2, merge_gap=0.5) == [
        (0.0, 0.2)
    ]


def test_select_then_merges():
    df = _ovl_df([(0.0, 1.0), (1.2, 2.0), (5.0, 6.0)])
    out = _select_overlap_regions(df, min_overlap_dur=0.2, merge_gap=0.5)
    assert out == [(0.0, 2.0), (5.0, 6.0)]


def test_select_empty_df():
    df = _ovl_df([])
    assert _select_overlap_regions(df, min_overlap_dur=0.2, merge_gap=0.5) == []


def test_routing_stage_handles_empty_diarization():
    """Regression: silent input → pyannote returns zero segments. The
    DataFrame the diarization stage builds for that case (explicit columns,
    no rows) must flow through routing without a KeyError."""
    import pandas as pd

    from asr_pipeline.config import RoutingConfig
    from asr_pipeline.context import DiarizationResult, PipelineContext
    from asr_pipeline.stages.routing import RoutingStage

    seg_df = pd.DataFrame([], columns=["start", "end", "duration", "speaker"])
    ovl_df = pd.DataFrame(columns=["start", "end", "duration"])
    ctx = PipelineContext()
    ctx.diarization = DiarizationResult(
        segments_df=seg_df, overlaps_df=ovl_df, total_duration_s=0.0
    )
    RoutingStage(RoutingConfig()).run(ctx)
    assert ctx.overlap_regions == []
    assert ctx.speakers == []


def _ctx_with_diarization(speaker_rows, overlap_rows):
    """Build a PipelineContext with a populated DiarizationResult."""
    from asr_pipeline.context import DiarizationResult, PipelineContext

    seg_df = pd.DataFrame(
        [{"start": s, "end": e, "duration": e - s, "speaker": spk}
         for s, e, spk in speaker_rows]
    )
    ovl_df = _ovl_df(overlap_rows)
    ctx = PipelineContext()
    ctx.diarization = DiarizationResult(
        segments_df=seg_df, overlaps_df=ovl_df, total_duration_s=0.0
    )
    return ctx


def test_routing_stage_populates_regions_and_speakers():
    from asr_pipeline.config import RoutingConfig
    from asr_pipeline.stages.routing import RoutingStage

    # Rows deliberately out of order + a duplicated speaker so the sorted-unique
    # speaker derivation and the overlap selection/merge both get exercised.
    speaker_rows = [
        (5.0, 6.0, "SPEAKER_01"),
        (0.0, 2.0, "SPEAKER_00"),
        (2.5, 3.0, "SPEAKER_00"),
    ]
    # Two overlaps within merge_gap (default 0.5) → merged; one short → dropped.
    overlap_rows = [(1.0, 1.5), (1.7, 2.0), (3.0, 3.05)]
    ctx = _ctx_with_diarization(speaker_rows, overlap_rows)
    RoutingStage(RoutingConfig()).run(ctx)
    assert ctx.overlap_regions == [(1.0, 2.0)]
    assert ctx.speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_routing_spill_writes_overlap_regions_json(tmp_path):
    import json

    from asr_pipeline.config import RoutingConfig
    from asr_pipeline.stages.routing import RoutingStage

    ctx = _ctx_with_diarization(
        [(0.0, 2.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")],
        [(1.0, 1.5), (1.7, 2.0)],
    )
    stage = RoutingStage(RoutingConfig())
    stage.run(ctx)
    stage.spill(ctx, tmp_path)

    payload = json.loads((tmp_path / "overlap_regions.json").read_text())
    # The spill's superset schema: speakers + {start, end, duration}.
    assert payload["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert payload["overlap_regions"] == [
        {"start": 1.0, "end": 2.0, "duration": 1.0}     # duration = e - s recompute
    ]


def test_routing_spill_noops_when_unrun(tmp_path):
    from asr_pipeline.config import RoutingConfig
    from asr_pipeline.context import PipelineContext
    from asr_pipeline.stages.routing import RoutingStage

    RoutingStage(RoutingConfig()).spill(PipelineContext(), tmp_path)
    assert list(tmp_path.iterdir()) == []
