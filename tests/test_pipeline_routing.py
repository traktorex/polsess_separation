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
