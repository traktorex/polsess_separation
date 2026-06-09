"""Tests for Stage 4's run-level attribution logic (asr_pipeline/stages/assembly.py).

Complements `test_pipeline_assembly.py` (pure numeric helpers) with the
functions that decide *whose audio goes where*: solo-interval derivation,
overlap→speaker assignment (with a stub ECAPA), the no-separation
mixture fill, and event-list construction.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.stages.assembly import (
    _assign_overlaps,
    _build_events,
    _derive_solo_intervals,
    _mixture_fill_overlaps,
)

SR = 16_000
DEVICE = torch.device("cpu")


class _StubEcapa:
    """Deterministic stand-in for the ECAPA encoder.

    Maps audio to a 2-D embedding by the sign of its mean: positive-mean
    audio → [1, 0], non-positive → [0, 1]. Lets tests steer the cosine
    pairing without a real model.
    """

    def encode_batch(self, audio: torch.Tensor) -> torch.Tensor:
        v = (
            torch.tensor([1.0, 0.0])
            if float(audio.mean()) > 0
            else torch.tensor([0.0, 1.0])
        )
        return v.view(1, 1, 2)


def _anchors():
    return {
        "SPK_A": torch.tensor([1.0, 0.0]),
        "SPK_B": torch.tensor([0.0, 1.0]),
    }


def _ovl(s1, s2, idx=0, pad_start=0.0, emit_start=0.0, emit_end=None):
    s1 = np.asarray(s1, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)
    if emit_end is None:
        emit_end = len(s1) / SR
    return {
        "idx": idx,
        "pad_start": pad_start,
        "emit_start": emit_start,
        "emit_end": emit_end,
        "s1_gated": s1,
        "s2_gated": s2,
    }


# ---------------------------------------------------------------------------
# _assign_overlaps
# ---------------------------------------------------------------------------


def test_assign_straight_pairing():
    # s1 positive-mean → matches SPK_A's anchor; s2 negative → SPK_B.
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5))
    out = _assign_overlaps(
        [ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert len(out) == 1
    assert out[0]["pairing"] == "straight"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s2_gated"])


def test_assign_swapped_pairing():
    ovl = _ovl(np.full(SR, -0.5), np.full(SR, 0.5))
    out = _assign_overlaps(
        [ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert out[0]["pairing"] == "swapped"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s2_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s1_gated"])


def test_assign_weak_anchor_falls_back_to_fixed():
    anchors = {"SPK_A": None, "SPK_B": torch.tensor([0.0, 1.0])}
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5))
    out = _assign_overlaps(
        [ovl], anchors, ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert out[0]["pairing"] == "arbitrary (weak anchor)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])


def test_assign_too_short_keeps_region_with_fixed_assignment():
    # Regression: streams under 0.1 s used to be silently dropped, losing
    # the region's audio for both speakers. Now: fixed assignment + keep.
    n = SR // 20  # 0.05 s
    ovl = _ovl(np.full(n, 0.5), np.full(n, -0.5))
    out = _assign_overlaps(
        [ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert len(out) == 1
    assert out[0]["pairing"] == "arbitrary (too short)"
    assert len(out[0]["emit_pieces"]["SPK_A"]) == n
    assert len(out[0]["emit_pieces"]["SPK_B"]) == n


def test_assign_missing_gated_raises_clear_error():
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5))
    del ovl["s1_gated"]
    with pytest.raises(RuntimeError, match="post_separation_processing"):
        _assign_overlaps(
            [ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
        )


# ---------------------------------------------------------------------------
# _mixture_fill_overlaps
# ---------------------------------------------------------------------------


def test_mixture_fill_attributes_same_clip_to_all_speakers():
    audio = np.arange(10 * SR, dtype=np.float32)
    out = _mixture_fill_overlaps([(1.0, 2.0)], ["SPK_A", "SPK_B"], audio, SR)
    assert len(out) == 1
    assert out[0]["pairing"] == "no_separation"
    np.testing.assert_array_equal(
        out[0]["emit_pieces"]["SPK_A"], audio[SR : 2 * SR]
    )
    np.testing.assert_array_equal(
        out[0]["emit_pieces"]["SPK_A"], out[0]["emit_pieces"]["SPK_B"]
    )


def test_mixture_fill_clamps_region_overshooting_audio_end():
    # Regression: pyannote turns can overshoot the audio end by a fraction
    # of a second; such a region used to be dropped entirely.
    audio = np.ones(10 * SR, dtype=np.float32)
    out = _mixture_fill_overlaps([(9.5, 10.4)], ["SPK_A"], audio, SR)
    assert len(out) == 1
    assert len(out[0]["emit_pieces"]["SPK_A"]) == SR // 2  # 9.5 → 10.0 only


def test_mixture_fill_skips_region_entirely_past_end():
    audio = np.ones(10 * SR, dtype=np.float32)
    out = _mixture_fill_overlaps([(10.2, 10.5)], ["SPK_A"], audio, SR)
    assert out == []


# ---------------------------------------------------------------------------
# _derive_solo_intervals + _build_events
# ---------------------------------------------------------------------------


def _ctx_with_segments(segments, overlap_separated=None, overlap_regions=None):
    seg_df = pd.DataFrame(
        [
            {"start": s, "end": e, "duration": e - s, "speaker": spk}
            for spk, s, e in segments
        ],
        columns=["start", "end", "duration", "speaker"],
    )
    ovl_df = pd.DataFrame(columns=["start", "end", "duration"])
    ctx = PipelineContext(sample_rate=SR)
    ctx.diarization = DiarizationResult(
        segments_df=seg_df, overlaps_df=ovl_df, total_duration_s=10.0
    )
    ctx.overlap_separated = overlap_separated or []
    ctx.overlap_regions = overlap_regions
    return ctx


def test_solo_intervals_subtract_emit_regions():
    ctx = _ctx_with_segments(
        [("SPK_A", 0.0, 5.0)],
        overlap_separated=[{"emit_start": 2.0, "emit_end": 3.0}],
    )
    solos = _derive_solo_intervals(ctx, ["SPK_A"])
    assert solos["SPK_A"] == [(0.0, 2.0), (3.0, 5.0)]


def test_solo_intervals_fall_back_to_overlap_regions_without_3b():
    ctx = _ctx_with_segments(
        [("SPK_A", 0.0, 4.0)],
        overlap_separated=[],
        overlap_regions=[(1.0, 2.0)],
    )
    solos = _derive_solo_intervals(ctx, ["SPK_A"])
    assert solos["SPK_A"] == [(0.0, 1.0), (2.0, 4.0)]


def test_build_events_combines_and_sorts_solos_and_overlaps():
    enhanced = np.arange(10 * SR, dtype=np.float32)
    solo_intervals = {"SPK_A": [(0.0, 1.0), (4.0, 5.0)]}
    assignments = [
        {
            "orig_start": 2.0,
            "orig_end": 3.0,
            "pairing": "straight",
            "emit_pieces": {"SPK_A": np.ones(SR, dtype=np.float32)},
        }
    ]
    events = _build_events(["SPK_A"], solo_intervals, assignments, enhanced, SR)
    kinds = [(e["kind"], e["orig_start"]) for e in events["SPK_A"]]
    assert kinds == [("solo", 0.0), ("overlap", 2.0), ("solo", 4.0)]
    # Solo audio comes from the enhanced full recording.
    np.testing.assert_array_equal(events["SPK_A"][0]["audio"], enhanced[:SR])


def test_build_events_drops_empty_overlap_pieces():
    enhanced = np.zeros(SR, dtype=np.float32)
    assignments = [
        {
            "orig_start": 0.1,
            "orig_end": 0.2,
            "pairing": "straight",
            "emit_pieces": {"SPK_A": np.zeros(0, dtype=np.float32)},
        }
    ]
    events = _build_events(["SPK_A"], {"SPK_A": []}, assignments, enhanced, SR)
    assert events["SPK_A"] == []
