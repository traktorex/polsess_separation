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

from asr_pipeline.config import AssemblyConfig
from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.stages.assembly import (
    AssemblyStage,
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


class _TableEcapa:
    """ECAPA stub returning a preset embedding keyed by the audio's first
    sample (rounded). Lets a test pick exact, non-orthogonal embeddings so
    the cosine pairing exercises *summed* similarity, not just sign. NaN
    entries are allowed, to drive the non-finite-cosine guard.
    """

    def __init__(self, table: dict[float, list[float]]) -> None:
        self.table = table

    def encode_batch(self, audio: torch.Tensor) -> torch.Tensor:
        key = round(float(audio.reshape(-1)[0]), 4)
        v = torch.tensor(self.table[key], dtype=torch.float32)
        return v.view(1, 1, -1)


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
    # SPK_A has no anchor → fixed (stream-order) fallback, never ECAPA.
    # Load-bearing content choice: s1 matches SPK_B's anchor and s2 would
    # match the (missing) SPK_A anchor, so a content-matching bug would assign
    # SPK_A=s2. The fixed fallback must keep stream order (SPK_A=s1); only this
    # content-vs-order divergence can tell the right branch from a coincidence.
    anchors = {"SPK_A": None, "SPK_B": torch.tensor([0.0, 1.0])}
    ovl = _ovl(np.full(SR, -0.5), np.full(SR, 0.5))   # s1 negative, s2 positive
    out = _assign_overlaps(
        [ovl], anchors, ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert out[0]["pairing"] == "arbitrary (weak anchor)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s2_gated"])


def test_assign_pairing_uses_summed_cosine_not_single_stream():
    # s1's embedding [0.6, 0.8] is individually closer to SPK_B's anchor
    # ([0,1]) than to SPK_A's ([1,0]); a matcher that decided on one stream
    # alone (or argmax-per-stream) would pick "swapped". The correct *summed*
    # cosine — s2's embedding [0,1] matches SPK_B decisively — makes "straight"
    # win 1.6 vs 0.8, keeping s1→SPK_A.
    anchors = {"SPK_A": torch.tensor([1.0, 0.0]), "SPK_B": torch.tensor([0.0, 1.0])}
    ecapa = _TableEcapa({0.6: [0.6, 0.8], 0.1: [0.0, 1.0]})
    ovl = _ovl(np.full(SR, 0.6), np.full(SR, 0.1))
    out = _assign_overlaps([ovl], anchors, ["SPK_A", "SPK_B"], ecapa, DEVICE, SR)
    assert out[0]["pairing"] == "straight"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s2_gated"])


def test_assign_pairing_exact_tie_resolves_to_straight():
    # Identical embeddings make straight == swapped exactly; the `>=` tie-break
    # must resolve to "straight" (keep stream order) rather than swapping.
    anchors = {"SPK_A": torch.tensor([1.0, 0.0]), "SPK_B": torch.tensor([0.0, 1.0])}
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, 0.5))   # both → [1, 0] via _StubEcapa
    out = _assign_overlaps(
        [ovl], anchors, ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR
    )
    assert out[0]["pairing"] == "straight"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])


def test_assign_non_finite_cosine_falls_back_to_fixed():
    # A NaN ECAPA embedding makes both cosines NaN; `straight >= swapped` is
    # then False and would silently pick "swapped". The isfinite guard must
    # instead drop to the fixed fallback and label it honestly.
    anchors = {"SPK_A": torch.tensor([1.0, 0.0]), "SPK_B": torch.tensor([0.0, 1.0])}
    ecapa = _TableEcapa({0.5: [float("nan"), float("nan")], -0.5: [0.0, 1.0]})
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5))
    out = _assign_overlaps([ovl], anchors, ["SPK_A", "SPK_B"], ecapa, DEVICE, SR)
    assert out[0]["pairing"] == "arbitrary (non-finite cosine)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s2_gated"])


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


def test_solo_intervals_no_duplication_from_overlapping_segments():
    # Two overlapping same-speaker segments are coalesced before subtraction,
    # so the shared 3-5 s region is not emitted twice (which would duplicate
    # that speech in the assembled stream). Without coalescing this returns
    # [(0, 5), (3, 8)] and 3-5 s is double-counted.
    ctx = _ctx_with_segments(
        [("SPK_A", 0.0, 5.0), ("SPK_A", 3.0, 8.0)],
        overlap_separated=[],
        overlap_regions=[],
    )
    solos = _derive_solo_intervals(ctx, ["SPK_A"])
    assert solos["SPK_A"] == [(0.0, 8.0)]


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


# ---------------------------------------------------------------------------
# AssemblyStage.run (stub ECAPA injected — no load(), no SpeechBrain download)
# ---------------------------------------------------------------------------


def _make_stage(ecapa=None, **config_kwargs) -> AssemblyStage:
    stage = AssemblyStage(AssemblyConfig(**config_kwargs))
    stage._ecapa = ecapa or _StubEcapa()
    stage._device = DEVICE
    return stage


def _diarization(segments, total_duration_s):
    seg_df = pd.DataFrame(
        [
            {"start": s, "end": e, "duration": e - s, "speaker": spk}
            for spk, s, e in segments
        ],
        columns=["start", "end", "duration", "speaker"],
    )
    return DiarizationResult(
        segments_df=seg_df,
        overlaps_df=pd.DataFrame(columns=["start", "end", "duration"]),
        total_duration_s=total_duration_s,
    )


def test_run_before_load_raises():
    stage = AssemblyStage(AssemblyConfig())   # no ECAPA / device injected
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = np.zeros(SR, dtype=np.float32)
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(ctx)


def test_run_requires_upstream_context():
    stage = _make_stage()
    ctx = PipelineContext(sample_rate=SR)   # no audio / diarization / regions
    with pytest.raises(RuntimeError, match="requires"):
        stage.run(ctx)


def test_run_no_speakers_yields_empty_map():
    stage = _make_stage()
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = np.zeros(SR, dtype=np.float32)
    ctx.diarization = _diarization([], total_duration_s=1.0)
    ctx.overlap_regions = []
    ctx.speakers = []
    stage.run(ctx)
    assert ctx.timestamp_map.per_speaker == {}
    assert ctx.assembled == {}


def test_run_uses_raw_mixture_when_enhancement_disabled():
    # enhanced_full=None → assembly slices solos from the raw mixture
    # (ctx.audio). With one speaker, no overlaps, and post-processing off, the
    # full_length output is exactly the raw-mixture slice placed at its time.
    audio = np.linspace(0.1, 0.9, 2 * SR).astype(np.float32)   # positive ramp
    stage = _make_stage(
        output_mode="full_length",
        min_solo_for_anchor_s=1.0,
        crossfade_ms=0.0,
        edge_fade_ms=0.0,
        overlap_rms_match_solo=False,
    )
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = audio
    ctx.enhanced_full = None
    ctx.diarization = _diarization([("SPK_A", 0.0, 2.0)], total_duration_s=2.0)
    ctx.overlap_regions = []
    ctx.speakers = ["SPK_A"]
    ctx.overlap_separated = []
    stage.run(ctx)
    np.testing.assert_array_equal(ctx.assembled["SPK_A"], audio)


def test_assembly_run_end_to_end():
    # Two speakers with distinct solo regions + one separated overlap.
    # enhanced_full is positive in A's solo and negative in B's solo, so the
    # stub ECAPA derives anchors A=[1,0], B=[0,1]; the overlap's positive s1
    # then matches A and negative s2 matches B (straight pairing).
    enhanced = np.zeros(5 * SR, dtype=np.float32)
    enhanced[0 : 2 * SR] = 0.5          # SPK_A solo 0–2 s (positive mean)
    enhanced[3 * SR : 5 * SR] = -0.5    # SPK_B solo 3–5 s (negative mean)
    ovl = _ovl(
        np.full(SR, 0.5), np.full(SR, -0.5),
        idx=0, pad_start=2.0, emit_start=2.0, emit_end=3.0,
    )
    stage = _make_stage(
        min_solo_for_anchor_s=1.0,      # 2 s solos are not "weak"
        crossfade_ms=0.0,
        edge_fade_ms=0.0,
        overlap_rms_match_solo=False,
    )
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = enhanced.copy()
    ctx.enhanced_full = enhanced
    ctx.diarization = _diarization(
        [("SPK_A", 0.0, 2.5), ("SPK_B", 2.5, 5.0)], total_duration_s=5.0
    )
    ctx.overlap_regions = [(2.0, 3.0)]
    ctx.speakers = ["SPK_A", "SPK_B"]
    ctx.overlap_separated = [ovl]

    stage.run(ctx)

    assert set(ctx.assembled) == {"SPK_A", "SPK_B"}
    assert ctx.spk_to_label == {"SPK_A": "A", "SPK_B": "B"}
    assert set(ctx.timestamp_map.per_speaker) == {"SPK_A", "SPK_B"}
    assert ctx.weak_anchor is False
    # SPK_A: solo (0–2) then overlap (2–3); SPK_B: overlap (2–3) then solo (3–5).
    assert [e.kind for e in ctx.timestamp_map.per_speaker["SPK_A"]] == [
        "solo", "overlap",
    ]
    assert [e.kind for e in ctx.timestamp_map.per_speaker["SPK_B"]] == [
        "overlap", "solo",
    ]


def test_run_warns_on_more_than_two_speakers(capsys):
    # >2 speakers: overlap audio is only ever assigned to speakers[:2]; the
    # third must not be dropped in silence without a trace.
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5),
               idx=0, pad_start=0.0, emit_start=0.0, emit_end=1.0)
    out = _assign_overlaps(
        [ovl], _anchors(), ["SPK_A", "SPK_B", "SPK_C"], _StubEcapa(), DEVICE, SR
    )
    # The two handled speakers are still assigned; SPK_C gets nothing
    # (documented limitation), but a warning naming it must be emitted.
    assert set(out[0]["emit_pieces"]) == {"SPK_A", "SPK_B"}
    logged = capsys.readouterr().out
    assert "WARNING" in logged and "SPK_C" in logged
