"""Unit tests for Stage 4 assembly helpers (pure functions, no ECAPA).

Covers interval subtraction (solo derivation), anchor-audio capping, RMS
matching/normalisation, fades, emit-region slicing, and both concat modes
(shortened / full_length) including the timestamp map they produce.
"""

import numpy as np
import pytest

from asr_pipeline.config import AssemblyConfig
from asr_pipeline.stages.assembly import (
    _apply_fade,
    _cap_anchor_audio,
    _concat_full_length,
    _concat_shortened,
    _match_overlap_rms_to_solo,
    _rms,
    _rms_normalise,
    _slice_emit,
    _subtract,
)


# ---------------------------------------------------------------------------
# _subtract (interval set difference)
# ---------------------------------------------------------------------------


def test_subtract_disjoint_untouched():
    assert _subtract([(0.0, 10.0)], [(20.0, 30.0)]) == [(0.0, 10.0)]


def test_subtract_full_overlap_removes():
    assert _subtract([(0.0, 10.0)], [(0.0, 10.0)]) == []


def test_subtract_left_trim():
    assert _subtract([(0.0, 10.0)], [(0.0, 4.0)]) == [(4.0, 10.0)]


def test_subtract_right_trim():
    assert _subtract([(0.0, 10.0)], [(6.0, 10.0)]) == [(0.0, 6.0)]


def test_subtract_split():
    assert _subtract([(0.0, 10.0)], [(4.0, 6.0)]) == [(0.0, 4.0), (6.0, 10.0)]


def test_subtract_multiple_blockers():
    out = _subtract([(0.0, 10.0)], [(1.0, 2.0), (3.0, 4.0)])
    assert out == [(0.0, 1.0), (2.0, 3.0), (4.0, 10.0)]


def test_subtract_drops_tiny_residuals():
    # Residual 0.0005 s < the 1e-3 floor -> dropped.
    assert _subtract([(0.0, 1.0)], [(0.0005, 1.0)]) == []


# ---------------------------------------------------------------------------
# _cap_anchor_audio
# ---------------------------------------------------------------------------


def test_cap_none_is_noop():
    audio = np.arange(1000, dtype=np.float32)
    assert np.array_equal(_cap_anchor_audio(audio, 100, None), audio)


def test_cap_short_audio_unchanged():
    audio = np.arange(150, dtype=np.float32)
    assert np.array_equal(_cap_anchor_audio(audio, 100, 2.0), audio)


def test_cap_long_audio_uniform_stride():
    sr = 100
    audio = np.arange(1000, dtype=np.float32)   # 10 s
    out = _cap_anchor_audio(audio, sr, max_duration_s=2.0)
    assert len(out) == 200
    # Chunks are evenly spread: first second from the start, last from the end.
    assert np.array_equal(out[:100], audio[0:100])
    assert np.array_equal(out[100:], audio[900:1000])


def test_cap_when_chunks_cover_everything_truncates():
    sr = 100
    audio = np.arange(250, dtype=np.float32)    # 2.5 s, cap 2.0 s
    out = _cap_anchor_audio(audio, sr, max_duration_s=2.0)
    assert np.array_equal(out, audio[:200])


# ---------------------------------------------------------------------------
# RMS helpers
# ---------------------------------------------------------------------------


def test_rms_empty_is_zero():
    assert _rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_rms_of_ones():
    assert _rms(np.ones(10, dtype=np.float32)) == pytest.approx(1.0)


def test_rms_normalise_explicit_target():
    pieces = [2.0 * np.ones(10, dtype=np.float32)]
    out = _rms_normalise(pieces, target_rms=1.0)
    assert _rms(out[0]) == pytest.approx(1.0)


def test_rms_normalise_median_target():
    pieces = [
        1.0 * np.ones(10, dtype=np.float32),
        2.0 * np.ones(10, dtype=np.float32),
        3.0 * np.ones(10, dtype=np.float32),
    ]
    out = _rms_normalise(pieces, target_rms=None)
    for p in out:
        assert _rms(p) == pytest.approx(2.0)   # median of {1, 2, 3}


def test_rms_normalise_leaves_silent_and_empty_pieces():
    silent = np.zeros(10, dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)
    out = _rms_normalise([silent, empty, np.ones(10, dtype=np.float32)], 1.0)
    assert np.array_equal(out[0], silent)
    assert out[1].size == 0


def test_match_overlap_rms_to_solo_scales_only_overlaps():
    events = [
        {"kind": "solo", "audio": np.ones(10, dtype=np.float32)},
        {"kind": "solo", "audio": np.ones(10, dtype=np.float32)},
        {"kind": "overlap", "audio": 2.0 * np.ones(10, dtype=np.float32)},
    ]
    _match_overlap_rms_to_solo(events)
    assert _rms(events[2]["audio"]) == pytest.approx(1.0)   # scaled to solo median
    assert _rms(events[0]["audio"]) == pytest.approx(1.0)   # solos untouched


def test_match_overlap_rms_noop_without_solos():
    overlap = 2.0 * np.ones(10, dtype=np.float32)
    events = [{"kind": "overlap", "audio": overlap}]
    _match_overlap_rms_to_solo(events)
    assert np.array_equal(events[0]["audio"], overlap)


# ---------------------------------------------------------------------------
# _apply_fade
# ---------------------------------------------------------------------------


def test_fade_zero_zero_is_identity():
    audio = np.ones(20, dtype=np.float32)
    assert np.array_equal(_apply_fade(audio, 0, 0), audio)


def test_fade_attenuates_edges_not_middle():
    audio = np.ones(100, dtype=np.float32)
    out = _apply_fade(audio, 10, 10)
    assert out[0] == pytest.approx(0.0)          # half-Hann starts at 0
    assert out[-1] == pytest.approx(0.0)
    assert out[50] == pytest.approx(1.0)         # interior untouched
    assert np.all(audio == 1.0)                  # input not mutated


def test_fade_caps_on_short_pieces():
    audio = np.ones(4, dtype=np.float32)
    out = _apply_fade(audio, 10, 10)             # capped to len//2 = 2 each
    assert out.shape == (4,)
    assert out[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _slice_emit
# ---------------------------------------------------------------------------


def test_slice_emit_basic_offsets():
    gated = np.arange(20, dtype=np.float32)
    out = _slice_emit(gated, pad_start_s=1.0, emit_start_s=1.5, emit_end_s=2.0,
                      sample_rate=10)
    assert np.array_equal(out, gated[5:10])


def test_slice_emit_clamps_negative_offset():
    gated = np.arange(20, dtype=np.float32)
    out = _slice_emit(gated, 1.0, 0.5, 2.0, 10)
    # offset clamped to 0; length = 1.5 s -> 15 samples
    assert np.array_equal(out, gated[0:15])


def test_slice_emit_truncates_past_end():
    gated = np.arange(10, dtype=np.float32)
    out = _slice_emit(gated, 0.0, 0.5, 5.0, 10)
    assert np.array_equal(out, gated[5:10])


def test_slice_emit_empty_when_no_length():
    gated = np.arange(10, dtype=np.float32)
    out = _slice_emit(gated, 0.0, 2.0, 2.0, 10)
    assert out.size == 0


# ---------------------------------------------------------------------------
# Concatenation modes
# ---------------------------------------------------------------------------


def _events_two_pieces():
    return [
        {"orig_start": 1.0, "orig_end": 2.0,
         "audio": np.ones(10, dtype=np.float32), "kind": "solo"},
        {"orig_start": 3.0, "orig_end": 4.0,
         "audio": 2.0 * np.ones(10, dtype=np.float32), "kind": "overlap"},
    ]


def test_concat_shortened_layout_and_map():
    cfg = AssemblyConfig(silence_separator_s=0.5)
    stream, tmap = _concat_shortened(_events_two_pieces(), cfg, sr=10)
    assert len(stream) == 10 + 5 + 10            # piece + 0.5 s gap + piece
    assert np.all(stream[10:15] == 0.0)          # the gap
    assert tmap[0].concat_start == pytest.approx(0.0)
    assert tmap[0].concat_end == pytest.approx(1.0)
    assert tmap[1].concat_start == pytest.approx(1.5)
    assert tmap[1].concat_end == pytest.approx(2.5)
    assert (tmap[1].orig_start, tmap[1].orig_end) == (3.0, 4.0)
    assert tmap[0].kind == "solo" and tmap[1].kind == "overlap"


def test_concat_shortened_empty_gives_silence():
    cfg = AssemblyConfig()
    stream, tmap = _concat_shortened([], cfg, sr=10)
    assert len(stream) == 10 and np.all(stream == 0.0)
    assert tmap == []


def test_concat_full_length_places_at_orig_time():
    stream, tmap = _concat_full_length(_events_two_pieces(), total_dur_s=5.0, sr=10)
    assert len(stream) == 50
    assert np.all(stream[10:20] == 1.0)          # first piece at orig 1.0 s
    assert np.all(stream[30:40] == 2.0)          # second piece at orig 3.0 s
    assert np.all(stream[0:10] == 0.0) and np.all(stream[20:30] == 0.0)
    # In full_length mode concat times coincide with original times.
    assert tmap[0].concat_start == pytest.approx(tmap[0].orig_start)


def test_concat_full_length_skips_pieces_past_end():
    events = [{"orig_start": 5.0, "orig_end": 6.0,
               "audio": np.ones(10, dtype=np.float32), "kind": "solo"}]
    stream, tmap = _concat_full_length(events, total_dur_s=3.0, sr=10)
    assert len(stream) == 30 and np.all(stream == 0.0)
    assert tmap == []


def test_concat_full_length_clips_overrunning_piece():
    events = [{"orig_start": 2.5, "orig_end": 3.5,
               "audio": np.ones(10, dtype=np.float32), "kind": "solo"}]
    stream, tmap = _concat_full_length(events, total_dur_s=3.0, sr=10)
    assert np.all(stream[25:30] == 1.0)
    assert tmap[0].concat_end == pytest.approx(3.0)
