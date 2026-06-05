"""Unit tests for Stage 3b separation helpers (pure functions, no model).

Covers the VAD masking (soft threshold + dilation), context-window padding,
seam selection (zero-crossing + snap-to-silence), volume normalisation, and
the PIT chunk-alignment heuristic.
"""

import numpy as np
import pytest

from asr_pipeline.stages.separation import (
    _dilate_mask,
    _extend_end_to_silence,
    _extend_start_to_silence,
    _pick_seam_zero_crossing,
    _pit_swap_if_needed,
    _signal_aware_pad,
    _soft_threshold_mask,
    _volume_normalise,
    _window_expand_to_chunk,
    _window_fixed_pad,
    _window_none,
)


# ---------------------------------------------------------------------------
# _soft_threshold_mask (Schmitt-trigger VAD frame mask)
# ---------------------------------------------------------------------------


def test_soft_threshold_strict_when_lower_disabled():
    probs = np.array([0.9, 0.3, 0.6])
    mask = _soft_threshold_mask(probs, upper=0.5, lower=0.0)
    assert np.array_equal(mask, [1.0, 0.0, 1.0])


def test_soft_threshold_strict_when_lower_geq_upper():
    probs = np.array([0.9, 0.3, 0.6])
    mask = _soft_threshold_mask(probs, upper=0.5, lower=0.5)
    assert np.array_equal(mask, [1.0, 0.0, 1.0])


def test_soft_threshold_forward_propagation():
    # Strong frame followed by a chain of weak frames -> all speech;
    # the chain breaks at the sub-lower frame.
    probs = np.array([0.9, 0.3, 0.3, 0.1])
    mask = _soft_threshold_mask(probs, upper=0.5, lower=0.2)
    assert np.array_equal(mask, [1.0, 1.0, 1.0, 0.0])


def test_soft_threshold_backward_propagation():
    # Weak chain *preceding* a strong frame is also captured.
    probs = np.array([0.3, 0.3, 0.9])
    mask = _soft_threshold_mask(probs, upper=0.5, lower=0.2)
    assert np.array_equal(mask, [1.0, 1.0, 1.0])


def test_soft_threshold_isolated_weak_frame_excluded():
    # Weak frame not connected to any strong frame stays silence.
    probs = np.array([0.3, 0.1, 0.9])
    mask = _soft_threshold_mask(probs, upper=0.5, lower=0.2)
    assert np.array_equal(mask, [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# _dilate_mask (attack / release dilation)
# ---------------------------------------------------------------------------


def test_dilate_noop_when_zero():
    mask = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = _dilate_mask(mask, attack_frames=0, release_frames=0)
    assert np.array_equal(out, mask)


def test_dilate_extends_onset_and_offset():
    mask = np.array([0, 0, 1, 1, 0, 0], dtype=np.float32)
    out = _dilate_mask(mask, attack_frames=1, release_frames=1)
    assert np.array_equal(out, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])


def test_dilate_clamps_at_edges():
    mask = np.array([1, 0, 0, 0, 1], dtype=np.float32)
    out = _dilate_mask(mask, attack_frames=3, release_frames=3)
    # Run at idx 0 has no room to the left; run at idx 4 none to the right.
    # Release of the first run + attack of the second fill the middle.
    assert np.array_equal(out, [1.0, 1.0, 1.0, 1.0, 1.0])


def test_dilate_separate_runs():
    mask = np.array([0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    out = _dilate_mask(mask, attack_frames=1, release_frames=0)
    assert np.array_equal(out, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# _signal_aware_pad + context-window helpers
# ---------------------------------------------------------------------------


def test_pad_target_smaller_than_overlap_is_noop():
    assert _signal_aware_pad(10.0, 12.0, 100.0, 1.0) == (10.0, 12.0)


def test_pad_symmetric_when_room_on_both_sides():
    lo, hi = _signal_aware_pad(10.0, 12.0, 100.0, 4.0)
    assert lo == pytest.approx(9.0)
    assert hi == pytest.approx(13.0)


def test_pad_spills_right_when_left_saturated():
    lo, hi = _signal_aware_pad(0.5, 2.5, 100.0, 6.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(6.0)


def test_pad_spills_left_when_right_saturated():
    lo, hi = _signal_aware_pad(97.5, 99.5, 100.0, 6.0)
    assert lo == pytest.approx(94.0)
    assert hi == pytest.approx(100.0)


def test_pad_both_sides_saturated_gives_whole_recording():
    lo, hi = _signal_aware_pad(1.0, 2.0, 3.0, 10.0)
    assert (lo, hi) == (0.0, 3.0)  # narrower than target — recording too short


def test_window_none_is_overlap():
    w = _window_none(1.0, 2.0)
    assert (w.pad_start_s, w.pad_end_s) == (1.0, 2.0)


def test_window_fixed_pad_symmetric():
    w = _window_fixed_pad(10.0, 12.0, 100.0, context_pad_s=1.0)
    assert w.pad_start_s == pytest.approx(9.0)
    assert w.pad_end_s == pytest.approx(13.0)


def test_window_fixed_pad_min_fragment_floor():
    # Natural pad (0.5 + 2*0.25 = 1.0 s) is under the 4 s floor -> widened.
    w = _window_fixed_pad(
        10.0, 10.5, 100.0, context_pad_s=0.25, min_fragment_length_s=4.0
    )
    assert w.pad_end_s - w.pad_start_s == pytest.approx(4.0)
    assert w.pad_start_s == pytest.approx(8.25)


def test_window_expand_to_chunk_hits_target_width():
    w = _window_expand_to_chunk(10.0, 11.0, 100.0, target_total_s=4.0)
    assert w.pad_end_s - w.pad_start_s == pytest.approx(4.0)
    assert w.pad_start_s == pytest.approx(8.5)


def test_window_expand_to_chunk_min_fragment_dominates():
    w = _window_expand_to_chunk(
        10.0, 11.0, 100.0, target_total_s=2.0, min_fragment_length_s=6.0
    )
    assert w.pad_end_s - w.pad_start_s == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Seam helpers
# ---------------------------------------------------------------------------


def test_zero_crossing_finds_nearest_sign_change():
    audio = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    # Transitions at indices 2 and 4; target 3 is equidistant -> first wins.
    assert _pick_seam_zero_crossing(audio, target_idx=3, search_radius_samples=2) == 2


def test_zero_crossing_falls_back_without_transition():
    audio = np.ones(8)
    assert _pick_seam_zero_crossing(audio, target_idx=3, search_radius_samples=2) == 3


def test_zero_crossing_degenerate_window_falls_back():
    audio = np.array([1.0, -1.0, 1.0])
    assert _pick_seam_zero_crossing(audio, target_idx=0, search_radius_samples=0) == 0


def test_extend_start_finds_latest_silence():
    vad = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    assert _extend_start_to_silence(vad, zc_start_idx=4, max_extend_n=4) == 1


def test_extend_start_never_widens_past_budget_and_never_narrows():
    vad = np.ones(5)
    assert _extend_start_to_silence(vad, zc_start_idx=4, max_extend_n=4) == 4
    assert _extend_start_to_silence(vad, zc_start_idx=4, max_extend_n=0) == 4


def test_extend_end_finds_first_silence():
    vad = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert _extend_end_to_silence(vad, zc_end_idx=1, max_extend_n=4) == 3


def test_extend_end_falls_back_without_silence():
    vad = np.ones(5)
    assert _extend_end_to_silence(vad, zc_end_idx=1, max_extend_n=4) == 1


# ---------------------------------------------------------------------------
# _volume_normalise
# ---------------------------------------------------------------------------


def test_volume_none_is_identity():
    s1, s2 = np.ones(100), 2 * np.ones(100)
    o1, o2, scale = _volume_normalise(s1, s2, np.ones(100), "none")
    assert scale == 1.0
    assert np.array_equal(o1, s1) and np.array_equal(o2, s2)


def test_volume_sum_equals_mix_matches_rms():
    mix = np.ones(100, dtype=np.float32)
    s1 = np.ones(100, dtype=np.float32)
    s2 = np.ones(100, dtype=np.float32)
    o1, o2, scale = _volume_normalise(s1, s2, mix, "sum_equals_mix")
    assert scale == pytest.approx(0.5)
    combined_rms = np.sqrt(np.mean((o1 + o2) ** 2))
    assert combined_rms == pytest.approx(1.0, rel=1e-6)


def test_volume_zero_combined_guard():
    z = np.zeros(100)
    o1, o2, scale = _volume_normalise(z, z, np.ones(100), "sum_equals_mix")
    assert scale == 1.0


def test_volume_bad_mode_raises():
    with pytest.raises(ValueError):
        _volume_normalise(np.ones(4), np.ones(4), np.ones(4), "bogus")


# ---------------------------------------------------------------------------
# _pit_swap_if_needed
# ---------------------------------------------------------------------------


def test_pit_keeps_when_straight_wins():
    prev1, prev2 = np.ones(8), -np.ones(8)
    cur1, cur2 = np.ones(8), -np.ones(8)
    r1, r2 = _pit_swap_if_needed(prev1, prev2, cur1, cur2, overlap_samples=8)
    assert np.array_equal(r1, cur1) and np.array_equal(r2, cur2)


def test_pit_swaps_when_swapped_wins():
    prev1, prev2 = np.ones(8), -np.ones(8)
    cur1, cur2 = -np.ones(8), np.ones(8)   # streams flipped in this chunk
    r1, r2 = _pit_swap_if_needed(prev1, prev2, cur1, cur2, overlap_samples=8)
    assert np.array_equal(r1, cur2) and np.array_equal(r2, cur1)


def test_pit_zero_overlap_is_noop():
    cur1, cur2 = np.ones(4), -np.ones(4)
    r1, r2 = _pit_swap_if_needed(np.ones(4), -np.ones(4), cur1, cur2, 0)
    assert r1 is cur1 and r2 is cur2


def test_pit_silent_tail_keeps_order():
    # Zero-norm tails -> cosine 0 for both pairings -> no swap (>= keeps).
    prev1, prev2 = np.zeros(8), np.zeros(8)
    cur1, cur2 = np.ones(8), -np.ones(8)
    r1, r2 = _pit_swap_if_needed(prev1, prev2, cur1, cur2, overlap_samples=8)
    assert np.array_equal(r1, cur1) and np.array_equal(r2, cur2)
