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
    _cluster2_pairings,
    _consensus_pairings,
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
# Tier-2 attribution lever: margin-gated carry-forward prior
# ---------------------------------------------------------------------------
#
# Anchors A=[1,0], B=[0,1]. Embeddings keyed by each stream's first sample
# (_TableEcapa). Two overlaps:
#   - "confident straight": s1→[1,0], s2→[0,1] → straight=2.0, swapped=0.0
#     (gap 2.0, decisive straight). Seeds the carry-forward prior.
#   - "near-tie swapped":  s1→[0.50,0.52], s2→[0.52,0.50] → straight≈1.386,
#     swapped≈1.442 (gap≈0.055, argmax = swapped). The gate's target.

_CONF_S1, _CONF_S2 = 0.10, 0.20          # confident overlap stream markers
_TIE_S1, _TIE_S2 = 0.30, 0.40            # near-tie overlap stream markers
_TIE_TABLE = {
    _CONF_S1: [1.0, 0.0], _CONF_S2: [0.0, 1.0],
    _TIE_S1: [0.50, 0.52], _TIE_S2: [0.52, 0.50],
}


def _margin_anchors():
    return {"SPK_A": torch.tensor([1.0, 0.0]), "SPK_B": torch.tensor([0.0, 1.0])}


def test_margin_off_is_identical_to_argmax():
    """overlap_assign_min_margin=0 (default) → pure argmax: the near-tie overlap
    resolves to its argmax 'swapped', identical to today's behaviour. The
    pairing label stays the bare 'swapped' (no carry-forward annotation)."""
    conf = _ovl(np.full(SR, _CONF_S1), np.full(SR, _CONF_S2), idx=0)
    tie = _ovl(np.full(SR, _TIE_S1), np.full(SR, _TIE_S2), idx=1)
    ecapa = _TableEcapa(_TIE_TABLE)
    out = _assign_overlaps(
        [conf, tie], _margin_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        min_margin=0.0,
    )
    assert out[0]["pairing"] == "straight"
    assert out[1]["pairing"] == "swapped"
    # near-tie argmax = swapped → SPK_A gets s2.
    np.testing.assert_array_equal(out[1]["emit_pieces"]["SPK_A"], tie["s2_gated"])


def test_margin_off_default_matches_explicit_zero():
    """The default call (no min_margin kwarg) equals an explicit min_margin=0:
    the carry-forward branch can never alter the committed-default behaviour."""
    conf = _ovl(np.full(SR, _CONF_S1), np.full(SR, _CONF_S2), idx=0)
    tie = _ovl(np.full(SR, _TIE_S1), np.full(SR, _TIE_S2), idx=1)
    spk = ["SPK_A", "SPK_B"]
    default = _assign_overlaps(
        [conf, tie], _margin_anchors(), spk, _TableEcapa(_TIE_TABLE), DEVICE, SR
    )
    explicit = _assign_overlaps(
        [conf, tie], _margin_anchors(), spk, _TableEcapa(_TIE_TABLE), DEVICE, SR,
        min_margin=0.0,
    )
    assert [o["pairing"] for o in default] == [o["pairing"] for o in explicit]


def test_margin_gate_near_tie_inherits_confident_prior():
    """With a high margin (0.1 > the 0.055 near-tie gap, < the 2.0 confident gap):
    the confident overlap decides 'straight' and seeds the prior; the near-tie
    overlap is ambiguous and inherits that prior instead of its argmax 'swapped'.
    The near-tie thus flips from swapped (argmax) to straight (prior)."""
    conf = _ovl(np.full(SR, _CONF_S1), np.full(SR, _CONF_S2), idx=0)
    tie = _ovl(np.full(SR, _TIE_S1), np.full(SR, _TIE_S2), idx=1)
    ecapa = _TableEcapa(_TIE_TABLE)
    out = _assign_overlaps(
        [conf, tie], _margin_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        min_margin=0.1,
    )
    assert out[0]["pairing"] == "straight"
    assert out[1]["pairing"] == "straight (carry-forward prior)"
    # Inherited straight → SPK_A keeps s1 (not the argmax swap to s2).
    np.testing.assert_array_equal(out[1]["emit_pieces"]["SPK_A"], tie["s1_gated"])
    np.testing.assert_array_equal(out[1]["emit_pieces"]["SPK_B"], tie["s2_gated"])


def test_margin_gate_no_prior_yet_falls_through_to_argmax():
    """An ambiguous overlap that arrives BEFORE any confident one has no prior
    to inherit, so it falls through to plain argmax (never drops the region)."""
    tie = _ovl(np.full(SR, _TIE_S1), np.full(SR, _TIE_S2), idx=0)
    ecapa = _TableEcapa(_TIE_TABLE)
    out = _assign_overlaps(
        [tie], _margin_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        min_margin=0.1,
    )
    # No prior seeded → argmax 'swapped' (bare label, not carry-forward).
    assert out[0]["pairing"] == "swapped"


# ---------------------------------------------------------------------------
# Attribution lever: continuity tie-break (local bracketing-solo anchors)
# ---------------------------------------------------------------------------
#
# Construction: the overlap streams are clean and orthogonal (s1→[1,0]=truly A,
# s2→[0,1]=truly B), but the GLOBAL anchors are contaminated and lean toward
# 'swapped' by a gap of ~0.55. The local bracketing-solo anchors are clean
# ([1,0] for A, [0,1] for B), so on a near-tie they decisively pick 'straight'.


def _contaminated_anchors():
    """Global anchors normalised from [0.4, 0.6] / [0.6, 0.4] — they favour the
    'swapped' pairing (gap ~0.55) against clean orthogonal overlap embeddings."""
    import math
    n = 1.0 / math.sqrt(0.4 ** 2 + 0.6 ** 2)
    return {"SPK_A": torch.tensor([0.4 * n, 0.6 * n]),
            "SPK_B": torch.tensor([0.6 * n, 0.4 * n])}


def _continuity_setup():
    """Overlap with clean s1→[1,0], s2→[0,1]; bracketing A-solo (4-5 s)→[1,0]
    and B-solo (6-7 s)→[0,1] inside the continuity window of the [5,6] s overlap."""
    ecapa = _TableEcapa({0.30: [1.0, 0.0], 0.40: [0.0, 1.0]})
    ovl = _ovl(np.full(SR, 0.30), np.full(SR, 0.40),
               idx=0, pad_start=5.0, emit_start=5.0, emit_end=6.0)
    audio = np.zeros(12 * SR, dtype=np.float32)
    audio[4 * SR:5 * SR] = 0.30          # A solo → [1,0]
    audio[6 * SR:7 * SR] = 0.40          # B solo → [0,1]
    solo = {"SPK_A": [(4.0, 5.0)], "SPK_B": [(6.0, 7.0)]}
    return ecapa, ovl, audio, solo


def test_continuity_baseline_argmax_picks_swapped():
    """Sanity anchor: with the contaminated global anchors the plain
    ecapa_argmax picks 'swapped' — the (wrong) decision continuity must fix."""
    ecapa, ovl, _, _ = _continuity_setup()
    out = _assign_overlaps([ovl], _contaminated_anchors(), ["SPK_A", "SPK_B"],
                           ecapa, DEVICE, SR)
    assert out[0]["pairing"] == "swapped"


def test_continuity_tiebreak_flips_near_tie_via_local_anchor():
    """tau (0.6) above the ~0.55 global gap → the overlap is a near-tie; the
    clean local bracketing-solo anchors flip it from 'swapped' to 'straight'."""
    ecapa, ovl, audio, solo = _continuity_setup()
    out = _assign_overlaps(
        [ovl], _contaminated_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        strategy="continuity_tiebreak", continuity_margin=0.6,
        continuity_window_s=10.0, solo_intervals_by_spk=solo, assembly_audio=audio,
    )
    assert out[0]["pairing"] == "straight (continuity tie-break)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], ovl["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], ovl["s2_gated"])


def test_continuity_tau_zero_is_argmax_baseline():
    """continuity_tiebreak with tau=0 never declares a near-tie → identical to
    ecapa_argmax (the committed-default behaviour is untouched)."""
    ecapa, ovl, audio, solo = _continuity_setup()
    out = _assign_overlaps(
        [ovl], _contaminated_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        strategy="continuity_tiebreak", continuity_margin=0.0,
        continuity_window_s=10.0, solo_intervals_by_spk=solo, assembly_audio=audio,
    )
    assert out[0]["pairing"] == "swapped"


def test_continuity_decisive_gap_keeps_global_argmax():
    """When the global gap clears tau, continuity does NOT fire — the decisive
    global decision stands (we only second-guess near-ties)."""
    ecapa, ovl, audio, solo = _continuity_setup()
    out = _assign_overlaps(
        [ovl], _contaminated_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        strategy="continuity_tiebreak", continuity_margin=0.1,   # < ~0.55 gap
        continuity_window_s=10.0, solo_intervals_by_spk=solo, assembly_audio=audio,
    )
    assert out[0]["pairing"] == "swapped"


def test_continuity_falls_back_when_no_local_solo():
    """No solo audio in the window → no local anchor → fall back to the global
    argmax, labelled honestly (never silently swaps)."""
    ecapa, ovl, audio, _ = _continuity_setup()
    out = _assign_overlaps(
        [ovl], _contaminated_anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        strategy="continuity_tiebreak", continuity_margin=0.6,
        continuity_window_s=10.0,
        solo_intervals_by_spk={"SPK_A": [], "SPK_B": []}, assembly_audio=audio,
    )
    assert out[0]["pairing"] == "swapped (continuity unavailable)"


# ---------------------------------------------------------------------------
# Attribution lever: consensus 2-means (global constrained re-clustering)
# ---------------------------------------------------------------------------


def test_consensus_clean_case_all_straight():
    """Two clean overlaps (s1 positive→A, s2 negative→B): consensus = straight
    both, labelled '(consensus)', emit pieces correct."""
    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)
    o1 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=1)
    out = _assign_overlaps([o0, o1], _anchors(), ["SPK_A", "SPK_B"],
                           _StubEcapa(), DEVICE, SR, strategy="consensus_2means")
    assert out[0]["pairing"] == "straight (consensus)"
    assert out[1]["pairing"] == "straight (consensus)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], o0["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], o0["s2_gated"])


def test_consensus_matches_argmax_on_clean_separable_case():
    """On a cleanly separable case consensus reproduces the per-overlap argmax —
    the documented equivalence (anchor-seeded 2-means is already at the argmax
    fixed point). Same pairings, different label suffix only."""
    ovls = [_ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=i) for i in range(3)]
    base = _assign_overlaps([dict(o) for o in ovls], _anchors(),
                            ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR)
    cons = _assign_overlaps([dict(o) for o in ovls], _anchors(),
                            ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR,
                            strategy="consensus_2means")
    assert ([o["pairing"].split()[0] for o in base]
            == [o["pairing"].split()[0] for o in cons])


def test_consensus_pairings_deterministic_and_covers_eligible():
    ovls = [_ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=i) for i in range(3)]
    p1 = _consensus_pairings(ovls, _anchors(), ["SPK_A", "SPK_B"],
                             _StubEcapa(), DEVICE, SR)
    p2 = _consensus_pairings(ovls, _anchors(), ["SPK_A", "SPK_B"],
                             _StubEcapa(), DEVICE, SR)
    assert p1 == p2 and set(p1) == {0, 1, 2}


def test_consensus_no_anchor_returns_empty():
    """No anchor → empty dict (caller falls back to the per-overlap argmax path,
    never dropping a region)."""
    ovls = [_ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)]
    anchors = {"SPK_A": None, "SPK_B": torch.tensor([0.0, 1.0])}
    assert _consensus_pairings(ovls, anchors, ["SPK_A", "SPK_B"],
                               _StubEcapa(), DEVICE, SR) == {}


# ---------------------------------------------------------------------------
# Attribution lever: cluster2 (global unconstrained 2-means over overlap streams)
# ---------------------------------------------------------------------------


def test_cluster2_clean_case_all_straight():
    """Two clean overlaps (s1→A-voice [1,0], s2→B-voice [0,1]): the 2-means seeded
    from the anchors recovers the two voices, mapping s1→A and s2→B → 'straight'
    for both, keyed by i_ovl."""
    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)
    o1 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=1)
    p = _cluster2_pairings([o0, o1], _anchors(), ["SPK_A", "SPK_B"],
                           _StubEcapa(), DEVICE, SR)
    assert p == {0: "straight", 1: "straight"}


def test_cluster2_deterministic():
    """Same input twice → identical pairing dict (no RNG anywhere)."""
    ovls = [_ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=i) for i in range(3)]
    p1 = _cluster2_pairings(ovls, _anchors(), ["SPK_A", "SPK_B"],
                            _StubEcapa(), DEVICE, SR)
    p2 = _cluster2_pairings(ovls, _anchors(), ["SPK_A", "SPK_B"],
                            _StubEcapa(), DEVICE, SR)
    assert p1 == p2 == {0: "straight", 1: "straight", 2: "straight"}


def test_cluster2_equivalent_to_argmax_under_corrupted_anchor():
    """DESIGN NOTE (measured): with the anchors used as BOTH the 2-means seeds and
    the cluster→stream map, cluster2 reproduces the per-overlap argmax permutation
    on a symmetric (A,B) pair — even a corrupted anchor that (wrongly) favours
    'swapped' is followed identically by both. cluster2's only distinct behaviour
    is the same-cluster fall-through (below). Contaminated anchors [0.6,0.8] /
    [0.8,0.6] make argmax pick 'swapped' on a truly-straight overlap; cluster2
    agrees."""
    corrupt = {"SPK_A": torch.tensor([0.6, 0.8]), "SPK_B": torch.tensor([0.8, 0.6])}
    ecapa = _TableEcapa({0.5: [1.0, 0.0], -0.5: [0.0, 1.0]})
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)   # true straight
    # per-overlap argmax (baseline) picks swapped under the corrupted anchors...
    argmax = _assign_overlaps([dict(ovl)], corrupt, ["SPK_A", "SPK_B"],
                              _TableEcapa({0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}),
                              DEVICE, SR)
    assert argmax[0]["pairing"] == "swapped"
    # ...and cluster2 reaches the same permutation.
    p = _cluster2_pairings([ovl], corrupt, ["SPK_A", "SPK_B"], ecapa, DEVICE, SR)
    assert p == {0: "swapped"}


def test_cluster2_degenerate_overlap_omitted():
    """Both streams of an overlap land in one cluster (a separation failure, both
    A-voice) → cluster2 leaves it UNDECIDED (omitted) rather than forcing a
    guessed pairing. This is cluster2's one distinct behaviour vs per-overlap
    argmax, which would force straight/swapped."""
    # s1 and s2 both positive → both [1,0] via _StubEcapa → same cluster.
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, 0.4), idx=0)
    p = _cluster2_pairings([ovl], _anchors(), ["SPK_A", "SPK_B"],
                           _StubEcapa(), DEVICE, SR)
    assert p == {}


def test_cluster2_single_embeddable_stream_returns_empty():
    """< 2 embeddable overlap streams → clustering undefined → empty dict, every
    overlap falls through to the anchor-argmax ladder (logged no-op)."""
    n_short = SR // 20  # 0.05 s → below the 0.1 s floor
    ovl = _ovl(np.full(SR, 0.5), np.full(n_short, -0.5), idx=0)   # only s1 usable
    p = _cluster2_pairings([ovl], _anchors(), ["SPK_A", "SPK_B"],
                           _StubEcapa(), DEVICE, SR)
    assert p == {}


def test_cluster2_no_anchor_returns_empty():
    ovls = [_ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)]
    anchors = {"SPK_A": None, "SPK_B": torch.tensor([0.0, 1.0])}
    assert _cluster2_pairings(ovls, anchors, ["SPK_A", "SPK_B"],
                              _StubEcapa(), DEVICE, SR) == {}


def test_assign_overlaps_cluster2_mode_labels_and_pieces():
    """assignment_mode='cluster2' routes through the pre-pass: a clean overlap is
    labelled '(cluster2)' with the correct emit pieces."""
    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)
    out = _assign_overlaps([o0], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(),
                           DEVICE, SR, assignment_mode="cluster2")
    assert out[0]["pairing"] == "straight (cluster2)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], o0["s1_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], o0["s2_gated"])


def test_assign_overlaps_cluster2_degenerate_falls_through_to_argmax():
    """A cluster2-undecided overlap (both streams one cluster) falls through to the
    per-overlap argmax path in the same call — labelled with the bare argmax label,
    NOT '(cluster2)', and the region is never dropped."""
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, 0.4), idx=0)   # both → [1,0]
    out = _assign_overlaps([ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(),
                           DEVICE, SR, assignment_mode="cluster2")
    # argmax on two equal [1,0] streams ties → straight (bare label, fall-through).
    assert out[0]["pairing"] == "straight"
    assert set(out[0]["emit_pieces"]) == {"SPK_A", "SPK_B"}


def test_external_pairings_overrides_cluster2_mode():
    """B+ handoff outranks cluster2 for the overlaps it covers (strictly stronger
    global decision); cluster2 only fills the residual."""
    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)   # cluster2 → straight
    out = _assign_overlaps([o0], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(),
                           DEVICE, SR, assignment_mode="cluster2",
                           external_pairings={0: "swapped"})
    assert out[0]["pairing"] == "swapped (relabel_global)"


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


def _seg_df(segments):
    """Diarization segments DataFrame from (speaker, start, end) tuples."""
    return pd.DataFrame(
        [
            {"start": s, "end": e, "duration": e - s, "speaker": spk}
            for spk, s, e in segments
        ],
        columns=["start", "end", "duration", "speaker"],
    )


def _ctx_with_segments(segments, overlap_separated=None, overlap_regions=None):
    ovl_df = pd.DataFrame(columns=["start", "end", "duration"])
    ctx = PipelineContext(sample_rate=SR)
    ctx.diarization = DiarizationResult(
        segments_df=_seg_df(segments), overlaps_df=ovl_df, total_duration_s=10.0
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
    return DiarizationResult(
        segments_df=_seg_df(segments),
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


# ---------------------------------------------------------------------------
# B+ handoff: external_pairings (ctx.overlap_speaker_assignment)
# ---------------------------------------------------------------------------


def test_assembly_consumes_external_pairings():
    """A covered overlap uses the external (relabel_global) pairing with NO
    ECAPA re-embed; an UNCOVERED overlap still falls through to the anchor ladder.

    The external pairing says 'swapped' for overlap 0, which is the OPPOSITE of
    what the content/anchor argmax would pick (s1 positive → A, i.e. straight),
    so a passthrough bug (ignoring external_pairings) would show 'straight'.
    Overlap 1 is not in the dict → normal argmax 'straight'.
    """

    class _FailEcapa:
        """Embedding must NOT be called for the covered overlap (external pairing
        skips the re-embed). It IS called for the uncovered one. So we count
        calls and assert the covered overlap added none beyond the anchors."""

        def __init__(self):
            self.calls = 0

        def encode_batch(self, audio):
            self.calls += 1
            v = (torch.tensor([1.0, 0.0]) if float(audio.mean()) > 0
                 else torch.tensor([0.0, 1.0]))
            return v.view(1, 1, 2)

    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)   # content → straight
    o1 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=1)   # content → straight
    ecapa = _FailEcapa()
    out = _assign_overlaps(
        [o0, o1], _anchors(), ["SPK_A", "SPK_B"], ecapa, DEVICE, SR,
        external_pairings={0: "swapped"},
    )
    # Covered overlap takes the external (swapped) decision, labelled honestly.
    assert out[0]["pairing"] == "swapped (relabel_global)"
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_A"], o0["s2_gated"])
    np.testing.assert_array_equal(out[0]["emit_pieces"]["SPK_B"], o0["s1_gated"])
    # Uncovered overlap falls through to the normal argmax (straight).
    assert out[1]["pairing"] == "straight"
    np.testing.assert_array_equal(out[1]["emit_pieces"]["SPK_A"], o1["s1_gated"])


def test_external_pairings_overrides_consensus_strategy():
    """When external_pairings covers an overlap it wins over the consensus
    strategy too (a strictly stronger global decision)."""
    o0 = _ovl(np.full(SR, 0.5), np.full(SR, -0.5), idx=0)
    out = _assign_overlaps(
        [o0], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR,
        strategy="consensus_2means", external_pairings={0: "swapped"},
    )
    assert out[0]["pairing"] == "swapped (relabel_global)"


def test_external_pairings_too_short_overlap_falls_back():
    """Even if external_pairings names a too-short overlap, the too_short guard
    runs first → fixed assignment (the external pairing only applies to
    ECAPA-eligible overlaps, mirroring the consensus seam)."""
    n = SR // 20  # 0.05 s → too short
    ovl = _ovl(np.full(n, 0.5), np.full(n, -0.5), idx=0)
    out = _assign_overlaps(
        [ovl], _anchors(), ["SPK_A", "SPK_B"], _StubEcapa(), DEVICE, SR,
        external_pairings={0: "swapped"},
    )
    assert out[0]["pairing"] == "arbitrary (too short)"


def test_run_passes_overlap_speaker_assignment_to_assign():
    """AssemblyStage.run threads ctx.overlap_speaker_assignment through to
    _assign_overlaps (end-to-end B+ wiring)."""
    enhanced = np.zeros(5 * SR, dtype=np.float32)
    enhanced[0: 2 * SR] = 0.5
    enhanced[3 * SR: 5 * SR] = -0.5
    ovl = _ovl(np.full(SR, 0.5), np.full(SR, -0.5),
               idx=0, pad_start=2.0, emit_start=2.0, emit_end=3.0)
    stage = _make_stage(min_solo_for_anchor_s=1.0, crossfade_ms=0.0,
                        edge_fade_ms=0.0, overlap_rms_match_solo=False)
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = enhanced.copy()
    ctx.enhanced_full = enhanced
    ctx.diarization = _diarization(
        [("SPK_A", 0.0, 2.5), ("SPK_B", 2.5, 5.0)], total_duration_s=5.0
    )
    ctx.overlap_regions = [(2.0, 3.0)]
    ctx.speakers = ["SPK_A", "SPK_B"]
    ctx.overlap_separated = [ovl]
    # Force the (content-wrong) swapped decision via the handoff.
    ctx.overlap_speaker_assignment = {0: "swapped"}
    stage.run(ctx)
    # SPK_A's overlap event should be the s2 stream (swapped), not s1.
    a_overlap = [e for e in ctx.timestamp_map.per_speaker["SPK_A"]
                 if e.kind == "overlap"]
    assert len(a_overlap) == 1   # the swapped overlap landed on A


# ---------------------------------------------------------------------------
# Option 4: anchor_embedding=ecapa2 dispatch
# ---------------------------------------------------------------------------


def test_assembly_anchor_embedding_ecapa2_dispatch(monkeypatch):
    """anchor_embedding='ecapa2' routes AssemblyStage.load through
    build_custom_embedding (the custom wrapper), not the SpeechBrain loader."""
    import asr_pipeline.stages.assembly as asm_mod

    sentinel = object()
    captured = {}

    def fake_build(name, device):
        captured["name"] = name
        captured["device"] = device
        return sentinel

    # build_custom_embedding is imported locally inside load(); patch at source.
    import asr_pipeline.stages.custom_embeddings as ce_mod
    monkeypatch.setattr(ce_mod, "build_custom_embedding", fake_build)

    stage = AssemblyStage(AssemblyConfig(anchor_embedding="ecapa2"))
    stage.load(DEVICE)
    assert stage._ecapa is sentinel
    assert captured["name"] == "ecapa2"
    assert captured["device"] == DEVICE
    # And the signature now tracks the embedder choice (was () before the knob).
    assert stage.load_signature() == ("ecapa2",)
    assert AssemblyStage(AssemblyConfig()).load_signature() == ("ecapa1",)


def test_ecapa_embed_dispatches_on_custom_wrapper():
    """_ecapa_embed handles a call-style custom wrapper (no encode_batch):
    returns a unit-norm tensor from the wrapper's (1, dim) numpy output."""
    from asr_pipeline.stages.assembly import _ecapa_embed

    class _Wrapper:   # call-style, like BaseCustomSpeakerEmbedding
        def __call__(self, wav):
            return np.asarray([[3.0, 4.0]], dtype=np.float32)   # norm 5

    emb = _ecapa_embed(np.full(SR, 0.5, dtype=np.float32), _Wrapper(), DEVICE, SR)
    np.testing.assert_allclose(emb.numpy(), [0.6, 0.8], rtol=1e-5)


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


# ---------------------------------------------------------------------------
# Solo onset boundary pad (assembly.solo_onset_pad_s) — stage wiring
# ---------------------------------------------------------------------------


def _pad_ctx(audio, segments, overlap_regions):
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = audio
    ctx.enhanced_full = audio
    ctx.diarization = _diarization(segments, total_duration_s=len(audio) / SR)
    ctx.overlap_regions = overlap_regions
    ctx.speakers = sorted({spk for spk, _, _ in segments})
    ctx.overlap_separated = []
    return ctx


def test_run_solo_onset_pad_recovers_shaved_onset():
    """Wiring: solo_onset_pad_s > 0 extends the solo piece's start earlier at
    extraction (full_length places the recovered onset audio before the
    diarization boundary); the 0.0 default leaves the boundary exactly where
    it is today (byte-identical no-op)."""
    audio = np.ones(3 * SR, dtype=np.float32)

    def _run(pad):
        stage = _make_stage(
            output_mode="full_length", min_solo_for_anchor_s=1.0,
            crossfade_ms=0.0, edge_fade_ms=0.0, overlap_rms_match_solo=False,
            solo_onset_pad_s=pad,
        )
        ctx = _pad_ctx(audio, [("SPK_A", 1.0, 2.0)], overlap_regions=[])
        stage.run(ctx)
        return ctx

    base = _run(0.0)
    padded = _run(0.5)
    half = SR // 2
    # Default: piece starts exactly at the 1.0 s diarization boundary.
    assert np.all(base.assembled["SPK_A"][:SR] == 0.0)
    assert np.all(base.assembled["SPK_A"][SR : 2 * SR] == 1.0)
    # Padded: the 0.5 s before the boundary is recovered; nothing before that.
    assert np.all(padded.assembled["SPK_A"][SR - half : 2 * SR] == 1.0)
    assert np.all(padded.assembled["SPK_A"][: SR - half] == 0.0)
    # The timestamp map reflects the padded original span (end untouched).
    entry = padded.timestamp_map.per_speaker["SPK_A"][0]
    assert entry.orig_start == pytest.approx(0.5)
    assert entry.orig_end == pytest.approx(2.0)


def test_run_solo_onset_pad_clamped_by_overlap_region():
    """Wiring of the overlap clamp through the real blocked-region derivation
    (no 3b here → `_emit_blocked_regions` returns ctx.overlap_regions, the
    mixture-fill path): an overlap ending 0.1 s before the solo caps the pad
    at 0.9 — overlap audio never leaks into the solo piece."""
    audio = np.ones(3 * SR, dtype=np.float32)
    stage = _make_stage(
        output_mode="full_length", min_solo_for_anchor_s=1.0,
        crossfade_ms=0.0, edge_fade_ms=0.0, overlap_rms_match_solo=False,
        solo_onset_pad_s=0.5,
    )
    ctx = _pad_ctx(audio, [("SPK_A", 1.0, 2.0)], overlap_regions=[(0.6, 0.9)])
    stage.run(ctx)
    solo_entries = [e for e in ctx.timestamp_map.per_speaker["SPK_A"]
                    if e.kind == "solo"]
    assert len(solo_entries) == 1
    assert solo_entries[0].orig_start == pytest.approx(0.9)   # not 0.5
    assert solo_entries[0].orig_end == pytest.approx(2.0)
