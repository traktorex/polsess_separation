"""CPU unit tests for rescore_stratified.py's pure statistics.

The rescorer is the HEADLINE measurement backbone of the definitive sweep
(recording-clustered CIs + the pre-registered Holm/FDR selection gate,
SWEEP_DESIGN §3.4). These pin the multiple-comparison math added for that gate:

- the bootstrap two-sided p-value (`boot_pvalue`),
- Holm-Bonferroni and Benjamini-Hochberg adjustment (`holm_bonferroni`,
  `benjamini_hochberg`),
- `cluster_boot_paired_draws` agrees with the existing `cluster_boot_paired`
  point estimate (so the gate reads the same effect the CIs report).

No data files / meeteval here (SCOPE §7): only the pure resampling + correction
arithmetic.
"""

import numpy as np
import pytest

from scripts.rescore_stratified import (
    benjamini_hochberg,
    boot_pvalue,
    cluster_boot_paired,
    cluster_boot_paired_draws,
    holm_bonferroni,
)


# ---------------------------------------------------------------------------
# boot_pvalue — two-sided percentile-bootstrap p
# ---------------------------------------------------------------------------


def test_boot_pvalue_all_one_side_is_floored_not_zero():
    # Draws entirely above 0 → strongest resolvable signal, p ≈ 2/n, never 0.
    draws = np.full(1000, 3.0)
    p = boot_pvalue(draws)
    assert 0.0 < p <= 2.0 / 1000 + 1e-9


def test_boot_pvalue_symmetric_around_zero_is_one():
    # Half the draws each side of 0 → p = 2 * 0.5 = 1.0 (cannot reject).
    draws = np.concatenate([np.full(500, -1.0), np.full(500, 1.0)])
    assert boot_pvalue(draws) == pytest.approx(1.0)


def test_boot_pvalue_empty_is_one():
    assert boot_pvalue(np.array([])) == 1.0
    assert boot_pvalue([np.nan, np.nan]) == 1.0


def test_boot_pvalue_partial_overlap_between_zero_and_one():
    # 90% above 0, 10% at/below → frac_ge=0.9, frac_le=0.1 → p = 2*0.1 = 0.2.
    draws = np.concatenate([np.full(900, 2.0), np.full(100, -1.0)])
    assert boot_pvalue(draws) == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_preserves_input_order():
    p = [0.04, 0.01, 0.5]
    adj = holm_bonferroni(p)
    assert len(adj) == 3
    # smallest raw p (index 1) gets multiplied by m=3
    assert adj[1] == pytest.approx(0.03)


def test_holm_is_monotone_and_more_conservative_than_bh():
    p = [0.01, 0.02, 0.03, 0.04]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    # Holm adjusted p >= BH adjusted p for every hypothesis (Holm is FWER, BH FDR).
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh))


def test_holm_a_clearly_significant_arm_survives():
    # One tiny p among large ones: m * p_min must stay < 0.05 to survive Holm.
    p = [0.001, 0.6, 0.7, 0.8]
    holm = holm_bonferroni(p)
    assert holm[0] < 0.05         # survives
    assert all(h >= 0.05 for h in holm[1:])   # the nulls do not


def test_holm_a_marginal_arm_does_not_survive_correction():
    # An uncorrected-significant p (0.03) fails once corrected across 4 arms.
    p = [0.03, 0.5, 0.6, 0.7]
    holm = holm_bonferroni(p)
    assert holm[0] >= 0.05        # 4 * 0.03 = 0.12, does NOT clear the gate


def test_holm_and_bh_empty():
    assert holm_bonferroni([]) == []
    assert benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_bh_adjusted_p_in_unit_interval_and_ordered():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    bh = benjamini_hochberg(p)
    assert all(0.0 <= v <= 1.0 for v in bh)
    # The most significant raw p has the smallest adjusted p.
    assert bh[0] == min(bh)


# ---------------------------------------------------------------------------
# cluster_boot_paired_draws agrees with cluster_boot_paired (same point est.)
# ---------------------------------------------------------------------------


def _recs(vals):
    """{recid: {'E': errors, 'L': length}} from a list of (id, err, len)."""
    return {rid: {"E": e, "L": l} for rid, e, l in vals}


def test_draws_point_matches_existing_cluster_boot_paired():
    # Anchor worse than cfg on every recording → positive paired delta.
    ra = _recs([("r1", 30, 100), ("r2", 40, 100), ("r3", 20, 100)])
    rb = _recs([("r1", 10, 100), ("r2", 12, 100), ("r3", 8, 100)])
    point_old, lo, hi = cluster_boot_paired(ra, rb, "E", "L",
                                            np.random.default_rng(0))
    point_new, draws = cluster_boot_paired_draws(ra, rb, "E", "L",
                                                 np.random.default_rng(0))
    assert point_new == pytest.approx(point_old)
    assert point_new > 0          # anchor has higher error → cfg better
    # Same seed + same B_DRAWS → the draws reproduce the CI percentiles.
    assert np.percentile(draws, [2.5, 97.5]) == pytest.approx([lo, hi])


def test_draws_empty_when_no_common_recording():
    ra = _recs([("r1", 10, 100)])
    rb = _recs([("r2", 10, 100)])
    point, draws = cluster_boot_paired_draws(ra, rb, "E", "L",
                                             np.random.default_rng(0))
    assert np.isnan(point)
    assert draws.size == 0
    assert boot_pvalue(draws) == 1.0     # degenerate → cannot reject
