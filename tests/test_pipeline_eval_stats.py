"""Golden + known-answer tests for asr_pipeline/eval/stats.py.

These pin the campaign statistics extracted verbatim from
``scripts/rescore_stratified.py`` (the ~600-line measurement backbone of the
definitive pipeline sweep — recording-clustered CIs + the pre-registered
Holm/FDR selection gate, SWEEP_DESIGN §3.4). The goldens were captured from the
ORIGINAL script functions *before* extraction (seed 0, B_DRAWS=10000) and the
extracted copies must reproduce them bit-identically. Two kinds of assertion:

- **known-answer** (algebraic, RNG-free): micro-average, boot_pvalue on crafted
  draws, Holm/BH on hand-verifiable p-vectors, tertile strata. Version-proof.
- **seed-0 regression pins** for the cluster bootstrap point estimates + CIs
  (point estimates are RNG-free; CI bounds are seed-0 pinned, exactly as
  ``tests/test_rescore_stratified.py`` already relies on).

Pure numpy only — no meeteval, no data files (SCOPE §7).
"""
import numpy as np
import pytest

from asr_pipeline.eval.stats import (
    B_DRAWS,
    SEED,
    assign_strata,
    benjamini_hochberg,
    boot_pvalue,
    cluster_boot_2key,
    cluster_boot_gap,
    cluster_boot_paired,
    cluster_boot_paired_draws,
    holm_bonferroni,
    micro,
    recording_means,
    significance_star,
)


# ---------------------------------------------------------------------------
# Constants — the extraction must preserve the campaign's fixed seed/draw count
# ---------------------------------------------------------------------------


def test_campaign_constants_unchanged():
    assert B_DRAWS == 10000
    assert SEED == 0


# ---------------------------------------------------------------------------
# micro — pooled (never mean-of-percentages) error rate
# ---------------------------------------------------------------------------


def test_micro_pools_errors_and_lengths():
    rows = [{"E": 3, "L": 10}, {"E": 5, "L": 20}, {"E": 0, "L": 0}]
    assert micro(rows, "E", "L") == pytest.approx(100 * 8 / 30)  # 26.666…


def test_micro_all_zero_length_is_nan():
    assert np.isnan(micro([{"E": 0, "L": 0}], "E", "L"))


def test_micro_skips_rows_missing_length_key():
    rows = [{"E": 2, "L": 10}, {"E": 99}]  # second row has no "L" → skipped
    assert micro(rows, "E", "L") == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# cluster_boot_paired / _draws — golden point + seed-0 CI
# ---------------------------------------------------------------------------

_RA = {"r1": {"E": 30, "L": 100}, "r2": {"E": 40, "L": 100},
       "r3": {"E": 20, "L": 100}, "r4": {"E": 35, "L": 100}}
_RB = {"r1": {"E": 10, "L": 100}, "r2": {"E": 12, "L": 100},
       "r3": {"E": 8,  "L": 100}, "r4": {"E": 15, "L": 100}}


def test_cluster_boot_paired_point_is_micro_delta():
    # anchor micro = 125/400 = 31.25 %, cfg micro = 45/400 = 11.25 % → Δ = 20.0
    # (RNG-free point estimate; + = cfg better).
    p, lo, hi = cluster_boot_paired(_RA, _RB, "E", "L", np.random.default_rng(SEED))
    assert p == pytest.approx(20.0)
    assert (lo, hi) == pytest.approx((14.0, 26.0))   # seed-0 pinned CI


def test_cluster_boot_paired_draws_matches_paired_point_and_ci():
    p0, lo, hi = cluster_boot_paired(_RA, _RB, "E", "L", np.random.default_rng(SEED))
    p1, draws = cluster_boot_paired_draws(_RA, _RB, "E", "L",
                                          np.random.default_rng(SEED))
    assert p1 == pytest.approx(p0)
    assert draws.size == B_DRAWS
    # same seed + resampling → the draws reproduce the CI percentiles exactly.
    assert np.percentile(draws, [2.5, 97.5]) == pytest.approx([lo, hi])


def test_cluster_boot_paired_no_common_recording_is_nan_empty():
    ra = {"r1": {"E": 10, "L": 100}}
    rb = {"r2": {"E": 10, "L": 100}}
    p, draws = cluster_boot_paired_draws(ra, rb, "E", "L", np.random.default_rng(SEED))
    assert np.isnan(p)
    assert draws.size == 0
    assert boot_pvalue(draws) == 1.0


def test_cluster_boot_paired_excludes_zero_length_recordings():
    # r2 has zero reference length on the cfg side → excluded up front, so the
    # paired delta reduces to r1 alone (no 0/0 nan draw contaminates the CI).
    ra = {"r1": {"E": 30, "L": 100}, "r2": {"E": 40, "L": 100}}
    rb = {"r1": {"E": 10, "L": 100}, "r2": {"E": 0, "L": 0}}
    p, _, _ = cluster_boot_paired(ra, rb, "E", "L", np.random.default_rng(SEED))
    assert p == pytest.approx(20.0)   # only r1: 30% - 10%


# ---------------------------------------------------------------------------
# cluster_boot_2key / cluster_boot_gap — golden points + seed-0 CIs
# ---------------------------------------------------------------------------


def test_cluster_boot_2key_golden():
    recs = {"r1": {"aE": 50, "aL": 100, "bE": 20, "bL": 100},
            "r2": {"aE": 60, "aL": 100, "bE": 25, "bL": 100},
            "r3": {"aE": 40, "aL": 100, "bE": 30, "bL": 100}}
    d, lo, hi = cluster_boot_2key(recs, "aE", "aL", "bE", "bL",
                                  np.random.default_rng(SEED))
    assert d == pytest.approx(25.0)               # 150/300 − 75/300 = 50 − 25
    assert (lo, hi) == pytest.approx((10.0, 35.0))  # seed-0 pinned


def test_cluster_boot_gap_golden():
    rag = {"r1": {"cE": 40, "cL": 100, "gE": 20, "gL": 100},
           "r2": {"cE": 50, "cL": 100, "gE": 25, "gL": 100},
           "r3": {"cE": 30, "cL": 100, "gE": 15, "gL": 100}}
    rbg = {"r1": {"cE": 35, "cL": 100, "gE": 30, "gL": 100},
           "r2": {"cE": 45, "cL": 100, "gE": 33, "gL": 100},
           "r3": {"cE": 25, "cL": 100, "gE": 20, "gL": 100}}
    d, lo, hi = cluster_boot_gap(rag, rbg, "cE", "cL", "gE", "gL",
                                 np.random.default_rng(SEED))
    # gap_anchor = 40 − 20 = 20; gap_cfg = 35 − 27.6667 = 7.3333; Δ = 12.6667.
    assert d == pytest.approx(12.6666667)
    assert (lo, hi) == pytest.approx((10.0, 15.0))  # seed-0 pinned


# ---------------------------------------------------------------------------
# boot_pvalue — two-sided percentile-bootstrap p (known answers)
# ---------------------------------------------------------------------------


def test_boot_pvalue_all_one_side_floored_to_resolution():
    assert boot_pvalue(np.full(1000, 3.0)) == pytest.approx(0.001)   # 1/1000, not 0


def test_boot_pvalue_symmetric_is_one():
    draws = np.concatenate([np.full(500, -1.0), np.full(500, 1.0)])
    assert boot_pvalue(draws) == pytest.approx(1.0)


def test_boot_pvalue_partial_overlap():
    draws = np.concatenate([np.full(900, 2.0), np.full(100, -1.0)])
    assert boot_pvalue(draws) == pytest.approx(0.2)   # 2 * min(0.1, 0.9)


def test_boot_pvalue_empty_or_all_nan_is_one():
    assert boot_pvalue(np.array([])) == 1.0
    assert boot_pvalue(np.array([np.nan, np.nan])) == 1.0


# ---------------------------------------------------------------------------
# Holm-Bonferroni / Benjamini-Hochberg — hand-verifiable known answers
# ---------------------------------------------------------------------------


def test_holm_known_answer_preserves_input_order():
    # order idx1(0.01),idx0(0.04),idx2(0.5): 3*.01=.03, 2*.04=.08, 1*.5=.5.
    assert holm_bonferroni([0.04, 0.01, 0.5]) == pytest.approx([0.08, 0.03, 0.5])


def test_holm_known_answer_monotone_running_max():
    assert holm_bonferroni([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.06, 0.06, 0.06])


def test_bh_known_answer():
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.04, 0.04, 0.04])
    assert benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.9]) == pytest.approx(
        [0.005, 0.02, 0.05125, 0.05125, 0.9])


def test_holm_marginal_arm_does_not_clear_gate():
    # 0.03 uncorrected-significant → 4*0.03 = 0.12, does NOT clear α=0.05.
    holm = holm_bonferroni([0.03, 0.5, 0.6, 0.7])
    assert holm == pytest.approx([0.12, 1.0, 1.0, 1.0])
    assert holm[0] >= 0.05


def test_holm_is_at_least_as_conservative_as_bh():
    p = [0.01, 0.02, 0.03, 0.04]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh))


def test_holm_and_bh_empty():
    assert holm_bonferroni([]) == []
    assert benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# significance_star — the CI-excludes-zero flag (nan CI is NOT significant)
# ---------------------------------------------------------------------------


def test_significance_star_only_when_ci_excludes_zero():
    assert significance_star(1.0, 3.0) == " *"     # wholly positive
    assert significance_star(-3.0, -1.0) == " *"    # wholly negative
    assert significance_star(-1.0, 2.0) == ""       # straddles 0
    assert significance_star(0.0, 2.0) == ""        # touches 0
    assert significance_star(float("nan"), 2.0) == ""   # nan → not significant
    assert significance_star(1.0, float("nan")) == ""


# ---------------------------------------------------------------------------
# strata — recording-level tertiles (recording_means + assign_strata)
# ---------------------------------------------------------------------------


def test_recording_means_collapses_fragments_by_recid():
    frag_scores = {"rec1__0": 10.0, "rec1__1": 20.0, "rec2__0": 5.0}
    assert recording_means(frag_scores) == {"rec1": 15.0, "rec2": 5.0}


def test_recording_means_custom_recid():
    frag_scores = {"a-0": 4.0, "a-1": 6.0, "b-0": 1.0}
    means = recording_means(frag_scores, recid=lambda f: f.split("-")[0])
    assert means == {"a": 5.0, "b": 1.0}


def test_assign_strata_even_thirds():
    means = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    strat = assign_strata(means)
    assert strat == {"a": "LOW", "b": "LOW", "c": "MID", "d": "MID",
                     "e": "HIGH", "f": "HIGH"}


def test_assign_strata_non_multiple_of_three_extras_to_high():
    # 7 recordings → t = 2: LOW=2, MID=2, HIGH=3 (the extra lands in HIGH).
    means = {f"r{i}": i for i in range(7)}
    strat = assign_strata(means)
    counts = {s: sum(v == s for v in strat.values()) for s in ("LOW", "MID", "HIGH")}
    assert counts == {"LOW": 2, "MID": 2, "HIGH": 3}
    assert strat["r0"] == "LOW" and strat["r6"] == "HIGH"


def test_assign_strata_partitions_every_recording():
    means = {f"r{i}": float(i % 4) for i in range(11)}  # ties allowed
    strat = assign_strata(means)
    assert set(strat) == set(means)                      # no recording dropped
    assert set(strat.values()) <= {"LOW", "MID", "HIGH"}
