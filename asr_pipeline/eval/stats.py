"""Campaign statistics — recording-clustered bootstrap + multiple-comparison
correction, extracted from ``scripts/rescore_stratified.py``.

These are the pure-arithmetic backbone of the definitive pipeline sweep: the
recording-clustered paired bootstrap that produced every CI in the thesis
campaign, the two-sided bootstrap p-value feeding the pre-registered
Holm-Bonferroni **selection gate** (SWEEP_DESIGN §3.4; BH/FDR reported
alongside), the micro-average used for every WER/CER cell, and the acoustic-
complexity tertile assignment done at the RECORDING level (so a multi-segment
recording lands in exactly one stratum). They backed:

- the one-shot **test** decision that only the separator (nosep arm) cleared
  Holm (``docs/sweep_plan/TEST_ANALYSIS.md``);
- the **dev** finding that OA-0.5 was the sole Holm-significant knob
  (``docs/sweep_plan/ADAPTIVE_ANALYSIS.md``);
- the shipped-best selection (``configs/sweep_best_e31_refineplus.yaml``).

Extracted verbatim (bit-identical: same ``B_DRAWS``, same RNG-draw order, same
seed convention) so those thesis tables are regenerable through package code and
unit-tested here rather than trapped in a ~600-line script. ``scripts/
rescore_stratified.py`` re-imports these names and remains its own CLI driver.

Statistics only — no meeteval, no torch, no data-file IO (that stays in the
script). The caller owns RNG seeding: pass ``np.random.default_rng(SEED)`` and
the resampling is reproducible.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Bootstrap draw count + the campaign's fixed seed. ``B_DRAWS=10000`` sets the
# smallest resolvable bootstrap p (~2/B_DRAWS); ``SEED=0`` is the one-shot,
# pre-registered seed (never re-rolled — SWEEP_DESIGN §3.4).
B_DRAWS, SEED = 10000, 0


# ---------------------------------------------------------------------------
# Micro-average
# ---------------------------------------------------------------------------


def micro(rows: Iterable[Dict[str, float]], eK: str, lK: str) -> float:
    """Micro-averaged error rate (%) over ``rows``: ``100 * Σerrors / Σlength``.

    Micro (pool errors and lengths, then divide) — never a plain mean of
    per-fragment percentages, which would mis-weight short fragments. Rows
    missing the length key ``lK`` are skipped; an all-zero-length pool returns
    ``nan`` (0/0, e.g. an all-filler backchannel exchange). The one aggregation
    used for every WER/CER cell in the campaign tables."""
    e = sum(r[eK] for r in rows if lK in r)
    l = sum(r[lK] for r in rows if lK in r)
    return 100 * e / l if l else float("nan")


# ---------------------------------------------------------------------------
# Recording-clustered bootstrap
# ---------------------------------------------------------------------------


def _boot_resample(ids: Sequence[str], stat: Callable[[List[str]], float],
                   rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    """The shared cluster-bootstrap kernel: point estimate + the finite draws.

    ``stat(sample) -> float`` is the paired statistic over a list of recording ids
    (a resampled ``sample`` may repeat ids). Resamples the RECORDINGS with
    replacement ``B_DRAWS`` times and drops any non-finite draw (a resample can sum
    to a 0/0 nan; one nan poisons ``np.percentile`` and would flip a significance
    star ON for a meaningless delta). Returns ``(point, finite_draws_ndarray)`` —
    ``point`` is ``stat(ids)`` (nan and empty draws when ``ids`` is empty). The
    four bootstrap wrappers below differ ONLY in how they build ``ids`` and
    ``stat``; the resample/nan-guard/percentile logic lives here once.

    Clustering by RECORDING (not fragment) is the campaign's core correctness
    fix — fragment-level resampling fabricates significance for multi-segment
    recordings (88741282 has 3 segments, 9a651086 has 2)."""
    if not ids:
        return float("nan"), np.array([])
    point = stat(ids)
    idx = np.arange(len(ids))
    draws = np.array([stat([ids[i] for i in rng.choice(idx, len(idx), replace=True)])
                      for _ in range(B_DRAWS)])
    return point, draws[np.isfinite(draws)]


def _boot_ci(ids: Sequence[str], stat: Callable[[List[str]], float],
             rng: np.random.Generator) -> Tuple[float, float, float]:
    """``(point, lo, hi)`` — the 95% percentile CI form of :func:`_boot_resample`.
    nan CI bounds when ``ids`` is empty or no draw is finite."""
    point, draws = _boot_resample(ids, stat, rng)
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def _paired_ids(recs_a: Dict[str, Dict[str, float]],
                recs_b: Dict[str, Dict[str, float]], lK: str) -> List[str]:
    """Recordings present on both sides with non-zero reference length on each —
    so a resample can never sum to a 0/0 nan draw."""
    return [r for r in recs_a if r in recs_b
            and recs_a[r].get(lK, 0) > 0 and recs_b[r].get(lK, 0) > 0]


def _paired_delta(recs_a: Dict[str, Dict[str, float]],
                  recs_b: Dict[str, Dict[str, float]], eK: str, lK: str
                  ) -> Callable[[List[str]], float]:
    """micro(recs_a) − micro(recs_b) over a sample; nan if either side is empty."""
    def delta(sample):
        ae = sum(recs_a[r][eK] for r in sample); al = sum(recs_a[r][lK] for r in sample)
        be = sum(recs_b[r][eK] for r in sample); bl = sum(recs_b[r][lK] for r in sample)
        return (100 * ae / al) - (100 * be / bl) if al and bl else float("nan")
    return delta


def cluster_boot_paired(recs_a, recs_b, eK, lK, rng):
    """Paired (anchor - cfg) micro-avg delta, cluster-bootstrap by recording.
    Positive => cfg better (lower error). Recordings with zero reference length
    on either side are excluded up front (see :func:`_paired_ids`).

    The per-arm effect estimate + 95% CI in every campaign table; the CI
    excluding 0 is the per-arm significance read (before the family-wise gate)."""
    ids = _paired_ids(recs_a, recs_b, lK)
    return _boot_ci(ids, _paired_delta(recs_a, recs_b, eK, lK), rng)


def cluster_boot_paired_draws(recs_a, recs_b, eK, lK, rng):
    """Like ``cluster_boot_paired`` but ALSO returns the finite bootstrap draws.

    Same point estimate and same resampling — so the Holm/FDR pass can derive a
    bootstrap p-value from the draws (fraction on the wrong side of 0, two-sided)
    WITHOUT re-running the bootstrap. Returns ``(point, draws_array)``;
    ``draws_array`` is empty when no finite draw exists. This is the arm-vs-anchor
    input to the pre-registered Holm gate."""
    ids = _paired_ids(recs_a, recs_b, lK)
    return _boot_resample(ids, _paired_delta(recs_a, recs_b, eK, lK), rng)


def cluster_boot_2key(recs, eKa, lKa, eKb, lKb, rng):
    """Paired Δ = micro(a) − micro(b) of TWO metrics over the SAME recordings,
    cluster-bootstrapped by recording. For the separation-vs-mixture contrast a =
    mixture content floor, b = pipeline content floor (both inside recs[anchor]),
    so + = pipeline lower error = separation recovered content. Same recording
    resample + nan-guard as cluster_boot_paired.

    Backed the assumption-light "separation helps" headline (content recovery,
    attribution removed on both sides) reported per stratum."""
    ids = [r for r in recs if recs[r].get(lKa, 0) > 0 and recs[r].get(lKb, 0) > 0]

    def delta(sample):
        ae = sum(recs[r][eKa] for r in sample); al = sum(recs[r][lKa] for r in sample)
        be = sum(recs[r][eKb] for r in sample); bl = sum(recs[r][lKb] for r in sample)
        return (100 * ae / al) - (100 * be / bl) if al and bl else float("nan")

    return _boot_ci(ids, delta, rng)


def cluster_boot_gap(recs_a, recs_b, eK, lK, gK, gL, rng):
    """Paired Δ of the ATTRIBUTION GAP (cpWER - content floor), cluster-boot
    by recording. gap = micro(cpWER) - micro(content); Δ = gap_anchor - gap_cfg,
    so positive => cfg has the SMALLER gap (better attribution). (eK,lK) = cpWER
    error/length; (gK,gL) = content-floor error/length. Excludes recordings with
    zero reference length on either side (same nan-guard as cluster_boot_paired).

    Backed the attribution-fix arms (the CER attribution gap narrowing 2.9→0.3
    that starred the separation effect in every stratum)."""
    ids = _paired_ids(recs_a, recs_b, lK)

    def gap(recs, sample):
        ce = sum(recs[r][eK] for r in sample); cl = sum(recs[r][lK] for r in sample)
        ge = sum(recs[r][gK] for r in sample); gl = sum(recs[r][gL] for r in sample)
        return (100 * ce / cl - 100 * ge / gl) if cl and gl else float("nan")

    def delta(sample):
        return gap(recs_a, sample) - gap(recs_b, sample)

    return _boot_ci(ids, delta, rng)


def boot_pvalue(draws) -> float:
    """Two-sided bootstrap p-value for H0: paired Δ = 0.

    Standard percentile-bootstrap p: p = 2 * min(frac draws <= 0, frac draws >= 0),
    clipped to [0, 1]. A Δ whose draws sit entirely on one side of 0 gets the
    smallest resolvable p (≈ 2/B_DRAWS, never exactly 0 — the bootstrap cannot
    resolve below its resolution). Empty draws → 1.0 (cannot reject).

    The raw p fed to Holm-Bonferroni / Benjamini-Hochberg for the selection gate."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    n = draws.size
    if n == 0:
        return 1.0
    frac_le = float(np.count_nonzero(draws <= 0)) / n
    frac_ge = float(np.count_nonzero(draws >= 0)) / n
    p = 2.0 * min(frac_le, frac_ge)
    # Floor at the bootstrap resolution so an all-one-side draw isn't reported p=0.
    return float(min(max(p, 1.0 / n), 1.0))


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (family-wise, conservative).

    ``pvals`` is a list of raw p-values; returns adjusted p-values in the SAME
    order. Reject H_i at level α iff adjusted p_i <= α. Monotone by construction
    (cumulative max along the sorted order).

    The pre-registered SELECTION GATE of the campaign (SWEEP_DESIGN §3.4): an arm
    "clears" only when its Holm-adjusted p < 0.05 AND its Δ is favourable."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (less conservative than Holm).

    Same order in / out as ``holm_bonferroni``. Standard step-up with the
    monotone (cumulative-min from the largest) enforcement. Reported ALONGSIDE
    Holm in the campaign tables (FDR context for the family-wise decision), never
    the gate itself."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj


def significance_star(lo: float, hi: float) -> str:
    """`" *"` iff the CI is finite AND excludes 0; `""` otherwise.

    A nan CI is NOT significant — this guards the bug that printed `*` on
    ``nan <= 0 == False``. Used to flag per-arm CIs in the campaign tables."""
    return " *" if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi) else ""


# ``_sig`` is the historical name in ``rescore_stratified.py``; keep an alias so
# the script's format strings need no churn on extraction.
_sig = significance_star


# ---------------------------------------------------------------------------
# Acoustic-complexity strata (recording-level tertiles)
# ---------------------------------------------------------------------------


def _default_recid(fid: str) -> str:
    """Recording id from a fragment id: everything before the first ``__``
    (a multi-segment recording's fragments share one recid)."""
    return fid.split("__")[0]


def recording_means(frag_scores: Dict[str, float],
                    recid: Callable[[str], str] = _default_recid
                    ) -> Dict[str, float]:
    """Collapse a ``{fragment_id: score}`` map to ``{recording_id: mean score}``.

    A recording's score is the mean of its fragments' scores, so a multi-segment
    recording resolves to exactly one value (and, downstream, one stratum). Used
    to turn per-fragment composite acoustic-complexity scores into the per-
    recording means that :func:`assign_strata` tertiles."""
    rec_scores: Dict[str, List[float]] = defaultdict(list)
    for fid, v in frag_scores.items():
        rec_scores[recid(fid)].append(v)
    return {r: sum(v) / len(v) for r, v in rec_scores.items()}


def assign_strata(rec_means: Dict[str, float]) -> Dict[str, str]:
    """Assign LOW/MID/HIGH tertiles over recordings by ascending score.

    ``rec_means`` is ``{recording_id: score}`` (higher = more acoustically
    complex). Recordings are sorted ascending and split into equal thirds
    (``t = n // 3``): the bottom ``t`` → LOW, the next ``t`` → MID, the remainder
    → HIGH (so a non-multiple-of-3 count leaves the extras in HIGH). Assigned at
    the RECORDING level so a multi-segment recording can never leak into two
    strata (which would double-count it). Returns ``{recording_id: stratum}``.

    The stratification used throughout the campaign to show the separation effect
    holds in every complexity tertile."""
    order = sorted(rec_means, key=lambda r: rec_means[r])
    t = len(order) // 3
    strat = {**{r: "LOW" for r in order[:t]},
             **{r: "MID" for r in order[t:2 * t]},
             **{r: "HIGH" for r in order[2 * t:]}}
    assert len(strat) == len(order), "strata must partition the recordings"
    return strat
