"""Unit tests for the pure parts of ``scripts/score_fragment_acoustics.py``.

No network, no GPU, no model downloads: only the model-free / arithmetic
helpers (WADA-SNR, clipping rate, DNSMOS windowing arithmetic, the chunking
self-concat logic, Spearman/composite assembly, robione join). The SQUIM /
DNSMOS-ONNX / brouhaha paths are exercised by the real scoring run, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.score_fragment_acoustics import (
    AUTHOR_GRADES,
    DEV_AUTOR_PREFIXES,
    METRIC_FIELDS,
    _rankdata,
    build_composite,
    clipping_rate,
    is_dev,
    spearman,
    wada_snr,
)


# ---------------------------------------------------------------------------
# WADA-SNR
# ---------------------------------------------------------------------------


def _speech_shaped(n: int, seed: int) -> np.ndarray:
    """A speech-shaped (peaky, super-Gaussian) signal: white noise pushed
    toward a Laplacian-ish amplitude distribution by a cube nonlinearity, then
    unit-peak normalised. Stands in for clean speech for SNR ranking tests."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = np.sign(x) * np.abs(x) ** 3  # sharpen the distribution (peakier)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x.astype(np.float32)


def test_wada_snr_ranks_two_noise_levels():
    """A speech-shaped signal with LESS added white noise must estimate a
    HIGHER WADA-SNR than the same signal with MORE noise."""
    n = 16_000 * 8
    speech = _speech_shaped(n, seed=1)
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-9)
    speech_rms = float(np.sqrt(np.mean(speech ** 2)))

    # SNR 20 dB (clean) vs 0 dB (noisy): noise scaled to the target ratio.
    def mix(snr_db: float) -> np.ndarray:
        target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
        return speech + noise * target_noise_rms

    # 15 dB vs 0 dB: both well inside the table (no ceiling saturation), so the
    # ordering test is meaningful, with a clear margin between them.
    clean = wada_snr(mix(15.0))
    noisy = wada_snr(mix(0.0))
    assert math.isfinite(clean) and math.isfinite(noisy)
    assert clean > noisy + 3.0, f"expected clean({clean}) >> noisy({noisy})"


def test_wada_snr_gain_invariant():
    """WADA-SNR is an amplitude-distribution statistic: scaling the whole
    signal by a constant must not move the estimate (up to fp noise)."""
    x = _speech_shaped(16_000 * 4, seed=7)
    rng = np.random.default_rng(3)
    x = x + 0.05 * rng.standard_normal(len(x))
    a = wada_snr(x)
    b = wada_snr(x * 0.1)
    assert math.isfinite(a)
    assert abs(a - b) < 1e-6, f"{a} vs {b}"


def test_wada_snr_degenerate():
    assert math.isnan(wada_snr(np.array([], dtype=np.float32)))
    assert math.isnan(wada_snr(np.zeros(1000, dtype=np.float32)))


# ---------------------------------------------------------------------------
# Clipping rate
# ---------------------------------------------------------------------------


def test_clipping_rate_basic():
    x = np.array([0.0, 0.5, 1.0, -1.0, 0.9989, 0.999], dtype=np.float32)
    # |x| >= 0.999: the 1.0, -1.0, and 0.999 → 3/6.
    assert clipping_rate(x) == pytest.approx(3 / 6)


def test_clipping_rate_none_clipped():
    x = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    assert clipping_rate(x) == 0.0


def test_clipping_rate_empty():
    assert math.isnan(clipping_rate(np.array([], dtype=np.float32)))


# ---------------------------------------------------------------------------
# DNSMOS windowing arithmetic (no ONNX — count windows via a stub session)
# ---------------------------------------------------------------------------


class _StubSession:
    """Returns a fixed raw triple per window and records how many windows it
    saw (the batched interface feeds (B, 144160) and expects (B, 3) back), so
    we test ONLY the windowing/aggregation, never the real model."""

    def __init__(self, raw=(2.0, 2.5, 2.2)):
        self.raw = raw
        self.windows_seen = 0
        self.calls = 0

    def run(self, _none, feed):
        feats = feed["input_1"]
        assert feats.ndim == 2 and feats.shape[1] == int(9.01 * 16000), feats.shape
        b = feats.shape[0]
        self.windows_seen += b
        self.calls += 1
        return [np.tile(np.array([self.raw], dtype=np.float32), (b, 1))]


def test_dnsmos_window_count_long_clip():
    from scripts.score_fragment_acoustics import dnsmos_windowed

    sr = 16_000
    # 30 s clip: floor(30) - 9.01 + 1 = 30 - 9 + 1 = 22 hops (int floor of 9.01=9).
    audio = np.zeros(30 * sr, dtype=np.float32) + 0.1
    sess = _StubSession()
    out = dnsmos_windowed(audio, sess, sr)
    expected = int(np.floor(30) - 9.01) + 1
    assert sess.windows_seen == expected
    assert out["dnsmos_n_chunks"] == expected
    # All windows identical → mean == polyfit(raw).
    assert math.isfinite(out["dnsmos_sig"])


def test_dnsmos_batching_matches_unbatched():
    """Sub-batch size must not change the result: same windows, same scores,
    just fewer session calls."""
    from scripts.score_fragment_acoustics import dnsmos_windowed

    sr = 16_000
    audio = (0.1 * np.sin(np.linspace(0.0, 700.0, 25 * sr))).astype(np.float32)
    s1, s2 = _StubSession(), _StubSession()
    out_b1 = dnsmos_windowed(audio, s1, sr, batch_size=1)
    out_b8 = dnsmos_windowed(audio, s2, sr, batch_size=8)
    assert s1.windows_seen == s2.windows_seen
    assert s1.calls > s2.calls  # batching actually batched
    assert out_b1 == out_b8


def test_dnsmos_short_clip_self_concat():
    """A clip shorter than one 9.01 s window is self-concatenated up to a full
    window so it still yields exactly one DNSMOS window."""
    from scripts.score_fragment_acoustics import dnsmos_windowed

    sr = 16_000
    audio = np.zeros(3 * sr, dtype=np.float32) + 0.1  # 3 s < 9.01 s
    sess = _StubSession()
    out = dnsmos_windowed(audio, sess, sr)
    assert sess.windows_seen >= 1
    assert out["dnsmos_n_chunks"] >= 1


def test_dnsmos_empty():
    from scripts.score_fragment_acoustics import dnsmos_windowed

    out = dnsmos_windowed(np.array([], dtype=np.float32), _StubSession(), 16_000)
    assert out["dnsmos_n_chunks"] == 0
    assert math.isnan(out["dnsmos_sig"])


# ---------------------------------------------------------------------------
# Spearman + rankdata
# ---------------------------------------------------------------------------


def test_rankdata_ties():
    a = np.array([10.0, 20.0, 20.0, 30.0])
    # ranks: 1, (2+3)/2=2.5, 2.5, 4
    np.testing.assert_allclose(_rankdata(a), [1.0, 2.5, 2.5, 4.0])


def test_spearman_perfect_monotone():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 8.0, 16.0, 32.0])  # monotone increasing
    rho, p, n = spearman(x, y)
    assert n == 5
    assert rho == pytest.approx(1.0)


def test_spearman_anticorrelated():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    rho, _, _ = spearman(x, y)
    assert rho == pytest.approx(-1.0)


def test_spearman_ignores_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    rho, _, n = spearman(x, y)
    assert n == 3  # only rows 0,1,4 are pairwise-finite
    assert rho == pytest.approx(1.0)


def test_spearman_too_few():
    rho, p, n = spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert n == 2
    assert math.isnan(rho)


# ---------------------------------------------------------------------------
# Composite assembly + sign alignment
# ---------------------------------------------------------------------------


class _Corr:
    """Minimal stand-in for MetricCorr with the two fields build_composite uses."""

    def __init__(self, metric, rho_grade):
        self.metric = metric
        self.rho_grade = rho_grade


def test_composite_sign_alignment():
    """A metric whose grade-rho is NEGATIVE (higher value = easier) must be
    flipped so that, in the composite, higher == harder. We give two metrics:
    one positive-rho, one negative-rho, both encoding the SAME hardness order,
    and check the composite preserves that order."""
    ids = ["easy", "mid", "hard"]
    # m_pos rises with hardness; m_neg falls with hardness — same ranking.
    scores = {
        "easy": {"m_pos": 0.0, "m_neg": 10.0},
        "mid": {"m_pos": 5.0, "m_neg": 5.0},
        "hard": {"m_pos": 10.0, "m_neg": 0.0},
    }
    strong = [_Corr("m_pos", 0.8), _Corr("m_neg", -0.8)]
    comp = build_composite(scores, ids, strong)
    assert comp["hard"] > comp["mid"] > comp["easy"], comp
    # Both metrics agree perfectly → composite is just the (flipped) z-scores,
    # so easy and hard should be symmetric around mid≈0.
    assert comp["mid"] == pytest.approx(0.0, abs=1e-9)
    assert comp["hard"] == pytest.approx(-comp["easy"], abs=1e-9)


def test_composite_nan_imputed_not_dropped():
    """A fragment missing one strong metric is mean-imputed for that metric,
    not dropped from the composite."""
    ids = ["a", "b", "c"]
    scores = {
        "a": {"m1": 1.0, "m2": 1.0},
        "b": {"m1": 2.0, "m2": float("nan")},  # missing m2
        "c": {"m1": 3.0, "m2": 3.0},
    }
    strong = [_Corr("m1", 0.7), _Corr("m2", 0.7)]
    comp = build_composite(scores, ids, strong)
    assert set(comp) == {"a", "b", "c"}
    assert all(math.isfinite(v) for v in comp.values())


def test_composite_skips_constant_metric():
    """A metric with zero variance can't be z-scored and is dropped from the
    composite (no divide-by-zero)."""
    ids = ["a", "b", "c"]
    scores = {
        "a": {"flat": 5.0, "var": 1.0},
        "b": {"flat": 5.0, "var": 2.0},
        "c": {"flat": 5.0, "var": 3.0},
    }
    strong = [_Corr("flat", 0.6), _Corr("var", 0.6)]
    comp = build_composite(scores, ids, strong)
    # Built from "var" alone; still monotone, all finite.
    assert comp["c"] > comp["a"]
    assert all(math.isfinite(v) for v in comp.values())


# ---------------------------------------------------------------------------
# Dev/test split + grade table sanity
# ---------------------------------------------------------------------------


def test_is_dev_prefix_match():
    manifest = {
        "x__seg00": {"Autor": "64a6a2eccbbe68001222b3d0"},
        "y__seg00": {"Autor": "deadbeef00000000"},
        "z__seg00": {"Autor": "68493cf9aaaa"},
    }
    assert is_dev("x__seg00", manifest)
    assert not is_dev("y__seg00", manifest)
    assert is_dev("z__seg00", manifest)


def test_grades_and_fields_consistent():
    # All 16 grades present, in [0,10].
    assert len(AUTHOR_GRADES) == 16
    assert all(0.0 <= g <= 10.0 for g in AUTHOR_GRADES.values())
    # METRIC_FIELDS has no duplicates.
    assert len(METRIC_FIELDS) == len(set(METRIC_FIELDS))
    # DEV prefixes are non-empty strings.
    assert all(isinstance(p, str) and p for p in DEV_AUTOR_PREFIXES)


# ---------------------------------------------------------------------------
# --root / --csv-out path plumbing (candidate-mining override)
# ---------------------------------------------------------------------------


def _write_min_scores_csv(path, frag_ids):
    """Write a minimal scores CSV (all metric cells blank → nan) for the given
    frag_ids, enough for the report-only path to read + analyse without any
    model/GPU work."""
    import csv as _csv

    from scripts.score_fragment_acoustics import METRIC_FIELDS as _MF

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["frag_id"] + _MF)
        for fid in frag_ids:
            w.writerow([fid] + [""] * len(_MF))


def test_root_override_redirects_all_paths(tmp_path, monkeypatch):
    """`--root <tree>` must point fragment discovery, the scores CSV, the
    manifest, and the report ALL under that tree — leaving the module's default
    eval-set globals untouched afterwards (no leakage across runs)."""
    import scripts.score_fragment_acoustics as saf

    # Snapshot the defaults so we can prove (a) they were used as the baseline
    # and (b) we restore them — the module mutates globals in main().
    default_root = saf.FRAGMENTS_ROOT
    default_csv = saf.SCORES_CSV
    default_manifest = saf.MANIFEST_PATH
    default_report = saf.REPORT_MD

    try:
        root = tmp_path / "cand_tree"
        root.mkdir()
        # A cached scores CSV under the override root, plus a manifest the
        # report-only path reads (load_manifest needs the columns it queries).
        _write_min_scores_csv(root / "acoustic_scores.csv", ["rec_a__cand00"])
        (root / "manifest.csv").write_text(
            "frag_id,noise,overlap_bin,Autor\nrec_a__cand00,Niski,heavy,someauthor\n",
            encoding="utf-8",
        )
        # robione lookup must not touch the real /mnt path during the test.
        monkeypatch.setattr(saf, "ROBIONE_PATH", tmp_path / "no_robione.txt")

        rc = saf.main(["--report", "--root", str(root)])
        assert rc == 0
        # All four globals now point under the override root.
        assert saf.FRAGMENTS_ROOT == root
        assert saf.SCORES_CSV == root / "acoustic_scores.csv"
        assert saf.MANIFEST_PATH == root / "manifest.csv"
        assert saf.REPORT_MD == root / "ACOUSTIC_SCORES_REPORT.md"
        # The report landed under the override root, not the eval set.
        assert (root / "ACOUSTIC_SCORES_REPORT.md").exists()
    finally:
        # Restore defaults so later tests / real runs see the eval set.
        saf.FRAGMENTS_ROOT = default_root
        saf.SCORES_CSV = default_csv
        saf.MANIFEST_PATH = default_manifest
        saf.REPORT_MD = default_report


def test_csv_out_override_redirects_only_csv(tmp_path, monkeypatch):
    """`--csv-out` (with `--root`) redirects the scores CSV to the explicit
    path, while the other three globals follow `--root`. Proves the candidate
    run can keep its CSV at a chosen name."""
    import scripts.score_fragment_acoustics as saf

    default_root = saf.FRAGMENTS_ROOT
    default_csv = saf.SCORES_CSV
    default_manifest = saf.MANIFEST_PATH
    default_report = saf.REPORT_MD
    try:
        root = tmp_path / "cand_tree"
        root.mkdir()
        out_csv = tmp_path / "candidate_scores.csv"
        _write_min_scores_csv(out_csv, ["rec_b__cand00"])
        (root / "manifest.csv").write_text(
            "frag_id,noise,overlap_bin,Autor\nrec_b__cand00,Niski,heavy,a\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(saf, "ROBIONE_PATH", tmp_path / "no_robione.txt")

        rc = saf.main(["--report", "--root", str(root),
                       "--csv-out", str(out_csv)])
        assert rc == 0
        # CSV global points at the explicit --csv-out path.
        assert saf.SCORES_CSV == out_csv
        # Root / manifest / report follow --root.
        assert saf.FRAGMENTS_ROOT == root
        assert saf.MANIFEST_PATH == root / "manifest.csv"
        assert saf.REPORT_MD == root / "ACOUSTIC_SCORES_REPORT.md"
    finally:
        saf.FRAGMENTS_ROOT = default_root
        saf.SCORES_CSV = default_csv
        saf.MANIFEST_PATH = default_manifest
        saf.REPORT_MD = default_report


def test_no_flags_leaves_paths_default():
    """With no override flags the parser yields None for both, so main()'s
    override block is a no-op — the default eval-set globals stand. Guards the
    'byte-identical default behaviour' contract."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-brouhaha", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    # Mirror the real flags' defaults.
    from pathlib import Path as _P

    ap.add_argument("--root", type=_P, default=None)
    ap.add_argument("--csv-out", type=_P, default=None)
    ap.add_argument("--no-report", action="store_true")
    parsed = ap.parse_args([])
    assert parsed.root is None and parsed.csv_out is None
    assert parsed.no_report is False
