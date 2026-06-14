"""CPU unit tests for the sweep / run-mode drivers' pure logic.

These helpers produce every thesis L3 / config-sweep number, so the silent-
substitution paths the review flagged are pinned here:

- `_apply` must reject a typo'd dotted override path (else it runs the baseline
  under the knob's name — SCOPE §4.2 fail-loud).
- The shared hyp reader (`asr_pipeline.eval.layer3.read_per_speaker`, now used
  by the sweep instead of the deleted `_read_hyp`) keeps a single present
  speaker rather than dropping the mode — the adjudicated fix the script was
  not inheriting before.
- `MODES` appliers set the expected (sep, enh) state; `_fresh_cfg` forces the
  eval overrides L3 depends on.
- `score_configs` micro-averages cpWER over the recordings that have a hyp —
  pinned before any accumulator restructure.

No Pipeline / GPU here (SCOPE §7): only the dataclass-config and file-reading
surface.
"""

from pathlib import Path

import pytest

from asr_pipeline.config import PipelineConfig
from asr_pipeline.eval.layer3 import read_mixture, read_per_speaker
from scripts.run_pipeline_on_recording import MODES, _fresh_cfg
from scripts.sweep_pipeline import (
    CONFIGS,
    _apply,
    _selected_configs,
    bootstrap_microavg_ci,
    bootstrap_paired_diff_ci,
    score_configs,
)


# ---------------------------------------------------------------------------
# _apply — dotted-path overrides must fail loud on a typo
# ---------------------------------------------------------------------------


def test_apply_sets_known_path():
    cfg = _apply(PipelineConfig(), {"separation.vad_threshold": 0.5})
    assert cfg.separation.vad_threshold == 0.5


def test_apply_rejects_unknown_path():
    # `vad_treshold` (typo) would otherwise create a junk attribute, leave
    # vad_threshold at its default, and silently run the baseline.
    with pytest.raises(AttributeError, match="vad_treshold"):
        _apply(PipelineConfig(), {"separation.vad_treshold": 0.5})


def test_apply_rejects_unknown_nested_stage():
    with pytest.raises(AttributeError):
        _apply(PipelineConfig(), {"nonsense.field": 1})


def test_all_configs_apply():
    """Registry-rot net: every CONFIGS entry targets real config attributes.

    Only meaningful once `_apply` rejects unknown paths (test above): a typo'd
    override path in the registry now surfaces here instead of silently running
    the baseline.
    """
    for name, overrides in CONFIGS.items():
        _apply(PipelineConfig(), overrides)   # must not raise


# ---------------------------------------------------------------------------
# MODES appliers + _fresh_cfg overrides
# ---------------------------------------------------------------------------


_EXPECTED_MODE_STATE = {
    # mode -> (separation.enabled, enhancement.enabled, enhancement.backend or None)
    "pipeline":                  (True, True, None),
    "pipeline_nosep":            (False, True, None),
    "pipeline_noenh":            (True, False, None),
    "pipeline_minimal":          (False, False, None),
    "pipeline_nosep_mossformer": (False, True, "mossformer_gan_se_16k"),
}


def test_mode_appliers_set_expected_state():
    for name, applier in MODES:
        cfg = PipelineConfig()
        applier(cfg)
        sep_on, enh_on, backend = _EXPECTED_MODE_STATE[name]
        assert cfg.separation.enabled is sep_on, name
        assert cfg.enhancement.enabled is enh_on, name
        if backend is not None:
            assert cfg.enhancement.backend == backend, name


def test_modes_cover_expected_state_table():
    # Guard against a new mode landing without a state expectation above.
    assert {name for name, _ in MODES} == set(_EXPECTED_MODE_STATE)


def test_fresh_cfg_sets_eval_overrides(tmp_path):
    yaml_path = (
        Path(__file__).resolve().parent.parent
        / "asr_pipeline" / "configs" / "default.yaml"
    )
    cfg = _fresh_cfg(yaml_path)
    assert cfg.transcription.transcribe_mixture is True
    assert cfg.assembly.output_mode == "full_length"
    assert cfg.routing.min_overlap_dur == 0.0


# ---------------------------------------------------------------------------
# shared hyp reader — the un-inherited adjudicated fix
# ---------------------------------------------------------------------------


def _write_transcript(path: Path, start: float, end: float, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[{start:6.2f} → {end:6.2f}]  {text}\n", encoding="utf-8")


def test_read_per_speaker_keeps_single_present_speaker(tmp_path):
    # A one-speaker collapse writes only transcript_A.txt. The shared reader
    # keeps it (charging B as deletions downstream) rather than dropping the
    # whole mode — the optimistic bias the old `_read_hyp` carried.
    _write_transcript(tmp_path / "transcript_A.txt", 0.0, 1.0, "tylko a")
    hyp = read_per_speaker(tmp_path)
    assert set(hyp) == {"A"}
    assert hyp["A"][0].text == "tylko a"


def test_read_per_speaker_none_when_neither_present(tmp_path):
    assert read_per_speaker(tmp_path) is None


def test_read_mixture_present_and_absent(tmp_path):
    assert read_mixture(tmp_path) is None
    _write_transcript(tmp_path / "transcript_mixture.txt", 0.0, 1.0, "mix")
    assert read_mixture(tmp_path)[0].text == "mix"


# ---------------------------------------------------------------------------
# _selected_configs — dedup, baseline-first
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, configs=None, groups=None):
        self.configs = configs
        self.groups = groups


def test_selected_configs_baseline_first_and_deduped():
    names = _selected_configs(_Args(configs=["enh_frcrn", "baseline", "enh_frcrn"]))
    assert names[0] == "baseline"
    assert names.count("enh_frcrn") == 1
    assert names.count("baseline") == 1


def test_selected_configs_default_excludes_baseline_dupe():
    names = _selected_configs(_Args())
    assert names[0] == "baseline"
    assert names.count("baseline") == 1
    assert set(names) == set(CONFIGS)


# ---------------------------------------------------------------------------
# score_configs — micro-average over the recordings that have a hyp
# ---------------------------------------------------------------------------


def _make_rec(root: Path, fid: str, ref_text: str, hyp_text: str,
              config_name: str) -> None:
    rec = root / fid
    (rec / "reference").mkdir(parents=True)
    (rec / f"{fid}.wav").touch()
    _write_transcript(rec / "reference" / "speaker_A.txt", 0.0, 1.0, ref_text)
    _write_transcript(rec / "reference" / "speaker_B.txt", 2.0, 3.0, "b mowi")
    _write_transcript(rec / "sweep" / config_name / "transcript_A.txt",
                      0.0, 1.0, hyp_text)
    _write_transcript(rec / "sweep" / config_name / "transcript_B.txt",
                      2.0, 3.0, "b mowi")


def test_score_configs_microaverage_over_present_recordings(tmp_path):
    # Two recordings: rec1 hyp == ref (0 errors), rec2 hyp drops one word
    # (1 sub). Micro-average cpWER = sum(errors)/sum(ref_len) over both.
    _make_rec(tmp_path, "rec1", "ala ma kota", "ala ma kota", "baseline")
    _make_rec(tmp_path, "rec2", "ala ma kota", "ala ma psa", "baseline")

    df = score_configs(["baseline"], tmp_path, ["rec1", "rec2"])
    row = df[df["config"] == "baseline"].iloc[0]
    assert int(row["n"]) == 2
    # ref length per rec = 5 tokens (3 A + 2 B); 1 substitution across 10 tokens.
    assert row["cpWER"] == pytest.approx(10.0)
    # MIMO-WER on the same per-speaker hyp is speaker-agnostic; with no
    # attribution error here it matches cpWER (1 sub / 10 ref tokens).
    assert row["mimoWER"] == pytest.approx(10.0)


def test_score_configs_skips_recording_without_hyp(tmp_path):
    _make_rec(tmp_path, "rec1", "ala ma kota", "ala ma kota", "baseline")
    # rec2 has GT but no sweep output for this config.
    rec2 = tmp_path / "rec2"
    (rec2 / "reference").mkdir(parents=True)
    (rec2 / "rec2.wav").touch()
    _write_transcript(rec2 / "reference" / "speaker_A.txt", 0.0, 1.0, "cokolwiek")

    df = score_configs(["baseline"], tmp_path, ["rec1", "rec2"])
    row = df[df["config"] == "baseline"].iloc[0]
    assert int(row["n"]) == 1     # only rec1 scored


# ---------------------------------------------------------------------------
# bootstrap CI over fragments — pure, deterministic at the fixed seed
# ---------------------------------------------------------------------------
#
# Under deterministic=True the pipeline is ~zero-variance on re-run, so the
# uncertainty these capture is DEV-SET SAMPLING uncertainty across the
# fragments — resample fragments with replacement, recompute the micro-average.


def _micro(frag_counts):
    """The point estimate the bootstrap brackets: 100 * Σerr / Σref."""
    err = sum(e for e, _ in frag_counts)
    ref = sum(w for _, w in frag_counts)
    return 100.0 * err / ref


def test_bootstrap_ci_brackets_point_estimate():
    # Heterogeneous fragments so resampling actually spreads the statistic.
    frag = [(2.0, 10.0), (0.0, 8.0), (5.0, 12.0), (1.0, 9.0), (3.0, 11.0)]
    lo, hi = bootstrap_microavg_ci(frag)
    point = _micro(frag)            # 22.0%
    assert lo <= point <= hi
    assert lo < hi                  # a non-degenerate interval


def test_bootstrap_ci_deterministic_for_fixed_seed():
    # Many heterogeneous fragments so the bootstrap distribution is rich enough
    # that different seeds land on different percentile values.
    frag = [(float(i % 4), 8.0 + i) for i in range(12)]
    assert bootstrap_microavg_ci(frag, seed=1234) == bootstrap_microavg_ci(frag, seed=1234)
    # A different seed should move the interval.
    assert bootstrap_microavg_ci(frag, seed=1234) != bootstrap_microavg_ci(frag, seed=99)


def test_bootstrap_ci_uses_microaverage_not_per_fragment_mean():
    # One large, low-error fragment + one tiny, high-error fragment. The micro
    # average is dominated by the large fragment; a per-fragment mean of rates
    # would sit much higher. The CI must bracket the micro-average.
    frag = [(1.0, 100.0), (2.0, 2.0)]        # micro = 3/102 ≈ 2.94%
    lo, hi = bootstrap_microavg_ci(frag)
    point = _micro(frag)
    assert lo <= point <= hi
    assert point < 50.0                       # per-fragment mean would be ~50%


def test_bootstrap_ci_single_fragment_is_degenerate():
    # One fragment carries no sampling spread: every resample is that fragment.
    lo, hi = bootstrap_microavg_ci([(3.0, 10.0)])
    assert lo == hi == pytest.approx(30.0)


def test_bootstrap_ci_empty_is_nan():
    import math
    lo, hi = bootstrap_microavg_ci([])
    assert math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------------------
# paired bootstrap vs baseline — pure, deterministic
# ---------------------------------------------------------------------------


def test_paired_diff_identical_configs_ci_contains_zero_and_not_sig():
    # Same per-fragment counts on both sides → delta exactly 0, CI ≡ {0},
    # not significant.
    frag = [(2.0, 10.0), (0.0, 8.0), (5.0, 12.0), (1.0, 9.0)]
    delta, lo, hi, sig = bootstrap_paired_diff_ci(frag, list(frag))
    assert delta == pytest.approx(0.0)
    assert lo <= 0.0 <= hi
    assert sig is False


def test_paired_diff_clear_separation_is_significant():
    # Config strictly worse on every fragment by a wide, consistent margin →
    # the paired CI should exclude 0 (sig True) and delta should be positive.
    base = [(1.0, 100.0), (2.0, 100.0), (0.0, 100.0), (1.0, 100.0), (3.0, 100.0)]
    cfg = [(40.0, 100.0), (45.0, 100.0), (38.0, 100.0), (42.0, 100.0), (50.0, 100.0)]
    delta, lo, hi, sig = bootstrap_paired_diff_ci(cfg, base)
    assert delta > 0.0
    assert lo > 0.0          # whole CI above zero
    assert sig is True


def test_paired_diff_delta_matches_microaverage_difference():
    base = [(2.0, 10.0), (1.0, 10.0)]        # micro 3/20 = 15%
    cfg = [(4.0, 10.0), (2.0, 10.0)]         # micro 6/20 = 30%
    delta, _lo, _hi, _sig = bootstrap_paired_diff_ci(cfg, base)
    assert delta == pytest.approx(15.0)


def test_paired_diff_deterministic_for_fixed_seed():
    base = [(2.0, 10.0), (1.0, 8.0), (3.0, 12.0)]
    cfg = [(3.0, 10.0), (2.0, 8.0), (1.0, 12.0)]
    a = bootstrap_paired_diff_ci(cfg, base, seed=1234)
    b = bootstrap_paired_diff_ci(cfg, base, seed=1234)
    assert a == b


def test_paired_diff_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_paired_diff_ci([(1.0, 10.0)], [(1.0, 10.0), (2.0, 10.0)])


# ---------------------------------------------------------------------------
# score_configs — the new additive columns wire through end to end
# ---------------------------------------------------------------------------


def _add_sweep_output(root: Path, fid: str, hyp_text: str, config_name: str) -> None:
    """Add one more config's per-speaker hyp to an existing rec dir."""
    sweep = root / fid / "sweep" / config_name
    _write_transcript(sweep / "transcript_A.txt", 0.0, 1.0, hyp_text)
    _write_transcript(sweep / "transcript_B.txt", 2.0, 3.0, "b mowi")


def test_score_configs_emits_ci_and_paired_columns(tmp_path):
    _make_rec(tmp_path, "rec1", "ala ma kota", "ala ma kota", "baseline")
    _make_rec(tmp_path, "rec2", "ala ma kota", "ala ma psa", "baseline")
    _add_sweep_output(tmp_path, "rec1", "ala ma kota", "cfgX")
    _add_sweep_output(tmp_path, "rec2", "ala ma psa", "cfgX")

    df = score_configs(["baseline", "cfgX"], tmp_path, ["rec1", "rec2"])
    for col in ("mimoWER", "CER", "cpwer_ci_lo", "cpwer_ci_hi", "secs_per_frag",
                "vs_base_delta", "vs_base_ci_lo", "vs_base_ci_hi", "sig"):
        assert col in df.columns

    base = df[df["config"] == "baseline"].iloc[0]
    # CI brackets the point estimate.
    assert base["cpwer_ci_lo"] <= base["cpWER"] <= base["cpwer_ci_hi"]
    # baseline's own paired-vs-baseline columns stay blank.
    import math
    assert math.isnan(base["vs_base_delta"])
    assert base["sig"] is None

    cfgx = df[df["config"] == "cfgX"].iloc[0]
    # Identical outputs to baseline → paired delta 0, not significant.
    assert cfgx["vs_base_delta"] == pytest.approx(0.0)
    assert cfgx["sig"] is False
    # No run_meta.json written by the test fixtures → blank secs_per_frag.
    assert math.isnan(cfgx["secs_per_frag"])


def test_score_configs_secs_per_frag_reads_run_meta(tmp_path):
    import json as _json
    _make_rec(tmp_path, "rec1", "ala ma kota", "ala ma kota", "baseline")
    _make_rec(tmp_path, "rec2", "ala ma kota", "ala ma psa", "baseline")
    (tmp_path / "rec1" / "sweep" / "baseline" / "run_meta.json").write_text(
        _json.dumps({"seconds": 10.0}), encoding="utf-8")
    (tmp_path / "rec2" / "sweep" / "baseline" / "run_meta.json").write_text(
        _json.dumps({"seconds": 20.0}), encoding="utf-8")

    df = score_configs(["baseline"], tmp_path, ["rec1", "rec2"])
    row = df[df["config"] == "baseline"].iloc[0]
    assert row["secs_per_frag"] == pytest.approx(15.0)   # mean(10, 20)


def test_score_configs_paired_skipped_without_baseline(tmp_path, capsys):
    _make_rec(tmp_path, "rec1", "ala ma kota", "ala ma kota", "cfgX")
    _make_rec(tmp_path, "rec2", "ala ma kota", "ala ma psa", "cfgX")

    df = score_configs(["cfgX"], tmp_path, ["rec1", "rec2"])
    out = capsys.readouterr().out
    assert "baseline" in out and "skipping" in out
    import math
    cfgx = df[df["config"] == "cfgX"].iloc[0]
    assert math.isnan(cfgx["vs_base_delta"])
    assert cfgx["sig"] is None
