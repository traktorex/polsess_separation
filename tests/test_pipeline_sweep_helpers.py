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
