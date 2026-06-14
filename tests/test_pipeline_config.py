"""YAML round-trip tests for the asr_pipeline config."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from asr_pipeline.config import (
    PipelineConfig,
    load_pipeline_config_from_dict,
    load_pipeline_config_from_yaml,
    redact_config_snapshot,
    save_pipeline_config_to_yaml,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml"


@pytest.fixture(autouse=True)
def _fake_hf_token(monkeypatch):
    """Provide HF_TOKEN so config validation doesn't reject diarization.

    `PipelineConfig.__post_init__` requires hf_token when
    diarization.enabled is True. The dataclass default reads $HF_TOKEN;
    on a fresh CI/checkout that env var isn't set, so we patch one in
    for the duration of each test.
    """
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")


def test_default_yaml_loads():
    cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.sample_rate == 16_000
    assert cfg.diarization.num_speakers == 2
    assert cfg.routing.min_overlap_dur == 0.20
    assert cfg.separation.context_window_mode == "expand_to_chunk"
    assert cfg.post_separation_processing.backend == "ap_bwe"
    assert cfg.assembly.output_mode == "shortened"
    assert cfg.transcription.backend == "whisperx"
    assert cfg.transcription.model_name == "large-v2"
    # default.yaml no longer pins the aligner; None → WhisperX picks the
    # per-language default (pl → the previously-pinned jonatasgrosman model).
    assert cfg.transcription.align_model_name is None


def test_yaml_roundtrip(tmp_path):
    """YAML -> PipelineConfig -> YAML -> PipelineConfig is identity. The saver
    masks hf_token to the _REDACTED placeholder on disk; the loader drops that
    placeholder and re-resolves $HF_TOKEN, so under the fixture's token the
    round-trip is exact — never the literal string 'REDACTED'."""
    cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))

    out_yaml = tmp_path / "roundtrip.yaml"
    save_pipeline_config_to_yaml(cfg, str(out_yaml))

    cfg_again = load_pipeline_config_from_yaml(str(out_yaml))
    assert asdict(cfg) == asdict(cfg_again)
    assert cfg_again.diarization.hf_token == "test-hf-token"


def test_redacted_token_does_not_survive_reload_without_env(tmp_path, monkeypatch):
    """The dangerous leg: a saved (redacted) config reloaded with no $HF_TOKEN
    must fail loud, not silently hand pyannote the literal 'REDACTED'."""
    cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))
    out_yaml = tmp_path / "roundtrip.yaml"
    save_pipeline_config_to_yaml(cfg, str(out_yaml))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="hf_token"):
        load_pipeline_config_from_yaml(str(out_yaml))


def test_saved_yaml_never_contains_token(tmp_path):
    """A live hf_token must not appear in a saved config file."""
    cfg = PipelineConfig()
    cfg.diarization.hf_token = "hf_live_secret_value"
    out_yaml = tmp_path / "cfg.yaml"
    save_pipeline_config_to_yaml(cfg, str(out_yaml))
    text = out_yaml.read_text()
    assert "hf_live_secret_value" not in text
    assert "REDACTED" in text
    # The in-memory config is untouched.
    assert cfg.diarization.hf_token == "hf_live_secret_value"


def test_redact_does_not_mutate_input():
    """The redactor deep-copies — a caller's live config must be untouched."""
    src = {"diarization": {"hf_token": "LIVE", "num_speakers": 2}}
    out = redact_config_snapshot(src)
    assert src["diarization"]["hf_token"] == "LIVE"       # caller untouched
    assert out["diarization"]["hf_token"] == "REDACTED"


def test_deterministic_defaults_true():
    """Reproducible-by-default: both the dataclass and shipped YAML enable it."""
    assert PipelineConfig().deterministic is True
    assert load_pipeline_config_from_yaml(str(DEFAULT_YAML)).deterministic is True


def test_dict_roundtrip():
    """asdict -> from_dict round-trips."""
    cfg = PipelineConfig()
    cfg_again = load_pipeline_config_from_dict(asdict(cfg))
    assert asdict(cfg) == asdict(cfg_again)


def test_invalid_enum_raises():
    """Bad enum-string values are rejected by __post_init__."""
    with pytest.raises(ValueError):
        cfg = PipelineConfig()
        cfg.separation.context_window_mode = "nonsense"
        cfg.__post_init__()


@pytest.mark.parametrize("section,field,name", [
    ("separation", "seam_mode", "seam_mode"),
    ("separation", "volume_normalization", "volume_normalization"),
    ("assembly", "output_mode", "output_mode"),
    ("enhancement", "backend", "enhancement.backend"),
    ("post_separation_processing", "backend", "post_separation_processing.backend"),
    ("transcription", "backend", "transcription.backend"),
])
def test_each_enum_guard_rejects_bad_value(section, field, name):
    """Every enum-string guard in __post_init__ (not just context_window_mode)
    rejects an out-of-set value, naming the offending knob in the message — a
    YAML typo on any of them fails loud at config time."""
    cfg = PipelineConfig()
    setattr(getattr(cfg, section), field, "definitely_not_valid")
    with pytest.raises(ValueError, match=name):
        cfg.__post_init__()


def test_spill_without_artifact_dir_raises():
    with pytest.raises(ValueError):
        cfg = PipelineConfig()
        cfg.spill_intermediate = True
        cfg.artifact_dir = None
        cfg.__post_init__()


def test_missing_hf_token_raises(monkeypatch):
    """Diarization enabled + no hf_token fails at config-load time."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="hf_token"):
        cfg = PipelineConfig()
        cfg.diarization.hf_token = None
        cfg.__post_init__()


def test_default_yaml_separation_matches_dataclass():
    """Pin: dataclass defaults and default.yaml must name the same separator
    setup. `load_pipeline_config_from_dict` falls back to the dataclass when
    the `separation:` block is absent, so any divergence means programmatic
    callers silently run a different separator/seam than YAML users."""
    yaml_cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML)).separation
    dc_cfg = PipelineConfig().separation
    assert yaml_cfg.checkpoint_path == dc_cfg.checkpoint_path
    assert yaml_cfg.seam_mode == dc_cfg.seam_mode
    assert yaml_cfg.vad_threshold == dc_cfg.vad_threshold
    assert yaml_cfg.vad_soft_threshold == dc_cfg.vad_soft_threshold


@pytest.mark.parametrize("field,value", [
    ("training_chunk_length_s", 0.0),
    ("training_chunk_length_s", -1.0),
    ("overlap_add_threshold_s", 0.0),
    ("vad_soft_threshold", -0.1),
])
def test_invalid_separation_numbers_raise(field, value):
    """YAML typos that would hang (hop=0 infinite loop) or flood the VAD
    mask must fail at config time, not at run time."""
    with pytest.raises(ValueError, match=field):
        cfg = PipelineConfig()
        setattr(cfg.separation, field, value)
        cfg.__post_init__()


def test_flowhigh_input_sr_must_be_positive():
    """The fourth numeric guard (separate from the separation.* block above):
    a non-positive FlowHigh input rate fails at config time, not deep inside
    the post-separation backend."""
    with pytest.raises(ValueError, match="flowhigh_input_sr"):
        cfg = PipelineConfig()
        cfg.post_separation_processing.flowhigh_input_sr = 0
        cfg.__post_init__()


# ---------------------------------------------------------------------------
# Observation Adding (OA) / dry-wet mix (enhancement.observation_mix_ratio)
# ---------------------------------------------------------------------------


def test_observation_mix_ratio_default_is_zero():
    """OA is off by default in both the dataclass and the shipped YAML, so a
    baseline run stays pure-enhanced (current behaviour)."""
    assert PipelineConfig().enhancement.observation_mix_ratio == 0.0
    yaml_cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))
    assert yaml_cfg.enhancement.observation_mix_ratio == 0.0


def test_observation_mix_ratio_loads_from_dict():
    """The knob round-trips through the nested-dict loader."""
    cfg = load_pipeline_config_from_dict(
        {"enhancement": {"observation_mix_ratio": 0.4}}
    )
    assert cfg.enhancement.observation_mix_ratio == 0.4


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_observation_mix_ratio_out_of_range_rejected(value):
    """Outside the convex [0, 1] range (or non-finite) fails at config time,
    naming the knob — a YAML typo can't silently scale the dry/wet mix."""
    with pytest.raises(ValueError, match="observation_mix_ratio"):
        cfg = PipelineConfig()
        cfg.enhancement.observation_mix_ratio = value
        cfg.__post_init__()


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_observation_mix_ratio_valid_values_accepted(value):
    """The edges 0.0 (pure enhanced) and 1.0 (pure observed) plus an interior
    value are all accepted."""
    cfg = PipelineConfig()
    cfg.enhancement.observation_mix_ratio = value
    cfg.__post_init__()  # must not raise
    assert cfg.enhancement.observation_mix_ratio == value


# ---------------------------------------------------------------------------
# Per-language alignment model (SCOPE §9 "Not Polish-only")
# ---------------------------------------------------------------------------

ENGLISH_YAML = REPO_ROOT / "asr_pipeline" / "configs" / "english.yaml"


def test_align_model_name_defaults_to_none():
    """The aligner is no longer pinned in the dataclass: None lets WhisperX
    pick its per-language default (pl → the previous jonatasgrosman pin)."""
    assert PipelineConfig().transcription.align_model_name is None


def test_align_model_name_yaml_null_round_trips_to_none():
    """`align_model_name: null` in YAML must load as Python None (not the
    string 'null') so WhisperX's per-language default kicks in."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {"align_model_name": None}}
    )
    assert cfg.transcription.align_model_name is None


def test_align_model_name_explicit_override():
    """An explicit aligner id round-trips unchanged — the override path the
    English preset relies on."""
    name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    cfg = load_pipeline_config_from_dict(
        {"transcription": {"align_model_name": name}}
    )
    assert cfg.transcription.align_model_name == name


def test_english_preset_loads_with_auto_aligner():
    """The shipped English preset selects English language and leaves the
    aligner to WhisperX's per-language default (author ruling 2026-06-11)."""
    cfg = load_pipeline_config_from_yaml(str(ENGLISH_YAML))
    assert cfg.transcription.language == "en"
    assert cfg.transcription.align_model_name is None


# ---------------------------------------------------------------------------
# Precedence + unknown-key rejection (C2)
# ---------------------------------------------------------------------------


def test_yaml_hf_token_overrides_env(monkeypatch):
    """env < YAML for a default_factory field: hf_token defaults to $HF_TOKEN,
    but an explicit YAML value must win. The fixture sets $HF_TOKEN; a config
    dict that names hf_token explicitly takes precedence."""
    monkeypatch.setenv("HF_TOKEN", "from-env")
    cfg = load_pipeline_config_from_dict(
        {"diarization": {"hf_token": "from-yaml"}}
    )
    assert cfg.diarization.hf_token == "from-yaml"


def test_env_used_when_yaml_omits_hf_token(monkeypatch):
    """The other precedence leg: when the dict doesn't name hf_token, the
    default_factory falls through to $HF_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "from-env")
    cfg = load_pipeline_config_from_dict({"diarization": {"num_speakers": 2}})
    assert cfg.diarization.hf_token == "from-env"


def test_unknown_top_level_key_rejected():
    """An unknown top-level key (a YAML typo at the root) raises — it would
    otherwise be silently ignored and the intended setting left at default."""
    with pytest.raises(TypeError):
        load_pipeline_config_from_dict({"sampel_rate": 8000})


def test_unknown_stage_level_key_rejected():
    """An unknown key inside a stage block raises too (the cls(**sub_dict)
    splat is strict)."""
    with pytest.raises(TypeError):
        load_pipeline_config_from_dict({"separation": {"vad_treshold": 0.5}})


# ---------------------------------------------------------------------------
# Transcription decode knobs (sweepable Whisper hyperparameters)
# ---------------------------------------------------------------------------


def test_transcription_decode_defaults_match_whisperx():
    """Defaults reproduce WhisperX's own default_asr_options exactly, so a
    baseline run with default config is byte-identical to before the knobs
    existed. Evidence: whisperx/asr.py load_model's default_asr_options."""
    t = PipelineConfig().transcription
    assert t.beam_size == 5
    assert t.temperature == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # WhisperX overrides faster-whisper's signature default of True here.
    assert t.condition_on_previous_text is False
    assert t.no_speech_threshold == 0.6
    assert t.compression_ratio_threshold == 2.4
    assert t.patience == 1.0


def test_default_yaml_decode_knobs_match_dataclass():
    """Pin: default.yaml ships the same decode knobs as the dataclass, so YAML
    users and programmatic callers decode identically."""
    y = load_pipeline_config_from_yaml(str(DEFAULT_YAML)).transcription
    d = PipelineConfig().transcription
    assert y.beam_size == d.beam_size
    assert y.temperature == d.temperature
    assert y.condition_on_previous_text == d.condition_on_previous_text
    assert y.no_speech_threshold == d.no_speech_threshold
    assert y.compression_ratio_threshold == d.compression_ratio_threshold
    assert y.patience == d.patience


def test_decode_knobs_load_from_yaml_dict():
    """All six knobs load from a config dict (the sweep path overrides them)."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {
            "beam_size": 1,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "no_speech_threshold": 0.3,
            "compression_ratio_threshold": 3.0,
            "patience": 2.0,
        }}
    )
    t = cfg.transcription
    assert t.beam_size == 1
    assert t.temperature == 0.0          # scalar accepted (no-fallback decode)
    assert t.condition_on_previous_text is True
    assert t.no_speech_threshold == 0.3
    assert t.compression_ratio_threshold == 3.0
    assert t.patience == 2.0


def test_temperature_schedule_list_round_trips(tmp_path):
    """A list temperature survives save→load (the tuple default would have
    serialised as !!python/tuple and broken safe_load — guard against a
    regression to a tuple default)."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {"temperature": [0.0, 0.5, 1.0]}}
    )
    out = tmp_path / "c.yaml"
    save_pipeline_config_to_yaml(cfg, str(out))
    assert "!!python/tuple" not in out.read_text()
    again = load_pipeline_config_from_yaml(str(out))
    assert again.transcription.temperature == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("field,value,token", [
    ("beam_size", 0, "beam_size"),
    ("beam_size", -1, "beam_size"),
    ("patience", 0.0, "patience"),
    ("patience", -1.0, "patience"),
    ("patience", float("inf"), "patience"),
    ("no_speech_threshold", float("nan"), "no_speech_threshold"),
    ("compression_ratio_threshold", float("inf"), "compression_ratio_threshold"),
    ("temperature", 1.5, "temperature"),
    ("temperature", -0.1, "temperature"),
    ("temperature", [0.0, 2.0], "temperature"),
    ("temperature", [], "temperature"),
    ("chunk_size", 0, "chunk_size"),
    ("chunk_size", -5, "chunk_size"),
])
def test_invalid_decode_knobs_raise(field, value, token):
    """Out-of-range decode knobs fail loud at config time, naming the offending
    knob — a sweep YAML typo can't silently produce a degenerate decode."""
    with pytest.raises(ValueError, match=token):
        cfg = PipelineConfig()
        setattr(cfg.transcription, field, value)
        cfg.__post_init__()


@pytest.mark.parametrize("field,value", [
    ("beam_size", 1),
    ("patience", 0.5),
    ("temperature", 0.0),
    ("temperature", 1.0),
    ("temperature", [0.0]),
    ("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ("no_speech_threshold", 0.0),
    ("compression_ratio_threshold", 2.4),
    ("chunk_size", 1),
])
def test_valid_decode_knob_edges_accepted(field, value):
    """Boundary-valid values are accepted (beam_size=1, temperature in {0,1},
    single-element schedule)."""
    cfg = PipelineConfig()
    setattr(cfg.transcription, field, value)
    cfg.__post_init__()      # must not raise


# ---------------------------------------------------------------------------
# Anti-hallucination decode knobs (faster-whisper / WhisperX only)
# ---------------------------------------------------------------------------


def test_antihallucination_defaults_match_faster_whisper():
    """Defaults equal faster-whisper's signature defaults, which are exactly
    what WhisperX's default_asr_options also carry — so a baseline run is
    byte-identical. Evidence (pinned venv 2026-06-14): faster-whisper
    WhisperModel.transcribe → no_repeat_ngram_size=0, repetition_penalty=1,
    hallucination_silence_threshold=None; whisperx/asr.py default_asr_options
    carries the same three values."""
    t = PipelineConfig().transcription
    assert t.no_repeat_ngram_size == 0
    assert t.repetition_penalty == 1.0
    assert t.hallucination_silence_threshold is None


def test_chunk_size_default_and_yaml():
    """chunk_size defaults to 30 (= WhisperX default → byte-identical baseline);
    default.yaml ships the same value."""
    assert PipelineConfig().transcription.chunk_size == 30
    assert load_pipeline_config_from_yaml(str(DEFAULT_YAML)).transcription.chunk_size == 30


def test_default_yaml_antihallucination_knobs_match_dataclass():
    """Pin: default.yaml ships the same anti-hallucination knobs as the
    dataclass, so YAML users and programmatic callers decode identically."""
    y = load_pipeline_config_from_yaml(str(DEFAULT_YAML)).transcription
    d = PipelineConfig().transcription
    assert y.no_repeat_ngram_size == d.no_repeat_ngram_size
    assert y.repetition_penalty == d.repetition_penalty
    assert y.hallucination_silence_threshold == d.hallucination_silence_threshold


def test_antihallucination_knobs_load_from_yaml_dict():
    """The three knobs load from a config dict (the sweep path overrides
    them)."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "hallucination_silence_threshold": 2.0,
        }}
    )
    t = cfg.transcription
    assert t.no_repeat_ngram_size == 3
    assert t.repetition_penalty == 1.2
    assert t.hallucination_silence_threshold == 2.0


def test_antihallucination_yaml_null_threshold_round_trips_to_none():
    """`hallucination_silence_threshold: null` loads as Python None (off),
    not the string 'null'."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {"hallucination_silence_threshold": None}}
    )
    assert cfg.transcription.hallucination_silence_threshold is None


@pytest.mark.parametrize("field,value,token", [
    ("no_repeat_ngram_size", -1, "no_repeat_ngram_size"),
    ("repetition_penalty", 0.0, "repetition_penalty"),
    ("repetition_penalty", -1.0, "repetition_penalty"),
    ("repetition_penalty", float("inf"), "repetition_penalty"),
    ("repetition_penalty", float("nan"), "repetition_penalty"),
    ("hallucination_silence_threshold", 0.0, "hallucination_silence_threshold"),
    ("hallucination_silence_threshold", -1.0, "hallucination_silence_threshold"),
    ("hallucination_silence_threshold", float("inf"), "hallucination_silence_threshold"),
    ("hallucination_silence_threshold", float("nan"), "hallucination_silence_threshold"),
])
def test_invalid_antihallucination_knobs_raise(field, value, token):
    """Out-of-range anti-hallucination knobs fail loud at config time, naming
    the offending knob."""
    with pytest.raises(ValueError, match=token):
        cfg = PipelineConfig()
        setattr(cfg.transcription, field, value)
        cfg.__post_init__()


@pytest.mark.parametrize("field,value", [
    ("no_repeat_ngram_size", 0),       # default / disabled
    ("no_repeat_ngram_size", 2),
    ("repetition_penalty", 1.0),       # default / no penalty
    ("repetition_penalty", 1.5),
    ("hallucination_silence_threshold", None),   # default / off
    ("hallucination_silence_threshold", 0.5),
])
def test_valid_antihallucination_knob_edges_accepted(field, value):
    """Boundary-valid anti-hallucination values are accepted (the defaults
    plus a representative enabled value for each)."""
    cfg = PipelineConfig()
    setattr(cfg.transcription, field, value)
    cfg.__post_init__()      # must not raise


# ---------------------------------------------------------------------------
# Detect-and-retry for collapsed WhisperX windows (WhisperX-only)
# ---------------------------------------------------------------------------


def test_retry_collapsed_defaults():
    """Retry ships ON: re-chunk 8 s, collapse filter 18 s / 0.7 w/s."""
    t = PipelineConfig().transcription
    assert t.retry_collapsed_chunk_size == 8
    assert t.collapse_min_duration_s == 18.0
    assert t.collapse_max_wps == 0.7


def test_retry_collapsed_default_yaml_matches_dataclass():
    """Pin: default.yaml ships the same retry knobs as the dataclass, so YAML
    users and programmatic callers get the same collapse-recovery behaviour."""
    y = load_pipeline_config_from_yaml(str(DEFAULT_YAML)).transcription
    d = PipelineConfig().transcription
    assert y.retry_collapsed_chunk_size == d.retry_collapsed_chunk_size
    assert y.collapse_min_duration_s == d.collapse_min_duration_s
    assert y.collapse_max_wps == d.collapse_max_wps


def test_retry_collapsed_knobs_load_from_yaml_dict():
    """The three retry knobs load from a config dict (the sweep override path)."""
    cfg = load_pipeline_config_from_dict(
        {"transcription": {
            "retry_collapsed_chunk_size": 0,    # disabled
            "collapse_min_duration_s": 20.0,
            "collapse_max_wps": 1.0,
        }}
    )
    t = cfg.transcription
    assert t.retry_collapsed_chunk_size == 0
    assert t.collapse_min_duration_s == 20.0
    assert t.collapse_max_wps == 1.0


@pytest.mark.parametrize("field,value,token", [
    ("retry_collapsed_chunk_size", -1, "retry_collapsed_chunk_size"),
    ("collapse_min_duration_s", 0.0, "collapse_min_duration_s"),
    ("collapse_min_duration_s", -1.0, "collapse_min_duration_s"),
    ("collapse_min_duration_s", float("inf"), "collapse_min_duration_s"),
    ("collapse_min_duration_s", float("nan"), "collapse_min_duration_s"),
    ("collapse_max_wps", 0.0, "collapse_max_wps"),
    ("collapse_max_wps", -1.0, "collapse_max_wps"),
    ("collapse_max_wps", float("inf"), "collapse_max_wps"),
    ("collapse_max_wps", float("nan"), "collapse_max_wps"),
])
def test_invalid_retry_collapsed_knobs_raise(field, value, token):
    """Out-of-range retry knobs fail loud at config time, naming the offending
    knob — a sweep YAML typo can't silently disable or break the retry."""
    with pytest.raises(ValueError, match=token):
        cfg = PipelineConfig()
        setattr(cfg.transcription, field, value)
        cfg.__post_init__()


@pytest.mark.parametrize("field,value", [
    ("retry_collapsed_chunk_size", 0),    # disabled
    ("retry_collapsed_chunk_size", 1),    # min enabled
    ("retry_collapsed_chunk_size", 8),    # default
    ("collapse_min_duration_s", 0.1),     # any positive
    ("collapse_min_duration_s", 18.0),    # default
    ("collapse_max_wps", 0.1),            # any positive
    ("collapse_max_wps", 0.7),            # default
])
def test_valid_retry_collapsed_knob_edges_accepted(field, value):
    """Boundary-valid retry values are accepted (disabled, min-enabled, default,
    and a small positive for each float filter)."""
    cfg = PipelineConfig()
    setattr(cfg.transcription, field, value)
    cfg.__post_init__()      # must not raise
