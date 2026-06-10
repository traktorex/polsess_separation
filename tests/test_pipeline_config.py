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
    assert cfg.transcription.align_model_name == "jonatasgrosman/wav2vec2-large-xlsr-53-polish"


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
