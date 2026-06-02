"""YAML round-trip tests for the asr_pipeline config."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from asr_pipeline.config import (
    PipelineConfig,
    load_pipeline_config_from_dict,
    load_pipeline_config_from_yaml,
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
    """YAML -> PipelineConfig -> asdict -> YAML -> PipelineConfig is identity."""
    cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))

    out_yaml = tmp_path / "roundtrip.yaml"
    save_pipeline_config_to_yaml(cfg, str(out_yaml))

    cfg_again = load_pipeline_config_from_yaml(str(out_yaml))
    assert asdict(cfg) == asdict(cfg_again)


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
