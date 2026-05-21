"""YAML round-trip tests for the asr_pipeline config."""

from dataclasses import asdict
from pathlib import Path

import yaml

from asr_pipeline.config import (
    PipelineConfig,
    load_pipeline_config_from_dict,
    load_pipeline_config_from_yaml,
    save_pipeline_config_to_yaml,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml"


def test_default_yaml_loads():
    cfg = load_pipeline_config_from_yaml(str(DEFAULT_YAML))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.sample_rate == 16_000
    assert cfg.diarization.num_speakers == 2
    assert cfg.routing.min_overlap_dur == 0.20
    assert cfg.separation.context_window_mode == "snap_to_vad"
    assert cfg.assembly.output_mode == "shortened"
    assert cfg.transcription.model_name == "large-v3"


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
    import pytest

    with pytest.raises(ValueError):
        cfg = PipelineConfig()
        cfg.separation.context_window_mode = "nonsense"
        cfg.__post_init__()


def test_spill_without_artifact_dir_raises():
    import pytest

    with pytest.raises(ValueError):
        cfg = PipelineConfig()
        cfg.spill_intermediate = True
        cfg.artifact_dir = None
        cfg.__post_init__()
