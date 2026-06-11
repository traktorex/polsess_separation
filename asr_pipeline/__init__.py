"""Multi-stage ASR pipeline (diarization -> SE/SS -> assembly -> Whisper).

Usage-agnostic library: importable from notebooks, callable from the CLI
(`python -m asr_pipeline run --config <yaml> --input <wav>`), and intended
to be liftable into the CLARIN platform with at most one local edit (the
seam to `utils.model_utils.load_model_for_inference` in `stages/separation.py`).
"""

from asr_pipeline.config import (
    PipelineConfig,
    load_pipeline_config_from_yaml,
    load_pipeline_config_from_dict,
    save_pipeline_config_to_yaml,
)
from asr_pipeline.context import PipelineContext
from asr_pipeline.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineContext",
    "load_pipeline_config_from_yaml",
    "load_pipeline_config_from_dict",
    "save_pipeline_config_to_yaml",
]
