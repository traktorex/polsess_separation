"""Utility functions and patches."""

from .logger import setup_logger
from .wandb_logger import WandbLogger
from .common import set_seed, setup_warnings, setup_device_and_amp, apply_eps_patch
from .model_utils import unwrap_compiled_model, count_parameters, format_parameter_count
from .config_utils import dataclass_to_dict
from .file_utils import ensure_dir

__all__ = [
    "apply_eps_patch",
    "setup_logger",
    "WandbLogger",
    "set_seed",
    "setup_warnings",
    "setup_device_and_amp",
    "unwrap_compiled_model",
    "count_parameters",
    "format_parameter_count",
    "dataclass_to_dict",
    "ensure_dir",
]
