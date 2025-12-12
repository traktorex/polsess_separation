"""Common utility functions shared across training scripts."""

import random
import warnings
import os
from typing import Dict, Any

import numpy as np
import torch
import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_eps_patch(eps_value: float = 1e-4) -> None:
    """Patch SpeechBrain's EPS to float16-safe value. Call before creating models."""
    if eps_value < 6e-5:
        print(f"WARNING: EPS value {eps_value} may underflow in float16 (min ~6e-5)")
    conv_tasnet_module.EPS = eps_value


def setup_warnings():
    """Suppress common warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def setup_device_and_amp(config, summary_info: Dict[str, Any]) -> str:
    """Setup device and AMP, populate summary_info, return device string."""
    if config.training.use_amp:
        apply_eps_patch(config.training.amp_eps)
        summary_info["eps_patch"] = f"{config.training.amp_eps} (enabled)"
    else:
        summary_info["eps_patch"] = "disabled"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_info["device"] = device

    return device
