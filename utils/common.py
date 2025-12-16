"""Common utility functions shared across training scripts."""

import random
import warnings
import os
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Union

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
    """Configure warning filters for cleaner output.
    
    Suppresses known warnings that don't affect functionality:
    - SpeechBrain/torchaudio deprecation warnings (future library changes)
    - TorchMetrics pkg_resources deprecation (setuptools migration)
    - Torch dynamo/inductor warnings (compilation optimizations)
    - ComplexHalf experimental warning (used in SPMamba STFT)
    """
    import warnings
    
    # SpeechBrain torchaudio backend deprecation (transition to TorchCodec)
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio._backend.list_audio_backends.*",
        category=UserWarning,
    )
    
    # TorchAudio load deprecation (transition to TorchCodec)
    warnings.filterwarnings(
        "ignore",
        message=".*this function's implementation will be changed.*torchcodec.*",
        category=UserWarning,
    )
    
    # TorchMetrics pkg_resources deprecation (setuptools migration)
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )
    
    # Torch dynamo warnings for pybind functions (SPMamba selective scan)
    warnings.filterwarnings(
        "ignore",
        message=".*Dynamo does not know how to trace.*selective_scan_cuda.*",
        category=UserWarning,
    )
    
    # ComplexHalf experimental support (used in SPMamba STFT)
    warnings.filterwarnings(
        "ignore",
        message=".*ComplexHalf support is experimental.*",
        category=UserWarning,
    )
    
    # Suppress pybind deprecation warnings from frozen importlib
    warnings.filterwarnings(
        "ignore",
        message=".*SwigPy.*has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*swigvarlink.*has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    
    # Suppress torch inductor SM warnings (logged to stderr)
    os.environ["TORCHINDUCTOR_WARNINGS"] = "0"


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


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataclass_to_dict(obj: Any) -> dict:
    """Convert dataclass or SimpleNamespace to dict recursively."""
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    else:
        return obj
