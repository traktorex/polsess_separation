"""Model factory for creating models from configuration.

This module provides a clean factory pattern for model instantiation,
eliminating the need for repetitive if-elif chains and manual parameter mapping.
"""

import torch.nn as nn
from typing import Any
from config import ModelConfig
from models import get_model


def create_model_from_config(config: ModelConfig, summary_info: dict = None) -> nn.Module:
    """Create model from config using dataclass unpacking.
    
    Automatically unpacks model-specific parameters, eliminating manual parameter mapping.
    """
    model_class = get_model(config.model_type)
    
    # Get model-specific parameters from nested config
    params = getattr(config, config.model_type, None)
    
    if params is None:
        raise ValueError(
            f"Model type '{config.model_type}' not configured. "
            f"Expected config.{config.model_type} to exist."
        )
    
    # Unpack dataclass parameters directly into model constructor
    model = model_class(**vars(params))
    
    # Populate summary info if requested
    if summary_info is not None:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        summary_info["model_params_millions"] = num_params
        summary_info["model_type"] = config.model_type
    
    return model
