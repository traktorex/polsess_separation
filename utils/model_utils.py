"""Model utility functions for common operations."""

import torch


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Extract underlying model from torch.compile() wrapper if present."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable parameters in model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(num_params: int) -> str:
    """Format parameter count as human-readable string (e.g., '1.23M')."""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)
