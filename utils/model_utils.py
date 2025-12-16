"""Model utility functions for common operations."""

import sys
import torch
import logging
from typing import Any, Dict, Optional
from pathlib import Path


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Extract underlying model from torch.compile() wrapper if present."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def apply_torch_compile(
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None,
    mode: str = "default",
) -> torch.nn.Module:
    """Apply torch.compile if available (PyTorch 2.0+, Linux only)."""
    if not hasattr(torch, "compile"):
        if logger:
            logger.info("torch.compile not available (PyTorch < 2.0)")
        return model
    
    if sys.platform != "linux":
        if logger:
            logger.info(
                f"Skipping torch.compile (requires Triton/Linux, detected: {sys.platform})"
            )
        return model
    
    try:
        if logger:
            logger.info(f"Compiling model with torch.compile (mode={mode})...")
        compiled_model = torch.compile(model, mode=mode)
        if logger:
            logger.info("Model compiled successfully!")
        return compiled_model
    except Exception as e:
        if logger:
            logger.warning(f"torch.compile failed: {e}")
        return model


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


def load_checkpoint_file(
    checkpoint_path: str, device: str = "cuda"
) -> Dict[str, Any]:
    """Load checkpoint file from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model weights from checkpoint, handling compiled models.

    Returns the loaded checkpoint dict for accessing metadata.
    """
    checkpoint = load_checkpoint_file(checkpoint_path, device)

    # Unwrap compiled model if needed
    model_to_load = unwrap_compiled_model(model)

    # Load state dict
    model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    return checkpoint
