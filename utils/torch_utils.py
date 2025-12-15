"""PyTorch-specific utilities for model compilation and device setup."""

import sys
import torch
import logging
from typing import Optional


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
