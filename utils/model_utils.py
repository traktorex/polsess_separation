"""Model utility functions for common operations."""

import sys
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, Optional

from models import MAMBA_MODELS
# The inference loaders live in models/ so that they ship with the
# polsess-models package; re-exported here for the callers inside this repository.
from models.inference import load_checkpoint_file, load_model_for_inference  # noqa: F401


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Extract underlying model from torch.compile() wrapper if present."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def apply_torch_compile(
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None,
    mode: str = "default",
    dynamic: Optional[bool] = None,
) -> torch.nn.Module:
    """Apply torch.compile if available (PyTorch 2.0+, Linux only).

    `dynamic` is passed straight to torch.compile: None (default) lets Dynamo
    auto-detect dynamic shapes; False forces per-shape static specialization
    (needed by MossFormer2, whose token-shift / group-rearrange cannot be lowered
    under symbolic shapes — see train.py).
    """
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
            logger.info(f"Compiling model with torch.compile (mode={mode}, dynamic={dynamic})...")
        compiled_model = torch.compile(model, mode=mode, dynamic=dynamic)
        if logger:
            logger.info("Model compiled successfully!")
        return compiled_model
    except Exception as e:
        if logger:
            logger.warning(f"torch.compile failed: {e}")
        return model


def compile_for_model_type(
    model: torch.nn.Module,
    model_type: str,
    logger: Optional[logging.Logger] = None,
) -> torch.nn.Module:
    """Apply torch.compile with per-architecture settings (shared by train.py and
    train_sweep.py so the dispatch can't drift between them).

      - Mamba: skipped. Dynamo tracing into mamba_ssm.MambaInnerFn breaks the
        delta/conv1d_out dtype contract on native Linux.
      - MossFormer2: compiled with dynamic=False. Its vendored rotary block has
        the seq-len cache disabled (cache_if_possible=False), and its token-shift
        / group-rearrange cannot be lowered under symbolic shapes — so we force
        per-shape static specialization. Fixed-length training crops compile once;
        a new length triggers a one-time static recompile. (If this ever regresses
        on another torch build, fall back to skipping compile for mossformer2.)
      - TF-MossFormer: compiled with dynamic=False, same reason — it uses the
        same rotary-embedding-torch library as MossFormer2, plus a banded
        attention mask whose shape is a function of the sequence length. Static
        per-shape specialization compiles each crop length once.
      - Everything else: torch.compile defaults.
    """
    if model_type in MAMBA_MODELS:
        if logger:
            logger.info(
                f"Skipping torch.compile for {model_type} "
                "(incompatible with mamba_ssm CUDA kernels)."
            )
        return model
    if model_type in ("mossformer2", "tf_mossformer"):
        return apply_torch_compile(model, logger=logger, dynamic=False)
    return apply_torch_compile(model, logger=logger)


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


def read_wandb_run_id(checkpoint_path: str) -> Optional[str]:
    """Peek a checkpoint for its saved W&B run id (survey gap 14).

    Returns the id string written by newer checkpoints, or None for older
    checkpoints and runs trained with W&B disabled. Loaded on CPU and discarded
    immediately; only called on ``--resume``, so the extra read is negligible.
    Never raises — a missing/unreadable checkpoint just yields None (caller falls
    back to starting a fresh W&B run).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    run_id = ckpt.get("wandb_run_id") if isinstance(ckpt, dict) else None
    del ckpt
    return run_id


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
