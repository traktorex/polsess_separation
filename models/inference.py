"""Loading trained separators for inference.

Two entry points, one per on-disk format:

- ``load_model_for_inference(checkpoint.pt)`` reads a *training checkpoint* as the
  Trainer writes it: weights + optimizer + scheduler + the pickled config dict.
  Used inside this repository (evaluation, notebooks, figure scripts).
- ``load_separator(bundle_dir)`` reads an *inference bundle* as
  ``scripts/export_separator.py`` writes it: ``weights.safetensors`` +
  ``separator.json``. Weights only, no pickle, self-describing. This is the
  format other projects consume.

Relative imports only: this file also ships as ``polsess_models.inference`` (see
the root ``pyproject.toml`` and ``models/README.md``), where the package is not
called ``models``.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import __version__, get_model

BUNDLE_WEIGHTS = "weights.safetensors"
BUNDLE_META = "separator.json"
BUNDLE_FORMAT_VERSION = 1
# separator.json keys load_separator cannot work without; the rest is provenance.
BUNDLE_REQUIRED_KEYS = ("model_type", "model_params", "sample_rate", "n_src")


def load_checkpoint_file(
    checkpoint_path: str, device: str = "cuda"
) -> Dict[str, Any]:
    """Load checkpoint file from disk.

    ``weights_only=False`` is explicit (survey gap 13): our checkpoints carry a
    pickled config dict (and now a provenance dict), and torch 2.6+ flipped the
    default to True, which would refuse to unpickle them. This matches config.py.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def resolve_architecture(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(model_type, constructor kwargs)`` from a checkpoint config dict.

    Applies the backward-compat defaults that old checkpoint configs need. The
    kwargs dict is the one inside ``config`` and is edited in place, as before
    this function was split out of ``load_model_for_inference``.
    """
    model_type = config.get("model", {}).get("model_type", "convtasnet")
    model_params = config.get("model", {}).get(model_type, {})

    # Backward compat: SepFormer checkpoints before 2026-03 were trained without
    # positional encoding (see models/sepformer.py module docstring for details).
    # Their saved configs lack this key, so default to False to match trained weights.
    if model_type == "sepformer" and "use_positional_encoding" not in model_params:
        model_params["use_positional_encoding"] = False

    # Backward compat: `sample_rate` was removed from SPMamba; drop it from
    # legacy checkpoint configs so SPMamba(**model_params) doesn't TypeError.
    if model_type == "spmamba":
        model_params.pop("sample_rate", None)

    return model_type, model_params


def normalize_state_dict(
    model_type: str, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Rename legacy checkpoint keys so they load into the current classes."""
    # Backward compat: SpeechBrain renamed SBTransformerBlock's inner attribute
    # from `transformer` to `mdl` (somewhere between the old training env and now).
    # Remap checkpoint keys so old checkpoints load into the current model.
    if model_type == "sepformer" and any(".transformer." in k for k in state_dict):
        state_dict = {k.replace(".transformer.", ".mdl."): v for k, v in state_dict.items()}

    # Defensive torch.compile-artifact strip: the save side (train.py) already
    # unwraps `_orig_mod` before saving, so our own checkpoints never carry the
    # prefix. This guards externally-produced checkpoints (e.g. saved directly
    # from a compiled model without unwrapping) so this generic loader doesn't
    # silently 0-match every key and raise a confusing "missing keys" error.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_for_inference(
    checkpoint_path: str,
    device: str = "cuda",
    config_override: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a trained model from a training checkpoint, ready for inference.

    Creates the model architecture from the config embedded in the checkpoint,
    loads trained weights, and sets the model to eval mode. This is the single
    entry point for all post-training use cases inside this repository
    (evaluation, notebooks, export).

    Args:
        checkpoint_path: Path to model checkpoint file.
        device: Device to load the model on.
        config_override: Optional config dict to use instead of the one in
            the checkpoint. Must follow the same structure as checkpoint configs
            (with 'model.model_type' and 'model.<model_type>' keys).

    Returns:
        Tuple of (model in eval mode, checkpoint dict with metadata).

    Raises:
        ValueError: If no config is available (neither in checkpoint nor override).
    """
    checkpoint = load_checkpoint_file(checkpoint_path, device)

    # Use override config if provided, otherwise use config from checkpoint
    config = config_override or checkpoint.get("config")
    if config is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain a config "
            "and no config_override was provided."
        )

    model_type, model_params = resolve_architecture(config)
    model = get_model(model_type)(**model_params)
    model.load_state_dict(
        normalize_state_dict(model_type, checkpoint["model_state_dict"])
    )
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_separator(
    bundle_dir: str, device: str = "cuda"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load an inference bundle written by ``scripts/export_separator.py``.

    Returns ``(model in eval mode, metadata)``; the metadata is the parsed
    ``separator.json`` (``sample_rate``, ``n_src``, ``model_type``, provenance).
    The model maps ``[batch, time]`` at ``sample_rate`` to ``[batch, n_src, time]``
    (SepFormer trims a sample at some input lengths).

    Every failure is loud: a directory that is not a bundle, a bundle format
    this version does not know, metadata without the required keys, an
    architecture this installation does not ship, or weights that do not fit the
    architecture (strict load). Architecture and weight failures name the
    exporting and the installed polsess-models version.
    """
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / BUNDLE_META
    weights_path = bundle_dir / BUNDLE_WEIGHTS
    for path in (meta_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Not a separator bundle: {bundle_dir} (missing {path.name}). "
                "Bundles are created with scripts/export_separator.py in "
                "polsess_separation."
            )

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except ValueError as err:                       # json.JSONDecodeError
        raise ValueError(f"{meta_path}: not valid JSON ({err}).") from err
    if meta.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"{meta_path}: bundle format {meta.get('format_version')!r}, this "
            f"polsess-models ({__version__}) reads format {BUNDLE_FORMAT_VERSION}."
        )

    missing = [key for key in BUNDLE_REQUIRED_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{meta_path}: missing required keys {missing}.")

    # Imported here, not at module top: training boxes import this module through
    # utils/ and need safetensors only when they export or load a bundle.
    from safetensors.torch import load_file

    # An architecture this version does not ship (ValueError), constructor kwargs
    # it does not know (TypeError) and weights that do not fit (RuntimeError) all
    # have the same likely cause, so each names both versions.
    try:
        model = get_model(meta["model_type"])(**meta["model_params"])
    except (ValueError, TypeError) as err:
        raise type(err)(
            f"{bundle_dir}: {err}. Bundle exported with polsess-models "
            f"{meta.get('polsess_models_version')}, installed: {__version__}."
        ) from err
    try:
        state_dict = load_file(str(weights_path), device="cpu")
    except Exception as err:                        # safetensors' own error type
        raise RuntimeError(
            f"{weights_path}: cannot be read as a safetensors file ({err}); "
            "a truncated or corrupt copy?"
        ) from err
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        raise RuntimeError(
            f"{bundle_dir}: {err}. Bundle exported with polsess-models "
            f"{meta.get('polsess_models_version')}, installed: {__version__}."
        ) from err
    model = model.to(device)
    model.eval()

    return model, meta
