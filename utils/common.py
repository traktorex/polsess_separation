"""Common utility functions shared across training scripts."""

import importlib
import logging
import platform
import random
import socket
import subprocess
import sys
import warnings
import os
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module

_REPO_ROOT = Path(__file__).resolve().parents[1]


def git_provenance() -> Dict[str, Any]:
    """Return the repo's git commit SHA and dirty flag for run/eval provenance.

    Runs ``git`` against the repository root regardless of the process cwd.
    Returns ``{"git_sha": "unknown", "git_dirty": "unknown"}`` if git is
    unavailable (e.g. a source export with no .git) rather than raising, so a
    provenance column is always populated. ``git_dirty`` is a bool on success.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return {"git_sha": sha, "git_dirty": bool(status)}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": "unknown"}


def _optional_pkg_version(module_name: str) -> Optional[str]:
    """Return an installed package's ``__version__`` or None if not importable.

    Used for the Mamba-stack deps (``mamba_ssm``/``triton``) that are present in
    ``venv_mamba3`` but not the main ``venv`` — their presence/version is what
    tells the two environments apart in a run manifest.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def collect_run_manifest(seed: Optional[int] = None) -> Dict[str, Any]:
    """Collect run provenance for reproducibility (survey gap 5).

    Captures git state, framework/driver versions, the optional Mamba deps
    (which distinguish ``venv`` from ``venv_mamba3``), GPU, host, seed and
    ``sys.argv``. Embedded under ``"provenance"`` in every checkpoint, written to
    ``run_manifest.yaml`` beside ``config.yaml``, and pushed to the W&B run
    config — so "which code/env produced this checkpoint" is always answerable.

    Every probe degrades gracefully (never raises), including in a CPU-only test
    context where ``torch.cuda`` is unavailable — those fields come back None.
    """
    manifest: Dict[str, Any] = dict(git_provenance())  # git_sha, git_dirty
    manifest["torch_version"] = torch.__version__
    manifest["cuda_version"] = torch.version.cuda  # None on CPU-only torch builds
    try:
        manifest["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        manifest["cudnn_version"] = None

    gpu_name = None
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = None
    manifest["gpu_name"] = gpu_name

    manifest["mamba_ssm_version"] = _optional_pkg_version("mamba_ssm")
    manifest["triton_version"] = _optional_pkg_version("triton")
    manifest["hostname"] = socket.gethostname()
    manifest["python_version"] = platform.python_version()
    manifest["seed"] = seed
    manifest["argv"] = list(sys.argv)
    return manifest


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Pins cuDNN to deterministic algorithms with autotuning off. This is the
    project's long-standing default and the state every past training run was
    born under; the optional ``training.deterministic`` flag layers extra
    strictness (or relaxation) on top via ``configure_determinism`` — this
    function is intentionally left unchanged so historical behavior is preserved.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_determinism(
    mode: Optional[bool] = None, logger: Optional[logging.Logger] = None
) -> None:
    """Apply the run's determinism policy on top of ``set_seed``'s cuDNN settings.

    ``set_seed`` already pins ``cudnn.deterministic=True`` and
    ``cudnn.benchmark=False``. This layers the optional ``training.deterministic``
    policy over that baseline:

    - ``mode is None`` (default): **no change** — reproduces today's behavior
      exactly. cuDNN stays deterministic with no benchmark autotuning, and TF32
      matmuls stay enabled (the entry points call
      ``set_float32_matmul_precision('high')``). This is the setting the upcoming
      SPMamba full-ks8 runs must stay comparable to, so it is the default and the
      strictness below is strictly opt-in.
    - ``mode is True``: additionally enable
      ``torch.use_deterministic_algorithms(True, warn_only=True)`` and set
      ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (required for deterministic cuBLAS
      GEMMs). ``warn_only=True`` downgrades a missing deterministic kernel to a
      warning instead of crashing a long run. TF32 is left on — turn it off
      yourself if you need bit-exact matmuls. NOTE: for the cuBLAS workspace
      setting to fully bind, ``CUBLAS_WORKSPACE_CONFIG`` should ideally be in the
      environment before the CUDA context initializes; setting it here is
      best-effort and pairs with ``warn_only`` for safety.
    - ``mode is False``: enable ``cudnn.benchmark=True`` for the free
      conv-autotune speedup on fixed-length conv-heavy runs. This makes
      convolution algorithm selection shape-dependent and non-deterministic — do
      not use it for a run whose numbers must be reproducible.
    """
    if mode is None:
        return
    if mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if logger:
            logger.info(
                "Determinism: strict "
                "(use_deterministic_algorithms=True, warn_only; CUBLAS_WORKSPACE_CONFIG set)"
            )
    else:
        torch.backends.cudnn.benchmark = True
        if logger:
            logger.info(
                "Determinism: relaxed (cudnn.benchmark=True — conv autotuning, non-deterministic)"
            )


def apply_eps_patch(eps_value: float = 1e-4) -> None:
    """Patch SpeechBrain's ConvTasNet EPS to a float16-safe value.

    ConvTasNet-only (survey gap 17: the generic name previously overstated its
    reach): this only rebinds
    ``speechbrain.lobes.models.conv_tasnet.EPS``, the module-level constant
    ConvTasNet's normalization layers read at forward time. Other architectures
    (DPRNN, SepFormer, MossFormer2, Mamba family) do not import EPS from this
    module and are unaffected by this call — each either uses its own epsilon
    or doesn't need this AMP-underflow guard. Call sites (``evaluate.py``,
    ``setup_device_and_amp`` below) call it unconditionally whenever AMP is
    enabled regardless of ``model_type``; that's harmless (a no-op for
    non-ConvTasNet models) but not a signal that it patches every architecture.
    Call before creating a ConvTasNet model.
    """
    if eps_value < 6e-5:
        print(f"WARNING: EPS value {eps_value} may underflow in float16 (min ~6e-5)")
    conv_tasnet_module.EPS = eps_value


def setup_warnings():
    """Configure warning filters for cleaner output.

    The filter list lives in ``utils.warning_filters`` so it can be applied
    before speechbrain is imported (some warnings fire at import time); see that
    module's docstring. This re-applies it for code paths that don't go through
    an entry point and sets a couple of subprocess/inductor env vars.

    Previously also set a blanket ``PYTHONWARNINGS=ignore::UserWarning`` (survey
    gap 17) — dropped: it silenced *every* UserWarning process-wide, including
    ones the targeted filters above were never meant to cover, undoing their
    precision. Rely on ``warning_filters.apply()`` for anything that should be
    suppressed; if new third-party noise turns up, add a targeted filter there
    instead of reinstating the blanket env var.
    """
    from . import warning_filters

    warning_filters.apply()

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
