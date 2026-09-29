"""Mamba building blocks for speech separation.

Adapted from xi-j/Mamba-TasNet (GPL-3.0, see LICENSE-Mamba-TasNet), with
low-level CUDA calls derived from mamba-ssm 2.3.1 (Apache-2.0, see
LICENSE-mamba-ssm) for API compatibility. Attribution summary: THIRD_PARTY.md.

Requires mamba-ssm library (Linux + CUDA only).
"""

try:
    from .mamba_blocks import MambaBlocksSequential
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
