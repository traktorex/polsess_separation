"""Mamba building blocks for speech separation.

Adapted from xi-j/Mamba-TasNet, with low-level CUDA calls derived from
mamba-ssm 2.3.1 for API compatibility.

Requires mamba-ssm library (Linux + CUDA only).
"""

try:
    from .mamba_blocks import MambaBlocksSequential
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
