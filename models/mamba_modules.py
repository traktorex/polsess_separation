"""Shared Mamba building blocks for speech separation architectures.

Provides the BiMambaBlock used by Mamba-TasNet, DPMamba, and SepMamba.
Implements BiMamba v2 from Vision Mamba (Zhu et al., 2024), as used in
the Mamba-TasNet / DPMamba papers (Jiang et al., 2024).

Key design: shared in_proj/out_proj between forward and backward directions,
with separate SSM parameters (conv1d, x_proj, dt_proj, A, D) per direction.
Uses mamba_inner_fn fused CUDA kernel for memory-efficient forward/backward.

Note: SPMamba uses its own Mamba integration (frequency-domain, concatenation-based,
interleaved with attention) and does NOT share these modules.

Requires mamba-ssm library (Linux + CUDA only).
"""

import math
import warnings
import torch
import torch.nn as nn

# Mamba dependencies (Linux + CUDA only)
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    MAMBA_AVAILABLE = True
except ImportError as e:
    MAMBA_AVAILABLE = False
    warnings.warn(
        f"mamba-ssm not available. Mamba-TasNet/DPMamba/SepMamba will not work. Error: {e}",
        UserWarning,
        stacklevel=2,
    )


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block (BiMamba v2) with pre-norm residual.

    Matches the architecture from xi-j/Mamba-TasNet (adapted from Vision Mamba):
    - Shared in_proj and out_proj between forward and backward directions
    - Separate SSM parameters (conv1d, x_proj, dt_proj, A, D) per direction
    - Forward + backward outputs averaged (0.5 * fwd + 0.5 * bwd)
    - Pre-norm: RMSNorm before the mixer, with residual connection

    Uses mamba_inner_fn fused CUDA kernel for memory-efficient computation.
    Since out_proj is linear, applying it per-direction then averaging is
    mathematically identical to averaging then applying out_proj.

    Args:
        d_model: Feature dimension.
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Inner dimension expansion factor (default 2).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for BiMambaBlock. "
                "Install with: pip install mamba-ssm (Linux + CUDA only)"
            )

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        # Pre-norm (applied before the mixer)
        self.norm = RMSNorm(d_model)

        # Shared input/output projections (single copy for both directions)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Forward direction SSM parameters
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(self._init_A(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Backward direction SSM parameters (separate from forward)
        self.conv1d_b = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_b_log = nn.Parameter(self._init_A(self.d_inner, d_state))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        # Special initialization for dt_proj biases (softplus → [dt_min, dt_max])
        self._init_dt_proj(self.dt_proj)
        self._init_dt_proj(self.dt_proj_b)

    @staticmethod
    def _init_A(d_inner: int, d_state: int) -> torch.Tensor:
        """S4D real initialization for state matrix A."""
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).contiguous()
        return torch.log(A)

    @staticmethod
    def _init_dt_proj(
        dt_proj: nn.Linear,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        """Initialize dt_proj bias so that softplus(bias) ∈ [dt_min, dt_max]."""
        dt = torch.exp(
            torch.rand(dt_proj.out_features)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input bidirectionally with pre-norm residual.

        Args:
            x: [B, L, D] input tensor.

        Returns:
            [B, L, D] output tensor (same shape as input).
        """
        residual = x
        x = self.norm(x)

        # Shared input projection: [B, L, D] → [B, 2*d_inner, L]
        xz = self.in_proj(x).transpose(1, 2).contiguous()

        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())

        # Forward direction (fused kernel — memory-efficient)
        out = mamba_inner_fn(
            xz,
            self.conv1d.weight, self.conv1d.bias,
            self.x_proj.weight, self.dt_proj.weight,
            self.out_proj.weight, self.out_proj.bias,
            A, None, None, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )  # [B, L, D]

        # Backward direction (flip → fused kernel → flip)
        out_b = mamba_inner_fn(
            xz.flip([-1]),
            self.conv1d_b.weight, self.conv1d_b.bias,
            self.x_proj_b.weight, self.dt_proj_b.weight,
            self.out_proj.weight, self.out_proj.bias,
            A_b, None, None, self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )  # [B, L, D]

        # Average directions (out_proj already applied by fused kernel;
        # since out_proj is linear: avg(proj(a), proj(b)) == proj(avg(a, b)))
        out = 0.5 * out + 0.5 * out_b.flip([1])

        return out + residual
