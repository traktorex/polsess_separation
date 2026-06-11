"""Bidirectional Mamba (BiMamba v2) for speech separation.

Adapted from xi-j/Mamba-TasNet's modules/mamba/bimamba.py, which is itself
adapted from Vision Mamba (Zhu et al., 2024):
  https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py

BiMamba v2 architecture:
  - Shared in_proj and out_proj between forward and backward directions
  - Separate SSM parameters (conv1d, x_proj, dt_proj, A, D) per direction
  - Processes input forward and backward, averages outputs, then projects

Uses mamba_inner_fn_no_out_proj (derived from mamba_ssm 2.3.1) to process
each direction without the output projection, enabling the average-then-project
pattern from the original paper.

Requires mamba-ssm library (Linux + CUDA only).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .selective_scan_interface import mamba_inner_fn_no_out_proj


class Mamba(nn.Module):
    """Bidirectional Mamba (BiMamba v2) mixer.

    This is the core SSM mixer — no norm, no residual. Those are handled by
    mamba_ssm.modules.block.Block which wraps this class.

    Adapted from xi-j/Mamba-TasNet's BiMamba class. Key differences from
    standard (unidirectional) Mamba:
      - Shared in_proj/out_proj between forward and backward directions
      - Separate per-direction SSM parameters (_b suffix for backward)
      - Forward + backward outputs averaged before out_proj

    Args:
        d_model: Input/output feature dimension.
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Inner dimension expansion factor (default 2).
        dt_rank: Rank for delta projection (default "auto" = ceil(d_model/16)).
        dt_min: Minimum dt value for initialization (default 0.001).
        dt_max: Maximum dt value for initialization (default 0.1).
        dt_init_floor: Floor for dt initialization (default 1e-4).
        conv_bias: Whether conv1d has bias (default True).
        bias: Whether linear projections have bias (default False).
        init_layer_scale: If not None, scale output by this learnable gamma.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        init_layer_scale: float = None,
        **kwargs,  # Accept and ignore extra kwargs (e.g., layer_idx from Block)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Shared projections (both directions use same in_proj and out_proj)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Forward direction SSM parameters
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # S4D real initialization
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Backward direction SSM parameters (separate from forward)
        self.conv1d_b = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=conv_bias,
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_b = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_b = A_b.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        # dt_proj bias initialization: softplus(bias) ∈ [dt_min, dt_max]
        self._init_dt_proj_bias(self.dt_proj, dt_min, dt_max, dt_init_floor)
        self._init_dt_proj_bias(self.dt_proj_b, dt_min, dt_max, dt_init_floor)

        # Optional layer scale (from author's implementation)
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones(d_model), requires_grad=True
            )
        else:
            self.gamma = None

    @staticmethod
    def _init_dt_proj_bias(dt_proj, dt_min, dt_max, dt_init_floor):
        """Initialize dt_proj bias so softplus(bias) samples uniformly from [dt_min, dt_max]."""
        dt = torch.exp(
            torch.rand(dt_proj.out_features)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Bidirectional SSM forward pass.

        Args:
            hidden_states: [B, L, D] input (after norm, from Block wrapper).

        Returns:
            [B, L, D] output.
        """
        # Shared input projection: [B, L, D] -> [B, 2*d_inner, L]
        xz = self.in_proj(hidden_states).transpose(1, 2).contiguous()

        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())

        # Forward direction (no out_proj): [B, d_inner, L]
        out = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight, self.conv1d.bias,
            self.x_proj.weight, self.dt_proj.weight,
            A, None, None, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # Backward direction (flip -> process -> flip): [B, d_inner, L]
        out_b = mamba_inner_fn_no_out_proj(
            xz.flip([-1]),
            self.conv1d_b.weight, self.conv1d_b.bias,
            self.x_proj_b.weight, self.dt_proj_b.weight,
            A_b, None, None, self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )

        # Average forward + backward, then apply shared output projection
        out = 0.5 * (out + out_b.flip([-1]))
        out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

        if self.gamma is not None:
            out = out * self.gamma

        return out
