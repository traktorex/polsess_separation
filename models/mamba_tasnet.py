"""Mamba-TasNet: Single-path Mamba for speech separation.

Based on: "Dual-Path Mamba: Short and Long-term Bidirectional Selective Structured
State Space Models for Speech Separation" (Jiang et al., 2024)
and "Speech Slytherin: Examining the Performance and Efficiency of Mamba for
Speech Separation, Recognition, and Synthesis" (Jiang et al., 2024)

Architecture: Encoder → Sequential BiMamba blocks → Decoder
Uses the same TasNet-style Conv1d encoder/decoder as ConvTasNet and DPRNN.

Requires mamba-ssm library (Linux + CUDA only).
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import Encoder, Decoder

from .mamba_modules import BiMambaBlock, MAMBA_AVAILABLE

# Conditional import — RMSNorm needed for norm_f
if MAMBA_AVAILABLE:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm


class MambaTasNet(nn.Module):
    """Mamba-TasNet: Conv-TasNet with Mamba blocks replacing the TCN separator.

    Args:
        N: Encoder/decoder channels.
        kernel_size: Encoder convolution kernel size.
        stride: Encoder convolution stride.
        C: Number of output sources (1=enhancement, 2=separation).
        bot_dim: Bottleneck dimension for Mamba blocks.
        n_mamba: Number of bidirectional Mamba blocks.
        d_state: SSM state dimension.
        d_conv: Local convolution width in Mamba.
        expand: Inner dimension expansion factor in Mamba.
    """

    def __init__(
        self,
        N: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 1,
        bot_dim: int = 256,
        n_mamba: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Shared TasNet encoder/decoder (same as ConvTasNet, DPRNN, SepFormer)
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)
        self.decoder = Decoder(
            in_channels=N, out_channels=1,
            kernel_size=kernel_size, stride=stride, bias=False,
        )

        # Separator: norm → bottleneck → Mamba blocks → mask projection
        self.norm = nn.GroupNorm(1, N)  # Channel-wise layer norm
        self.bottleneck = nn.Conv1d(N, bot_dim, 1)
        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(bot_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.norm_f = RMSNorm(bot_dim)  # Final norm after all Mamba blocks
        self.mask_conv = nn.Conv1d(bot_dim, C * N, 1)

        self.C = C
        self.N = N

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate mixture into C sources.

        Args:
            mixture: [B, 1, T] or [B, T].

        Returns:
            [B, T] if C=1, [B, C, T] if C>1.
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)

        # Encode: [B, T] → [B, N, L]
        mixture_w = self.encoder(mixture)

        # Separator
        x = self.norm(mixture_w)               # [B, N, L]
        x = self.bottleneck(x)                 # [B, bot_dim, L]
        x = x.transpose(1, 2)                  # [B, L, bot_dim] for Mamba
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm_f(x)                     # Final norm (paper: MambaBlocksSequential.norm_f)
        x = x.transpose(1, 2)                  # [B, bot_dim, L]

        # Generate and apply masks
        masks = self.mask_conv(x)              # [B, C*N, L]
        B, _, L = masks.shape
        masks = masks.view(B, self.C, self.N, L)
        masks = torch.relu(masks)

        # Decode each source
        estimates = []
        for i in range(self.C):
            masked = mixture_w * masks[:, i]   # [B, N, L]
            decoded = self.decoder(masked)     # [B, T]
            estimates.append(decoded)

        estimates = torch.stack(estimates, dim=1)  # [B, C, T]

        if self.C == 1:
            estimates = estimates.squeeze(1)    # [B, T]

        return estimates
