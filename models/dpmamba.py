"""DPMamba: Dual-Path Mamba for speech separation.

Based on: "Dual-Path Mamba: Short and Long-term Bidirectional Selective Structured
State Space Models for Speech Separation" (Jiang et al., 2024)

Architecture: Encoder → Dual_Path_Model(BiMamba intra, BiMamba inter) → Decoder
Reuses SpeechBrain's Dual_Path_Model (same framework as DPRNN and SepFormer),
swapping RNN/Transformer blocks for bidirectional Mamba blocks.

Requires mamba-ssm library (Linux + CUDA only).
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import (
    Encoder,
    Decoder,
    Dual_Path_Model,
)

from .mamba_modules import BiMambaBlock


class SBMambaBlock(nn.Module):
    """Mamba block compatible with SpeechBrain's Dual_Path_Model.

    Thin wrapper that matches the interface expected by Dual_Computation_Block:
    forward(x: [B, L, N]) → [B, L, N].
    """

    def __init__(self, input_size: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.bimamba = BiMambaBlock(input_size, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bimamba(x)


class DPMamba(nn.Module):
    """DPMamba: Dual-path processing with bidirectional Mamba blocks.

    Uses the same encoder/decoder as ConvTasNet, DPRNN, and SepFormer.
    The dual-path framework splits encoded features into overlapping chunks,
    applies intra-chunk (local) and inter-chunk (global) Mamba processing,
    then reconstructs via overlap-add.

    Args:
        N: Encoder/decoder channels.
        kernel_size: Encoder convolution kernel size.
        stride: Encoder convolution stride.
        C: Number of output sources (1=enhancement, 2=separation).
        num_layers: Number of dual-path iterations.
        chunk_size: Chunk length K for dual-path segmentation.
        d_state: SSM state dimension.
        d_conv: Local convolution width in Mamba.
        expand: Inner dimension expansion factor in Mamba.
    """

    def __init__(
        self,
        N: int = 64,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 1,
        num_layers: int = 8,
        chunk_size: int = 250,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Shared TasNet encoder/decoder
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)

        # Intra-chunk (local) and inter-chunk (global) Mamba blocks
        intra_block = SBMambaBlock(N, d_state=d_state, d_conv=d_conv, expand=expand)
        inter_block = SBMambaBlock(N, d_state=d_state, d_conv=d_conv, expand=expand)

        # Dual-path separator (same framework as DPRNN and SepFormer)
        # linear_layer_after_inter_intra=False because BiMamba output dim == input dim
        # (unlike SBRNNBlock which outputs 2*hidden_size when bidirectional)
        self.separator = Dual_Path_Model(
            in_channels=N,
            out_channels=N,
            intra_model=intra_block,
            inter_model=inter_block,
            num_layers=num_layers,
            norm="ln",
            K=chunk_size,
            num_spks=C,
            skip_around_intra=True,
            linear_layer_after_inter_intra=False,
        )

        self.decoder = Decoder(
            in_channels=N, out_channels=1,
            kernel_size=kernel_size, stride=stride, bias=False,
        )

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

        # Separate: [B, N, L] → [C, B, N, L]
        est_mask = self.separator(mixture_w)

        # Decode each source
        estimates = []
        for i in range(self.C):
            mask = est_mask[i]                 # [B, N, L]
            masked = mixture_w * mask
            decoded = self.decoder(masked)     # [B, T]
            estimates.append(decoded)

        estimates = torch.stack(estimates, dim=1)  # [B, C, T]

        if self.C == 1:
            estimates = estimates.squeeze(1)    # [B, T]

        return estimates
