"""DPMamba: Dual-Path Mamba for speech separation.

Adapted from xi-j/Mamba-TasNet. Uses MambaBlocksSequential as intra-chunk and
inter-chunk models within SpeechBrain's Dual_Path_Model framework.

The author's n_mamba_dp parameter is the total Mamba blocks across both paths:
each path gets n_mamba_dp // 2 blocks (e.g. n_mamba_dp=2 → 1 block per path).

References:
  - "Dual-Path Mamba" (Jiang et al., 2024, arXiv 2403.18257)

Requires mamba-ssm library (Linux + CUDA only).
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import (
    Encoder,
    Decoder,
    Dual_Path_Model,
)

from .mamba import MambaBlocksSequential


class DPMamba(nn.Module):
    """DPMamba: Dual-path processing with bidirectional Mamba blocks.

    Uses MambaBlocksSequential for each intra-chunk and inter-chunk path.
    Each path gets n_mamba_dp // 2 blocks (matching the author's config).
    SpeechBrain's Dual_Path_Model handles segmentation, overlap-add, and
    the dual-path iteration.

    Args:
        N: Encoder/decoder channels.
        kernel_size: Encoder convolution kernel size.
        stride: Encoder convolution stride.
        C: Number of output sources (1=enhancement, 2=separation).
        num_layers: Number of dual-path iterations.
        chunk_size: Chunk length K for dual-path segmentation.
        n_mamba_dp: Total BiMamba blocks across intra+inter (each gets half).
        d_state: SSM state dimension.
        d_conv: Local convolution width in Mamba.
        expand: Inner dimension expansion factor in Mamba.
        bidirectional: Use BiMamba (True) or standard Mamba (False).
        rms_norm: Use RMSNorm instead of LayerNorm in Mamba blocks.
        skip_around_intra: Add residual around intra-chunk processing.
        residual_in_fp32: Cast residual stream to fp32 between Mamba blocks.
            Prevents precision loss accumulation in deep sequential chains
            when training with bf16 AMP.
    """

    def __init__(
        self,
        N: int = 64,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 1,
        num_layers: int = 8,
        chunk_size: int = 250,
        n_mamba_dp: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        rms_norm: bool = True,
        skip_around_intra: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.C = C
        self.N = N

        # Shared TasNet encoder/decoder
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)
        self.decoder = Decoder(
            in_channels=N, out_channels=1,
            kernel_size=kernel_size, stride=stride, bias=False,
        )

        # Intra-chunk (local) and inter-chunk (global) Mamba blocks.
        # Author's convention: n_mamba_dp is total across both paths,
        # each path gets n_mamba_dp // 2 blocks (e.g. 2 → 1 per path).
        n_mamba_per_path = n_mamba_dp // 2
        mamba_kwargs = dict(
            n_mamba=max(n_mamba_per_path, 1),
            bidirectional=bidirectional,
            d_model=N,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            rms_norm=rms_norm,
            conv_bias=True,
            bias=False,
            residual_in_fp32=residual_in_fp32,
        )
        intra_block = MambaBlocksSequential(**mamba_kwargs)
        inter_block = MambaBlocksSequential(**mamba_kwargs)

        # Dual-path separator (same framework as DPRNN and SepFormer).
        # linear_layer_after_inter_intra=False because MambaBlocksSequential
        # preserves dimension (unlike SBRNNBlock which doubles it).
        self.separator = Dual_Path_Model(
            in_channels=N,
            out_channels=N,
            intra_model=intra_block,
            inter_model=inter_block,
            num_layers=num_layers,
            norm="ln",
            K=chunk_size,
            num_spks=C,
            skip_around_intra=skip_around_intra,
            linear_layer_after_inter_intra=False,
        )

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
            mask = est_mask[i]               # [B, N, L]
            masked = mixture_w * mask
            decoded = self.decoder(masked)   # [B, T]
            estimates.append(decoded)

        estimates = torch.stack(estimates, dim=1)  # [B, C, T]

        if self.C == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
