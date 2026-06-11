"""Mamba-TasNet: Single-path Mamba for speech separation.

Adapted from xi-j/Mamba-TasNet's MaskNet (modules/mamba_masknet.py):
  Encoder → LayerNorm → Bottleneck → MambaBlocksSequential → Mask → Decoder

References:
  - "Dual-Path Mamba" (Jiang et al., 2024, arXiv 2403.18257)
  - "Speech Slytherin" (Jiang et al., 2024, arXiv 2407.09732)

Requires mamba-ssm library (Linux + CUDA only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.dual_path import Encoder, Decoder

from .mamba import MambaBlocksSequential


class MambaTasNet(nn.Module):
    """Mamba-TasNet: Conv-TasNet with BiMamba blocks replacing the TCN separator.

    Adapted from the author's MaskNet. The separator operates in [B, L, D]
    format (time-first): LayerNorm → Linear bottleneck → MambaBlocksSequential
    → Linear mask → ReLU → masking → Decoder.

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
        bidirectional: Use BiMamba (True) or standard Mamba (False).
        rms_norm: Use RMSNorm instead of LayerNorm in Mamba blocks.
        residual_in_fp32: Keep residual stream in fp32 under AMP (stabilizes deep models).
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
        bidirectional: bool = True,
        rms_norm: bool = True,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.C = C
        self.N = N

        # Shared TasNet encoder/decoder (same as ConvTasNet, DPRNN, SepFormer)
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)
        self.decoder = Decoder(
            in_channels=N, out_channels=1,
            kernel_size=kernel_size, stride=stride, bias=False,
        )

        # Separator (adapted from author's MaskNet)
        # All operations in [B, L, D] format after the initial transpose
        self.layer_norm = nn.LayerNorm(N)
        self.bottleneck = nn.Linear(N, bot_dim, bias=False)
        self.mamba_net = MambaBlocksSequential(
            n_mamba=n_mamba,
            bidirectional=bidirectional,
            d_model=bot_dim,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            conv_bias=True,
            bias=False,
        )
        self.mask_linear = nn.Linear(bot_dim, C * N, bias=False)

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

        # Separator (matches author's MaskNet.forward)
        # Transpose to [B, L, N] for channel-last processing
        x = mixture_w.permute(0, 2, 1)
        B, L, D = x.shape
        x = self.layer_norm(x)
        x = self.bottleneck(x)             # [B, L, bot_dim]
        x = self.mamba_net(x)              # [B, L, bot_dim]
        score = self.mask_linear(x)        # [B, L, C*N]

        # Reshape to [C, B, N, L] mask format (matches author's MaskNet)
        score = score.reshape(B, L, self.C, D)
        masks = score.permute(2, 0, 3, 1)  # [C, B, N, L]
        masks = F.relu(masks)

        # Decode each source
        estimates = []
        for i in range(self.C):
            masked = mixture_w * masks[i]   # [B, N, L]
            decoded = self.decoder(masked)  # [B, T]
            estimates.append(decoded)

        estimates = torch.stack(estimates, dim=1)  # [B, C, T]

        if self.C == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
