"""SepMamba: U-Net with bidirectional Mamba for speech separation.

Based on: "SepMamba: State-space models for speaker separation using Mamba"
(Schin et al., 2024, ICASSP 2025)

Architecture: Multi-stage U-Net with progressive Conv1d downsampling,
BiMamba blocks at each level, and skip connections for upsampling.
Unlike Mamba-TasNet and DPMamba, this model uses its own hierarchical
encoder/decoder (not the shared TasNet-style Conv1d encoder).

Requires mamba-ssm library (Linux + CUDA only).
"""

import torch
import torch.nn as nn

from .mamba_modules import BiMambaBlock


class DownBlock(nn.Module):
    """Downsampling block: Conv1d (stride 2) + activation + BiMamba."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_mamba: int,
                 d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=2,
            padding=kernel_size // 2,
        )
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.PReLU()
        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(out_ch, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C_in, T] → [B, C_out, T//2]."""
        x = self.act(self.norm(self.conv(x)))
        # Mamba expects [B, L, D]
        x = x.transpose(1, 2)
        for block in self.mamba_blocks:
            x = block(x)
        return x.transpose(1, 2)


class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose1d (stride 2) + skip fusion + BiMamba."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int,
                 n_mamba: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_ch, in_ch, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1,
        )
        # 1x1 conv to fuse upsampled features with skip connection
        self.fusion = nn.Conv1d(in_ch + skip_ch, out_ch, 1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.PReLU()
        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(out_ch, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """[B, C_in, T] + skip [B, C_skip, T*2] → [B, C_out, T*2]."""
        x = self.upsample(x)
        # Trim to match skip length (stride-2 can cause ±1 mismatch)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.fusion(x)))
        # Mamba processing
        x = x.transpose(1, 2)
        for block in self.mamba_blocks:
            x = block(x)
        return x.transpose(1, 2)


class SepMamba(nn.Module):
    """SepMamba: U-Net speech separator with bidirectional Mamba.

    Progressive downsampling creates multi-scale representations.
    Channel progression: dim → 2*dim → 4*dim (encoder) → 2*dim → dim (decoder).

    Args:
        C: Number of output sources (1=enhancement, 2=separation).
        dim: Base channel dimension (doubles each stage).
        n_stages: Number of encoder/decoder stages (U-Net depth).
        n_mamba: Number of BiMamba blocks per stage.
        kernel_size: Convolution kernel size for down/up sampling.
        d_state: SSM state dimension.
        d_conv: Local convolution width in Mamba.
        expand: Inner dimension expansion factor in Mamba.
    """

    def __init__(
        self,
        C: int = 1,
        dim: int = 64,
        n_stages: int = 3,
        n_mamba: int = 6,
        kernel_size: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Channel dimensions per stage: dim, 2*dim, 4*dim, ...
        channels = [dim * (2 ** i) for i in range(n_stages)]

        # Input projection: raw waveform → first stage channels
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size, stride=2,
                                    padding=kernel_size // 2)
        self.input_norm = nn.GroupNorm(1, channels[0])
        self.input_act = nn.PReLU()

        # Encoder stages (after input projection)
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, n_stages):
            self.encoder_blocks.append(DownBlock(
                channels[i - 1], channels[i], kernel_size, n_mamba,
                d_state, d_conv, expand,
            ))

        # Bottleneck: BiMamba at deepest level
        self.bottleneck = nn.ModuleList([
            BiMambaBlock(channels[-1], d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])

        # Decoder stages (reverse order, with skip connections)
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            self.decoder_blocks.append(UpBlock(
                channels[i], channels[i - 1], channels[i - 1], kernel_size,
                n_mamba, d_state, d_conv, expand,
            ))

        # Output projection: first-stage channels → C waveforms
        self.output_conv = nn.ConvTranspose1d(
            channels[0], C, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1,
        )

        self.C = C

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate mixture into C sources.

        Args:
            mixture: [B, 1, T] or [B, T].

        Returns:
            [B, T] if C=1, [B, C, T] if C>1.
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # [B, T] → [B, 1, T]

        n_samples = mixture.shape[-1]

        # Input projection
        x = self.input_act(self.input_norm(self.input_conv(mixture)))

        # Encoder: collect skip connections
        skips = [x]
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = x.transpose(1, 2)
        for block in self.bottleneck:
            x = block(x)
        x = x.transpose(1, 2)

        # Decoder: use skip connections in reverse (skip deepest, which is bottleneck input)
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i + 2)]  # Match encoder stage
            x = block(x, skip)

        # Output projection
        estimates = self.output_conv(x)  # [B, C, T']

        # Trim to original length
        estimates = estimates[..., :n_samples]

        if self.C == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
