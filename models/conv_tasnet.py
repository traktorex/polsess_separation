"""ConvTasNet model using SpeechBrain components."""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import Encoder, Decoder
from speechbrain.lobes.models.conv_tasnet import MaskNet


class ConvTasNet(nn.Module):
    """
    ConvTasNet wrapper for SpeechBrain components.

    Encoder → MaskNet → Decoder architecture for speech separation/enhancement.

    Key params: N (encoder filters), B (bottleneck), H (conv channels),
                X (blocks/repeat), R (repeats), C (output sources)
    """

    def __init__(
        self,
        N: int = 256,
        B: int = 256,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 4,
        C: int = 2,
        norm_type: str = "gLN",
        causal: bool = False,
        mask_nonlinear: str = "relu",
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()

        # Encoder: [B, 1, T] -> [B, N, L]
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)

        # Mask network: [B, N, L] -> [B, C*N, L]
        self.mask_net = MaskNet(
            N=N,
            B=B,
            H=H,
            P=P,
            X=X,
            R=R,
            C=C,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=mask_nonlinear,
        )

        # Decoder: [B, N, L] -> [B, 1, T]
        self.decoder = Decoder(
            in_channels=N,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        self.C = C
        self.N = N

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate mixture into C sources. Input: [B, 1, T] or [B, T]. Output: [B, T] (C=1) or [B, C, T] (C>1)."""
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)

        mixture_w = self.encoder(mixture)
        est_mask = self.mask_net(mixture_w)

        # MaskNet outputs [C, batch, N, L], need to transpose to [batch, C, N, L]
        est_mask = est_mask.permute(1, 0, 2, 3)

        estimates = []
        for i in range(self.C):
            mask = est_mask[:, i, :, :]
            masked = mixture_w * mask
            decoded = self.decoder(masked)
            estimates.append(decoded.unsqueeze(1))

        estimates = torch.cat(estimates, dim=1)

        if self.C == 1:
            estimates = estimates.squeeze(1)

        return estimates
