"""DPRNN model using SpeechBrain components."""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import (
    Encoder,
    Decoder,
    Dual_Path_Model,
    SBRNNBlock,
)


class DPRNN(nn.Module):
    """
    Dual-Path RNN for speech separation (Luo et al., 2020).

    Encoder → Dual_Path_Model → Decoder architecture for speech separation/enhancement.

    Paper: "Dual-path RNN: efficient long sequence modeling for time-domain
           single-channel speech separation" (https://arxiv.org/abs/1910.06379)

    Key architecture features:
    - Segmentation: Splits long sequences into overlapping chunks
    - Intra-chunk RNN: Processes local information within each chunk (bidirectional)
    - Inter-chunk RNN: Processes global information across chunks
    - Alternating intra/inter processing reduces complexity from O(L) to O(√L)

    Args:
        N: Encoder/decoder channels (paper uses 64)
        kernel_size: Encoder convolution kernel size
        stride: Encoder convolution stride
        C: Number of output sources (1 for enhancement, 2 for separation)
        num_layers: Number of dual-path blocks (paper uses 6)
        chunk_size: Chunk length K (paper uses 100 for window=16)
        rnn_type: RNN type (LSTM or GRU)
        hidden_size: Hidden units per direction (paper uses 128)
        num_rnn_layers: Number of RNN layers in each block
        dropout: Dropout probability
        bidirectional: Whether inter-chunk RNN is bidirectional
        norm_type: Normalization type (ln, gln, cln, bn)
    """

    def __init__(
        self,
        N: int = 64,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 1,
        num_layers: int = 6,
        chunk_size: int = 100,
        rnn_type: str = "LSTM",
        hidden_size: int = 128,
        num_rnn_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        norm_type: str = "ln",
    ):
        super().__init__()

        # Encoder: [B, 1, T] -> [B, N, L]
        self.encoder = Encoder(
            kernel_size=kernel_size, out_channels=N, in_channels=1
        )

        # Create intra-chunk RNN block (always bidirectional for local processing)
        intra_block = SBRNNBlock(
            input_size=N,
            hidden_channels=hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            bidirectional=True,  # Paper always uses bidirectional for intra-chunk
        )

        # Create inter-chunk RNN block (can be uni-directional for online processing)
        inter_block = SBRNNBlock(
            input_size=N,
            hidden_channels=hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Dual-Path RNN separator
        self.separator = Dual_Path_Model(
            in_channels=N,
            out_channels=N,
            intra_model=intra_block,
            inter_model=inter_block,
            num_layers=num_layers,
            norm=norm_type,
            K=chunk_size,
            num_spks=C,
            skip_around_intra=True,
            linear_layer_after_inter_intra=True,  # Need linear layers to project RNN output back to N dims
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
        """
        Separate mixture into C sources.

        Args:
            mixture: Input mixture, shape [B, 1, T] or [B, T]

        Returns:
            Separated sources:
            - [B, T] if C=1 (single output for enhancement)
            - [B, C, T] if C>1 (multiple sources for separation)
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)

        # Encode: [B, T] -> [B, N, L]
        mixture_w = self.encoder(mixture)

        # Separate: [B, N, L] -> [C, B, N, L]
        est_mask = self.separator(mixture_w)

        # Decode each source
        estimates = []
        for i in range(self.C):
            mask = est_mask[i]  # [B, N, L]
            masked = mixture_w * mask
            decoded = self.decoder(masked)  # [B, T]
            estimates.append(decoded.unsqueeze(1))  # [B, 1, T]

        estimates = torch.cat(estimates, dim=1)  # [B, C, T]

        if self.C == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
