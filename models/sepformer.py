"""SepFormer using SpeechBrain's built-in Dual_Path_Model."""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import Encoder, Decoder, Dual_Path_Model
from speechbrain.nnet.containers import Sequential
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder


class TransformerWrapper(nn.Module):
    """Wrapper for TransformerEncoder to return only output (not attention weights)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x):
        """Forward pass returning only output, not attention weights."""
        output, _ = self.transformer(x)
        return output


class SepFormer(nn.Module):
    """SepFormer: Transformer for Speech Separation (Subakan et al. 2021)."""

    def __init__(
        self,
        N: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 2,
        causal: bool = False,
        num_blocks: int = 2,
        num_layers: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        chunk_size: int = 250,
        hop_size: int = 125,
    ):
        super().__init__()

        # Encoder
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)

        # Intra-chunk transformer
        intra_transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=nn.ReLU,
            normalize_before=True,
            causal=causal,
        )

        # Inter-chunk transformer
        inter_transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=nn.ReLU,
            normalize_before=True,
            causal=causal,
        )

        # Wrap transformers to return only output (not attention weights)
        wrapped_intra = TransformerWrapper(intra_transformer)
        wrapped_inter = TransformerWrapper(inter_transformer)

        # Dual-Path Model handles chunking, dual-path processing, overlap-add
        self.masknet = Dual_Path_Model(
            num_spks=C,
            in_channels=N,
            out_channels=d_model,
            num_layers=num_blocks,
            K=chunk_size,
            intra_model=wrapped_intra,
            inter_model=wrapped_inter,
            norm="ln",
            linear_layer_after_inter_intra=False,
            skip_around_intra=True,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=N,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, mixture):
        """Separate mixture into C sources.

        Args:
            mixture: [B, 1, T] or [B, T]

        Returns:
            estimates: [B, C, T] if C > 1, else [B, T]
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # [B, T]

        # Encode: [B, T] -> [B, N, L]
        mixture_w = self.encoder(mixture)

        # Estimate masks: [B, N, L] -> [num_spks, B, N, L]
        # Note: Dual_Path_Model returns [num_spks, B, N, L] not [B, num_spks, N, L]
        est_masks = self.masknet(mixture_w)

        # Apply masks and decode
        estimates = []
        for i in range(est_masks.shape[0]):  # num_spks is first dimension
            masked = mixture_w * est_masks[i, :, :, :]  # [B, N, L]
            decoded = self.decoder(masked)  # [B, T]
            estimates.append(decoded)

        estimates = torch.stack(estimates, dim=1)  # [B, C, T]

        if estimates.shape[1] == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
