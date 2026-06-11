"""RE-SepFormer using SpeechBrain's ResourceEfficientSeparator.

RE-SepFormer (Resource-Efficient Separation Transformer, Subakan et al. 2022,
arXiv:2206.09507) is a resource-efficient SepFormer variant. Instead of full
inter-chunk attention it processes non-overlapping chunks and carries an
averaged-summary memory token between them, cutting parameters and compute
substantially while staying competitive on separation quality.

This wrapper mirrors models/sepformer.py: it reuses the SAME dual_path Encoder
and Decoder as SepFormer, and swaps the mask network for SpeechBrain's
ResourceEfficientSeparator (which plays the same role Dual_Path_Model plays for
SepFormer — encoded features [B, N, L] in, masks [num_spks, B, N, L] out).

A few architecture constants are bound internally to match the official
SpeechBrain recipe (recipes/WSJ0Mix/separation/hparams/resepformer.yaml) and the
paper, rather than exposed as knobs:
  - d_model = N: the intra/inter transformers run at the encoder feature
    dimension, so their model dim must equal the encoder channel count.
  - mem_type = "av": RE-SepFormer's averaged-summary memory. Other mem_types
    invoke the memory model with an extra argument that the transformer
    sub-models do not accept, so only "av" (or None) is valid here.
  - nonlinear = "relu", unit = 256: recipe defaults. `unit` is vestigial under
    mem_type="av" with transformer sub-models (it only sizes the LSTM memory
    path), but is set to the recipe value for fidelity.

Note: ResourceEfficientSeparationPipeline deep-copies the seg (intra) and mem
(inter) sub-models internally, so num_blocks controls the number of seg copies
and (num_blocks - 1) mem copies — passing one instance of each is correct.
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import Encoder, Decoder
from speechbrain.lobes.models.resepformer import (
    ResourceEfficientSeparator,
    SBTransformerBlock_wnormandskip,
)


class RESepFormer(nn.Module):
    """RE-SepFormer: Resource-Efficient Separation Transformer (Subakan et al. 2022)."""

    def __init__(
        self,
        N: int = 128,
        kernel_size: int = 16,
        stride: int = 8,
        C: int = 2,
        causal: bool = False,
        num_blocks: int = 2,
        num_layers: int = 8,
        nhead: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        segment_size: int = 150,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        # Encoder (shared design with SepFormer)
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=N, in_channels=1)

        # Intra-chunk (seg) and inter-chunk (mem) transformers. d_model is bound
        # to N: the transformers operate on the encoder feature dimension.
        seg_model = SBTransformerBlock_wnormandskip(
            num_layers=num_layers,
            d_model=N,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            norm_before=True,
        )
        mem_model = SBTransformerBlock_wnormandskip(
            num_layers=num_layers,
            d_model=N,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            norm_before=True,
        )

        # Resource-efficient mask network. Returns masks [num_spks, B, N, L],
        # matching the shape contract SepFormer's Dual_Path_Model provides.
        self.masknet = ResourceEfficientSeparator(
            input_dim=N,
            causal=causal,
            num_spk=C,
            nonlinear="relu",
            layer=num_blocks,
            unit=256,
            segment_size=segment_size,
            dropout=dropout,
            mem_type="av",
            seg_model=seg_model,
            mem_model=mem_model,
        )

        # Decoder (shared design with SepFormer)
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
