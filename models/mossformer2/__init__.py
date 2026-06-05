"""MossFormer2 separation wrapper.

MossFormer2 (Zhao et al. 2023, arXiv:2312.11825) is a time-domain separation
model that augments the MossFormer transformer backbone with a recurrent
gated-FSMN (GFSMN) module. This thin wrapper adapts the vendored
ClearerVoice-Studio training implementation (``models/mossformer2/mossformer2.py``)
to this project's conventions, mirroring ``models/resepformer.py``:

  * it builds the ``args``-style namespace ``MossFormer2_SS`` expects from explicit
    keyword arguments, so the config factory's ``model_class(**vars(params))`` call
    works like every other model; and
  * it reshapes the per-speaker list output into the ``[B, C, T]`` tensor the
    trainer and SI-SDR loss expect.

The underlying ``MossFormer`` owns its learnable Conv1d encoder / ConvTranspose1d
decoder (stride = ``kernel_size // 2``) and pads/trims its output back to the
input length internally, so no extra length handling is needed here.

Note on ``N``: the vendored model takes a separate encoder dimension
(``encoder_embedding_dim``) and transformer working dimension
(``mossformer_sequence_dim``), but its final decoder is hard-wired to the encoder
dimension — the two must be equal or the channel counts mismatch. The paper uses
512 for both, so this wrapper exposes a single ``N`` and feeds it to both, which
matches the paper and removes the foot-gun.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from .mossformer2 import MossFormer2_SS


class MossFormer2(nn.Module):
    """MossFormer2: GFSMN-augmented MossFormer for speech separation."""

    def __init__(
        self,
        N: int = 512,
        kernel_size: int = 16,
        C: int = 2,
        num_blocks: int = 24,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.C = C
        args = SimpleNamespace(
            encoder_embedding_dim=N,        # encoder output / masknet feature channels
            mossformer_sequence_dim=N,      # transformer + GFSMN working dim (= N, see module docstring)
            num_mossformer_layer=num_blocks,  # GFSMN depth
            encoder_kernel_size=kernel_size,  # decoder stride = kernel_size // 2
            num_spks=C,
            attn_dropout=attn_dropout,      # self-attention dropout (upstream hard-codes 0.1)
        )
        self.model = MossFormer2_SS(args)

    def forward(self, mixture):
        """Separate mixture into C sources.

        Args:
            mixture: [B, 1, T] or [B, T]

        Returns:
            estimates: [B, C, T] if C > 1, else [B, T]
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # [B, T]

        # MossFormer2_SS returns a list of C tensors, each [B, T] (length == input).
        sources = self.model(mixture)
        estimates = torch.stack(sources, dim=1)  # [B, C, T]

        if estimates.shape[1] == 1:
            estimates = estimates.squeeze(1)  # [B, T] for ES/EB enhancement

        return estimates


__all__ = ["MossFormer2"]
