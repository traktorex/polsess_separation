"""TF-MossFormer separation wrapper.

TF-MossFormer (Zhao et al. 2026, arXiv:2607.21128) is a time-frequency-domain
separation model: TF-Locoformer's macaron block with its self-attention replaced
by a convolution-gated local & global attention module. No code was ever
released for it — not on arXiv, not on GitHub, not in the group's own
ClearerVoice-Studio — so this is a **re-implementation from the paper built on
the vendored TF-Locoformer skeleton**, never a vendored TF-MossFormer. Be
precise about that wherever the model is described.

Layout of this package:

  * ``tf_locoformer_blocks.py`` — Apache-2.0 copy of the vendored TF-Locoformer
    standalone separator (MERL), with the block's attention made pluggable; see
    its header for the full list of local edits and the licence pointer.
  * ``local_global_attention.py`` — the new module (``ConvGate``,
    ``WindowedMHSA``, ``LocalGlobalMHSA``), written from the paper.
  * this file — the project wrapper, matching the waveform-in / waveform-out
    contract every other model in ``models/`` obeys.

Paper sizes (Table 1; ``attention_dim`` = ``D``, heads = 4, groups = 4,
conv kernel/stride = 4/1, windows ``w_T``/``w_F`` = 31/7, gate kernel 4):

    S   D= 96  blocks=4  ffn_hidden_dim=256   ->  5.92 M   (paper: 5.9 / 6.0 M)
    M   D=128  blocks=6  ffn_hidden_dim=384   -> 17.34 M   (paper: 16.9 M)
    L   D=128  blocks=9  ffn_hidden_dim=384   -> 26.00 M   (paper: 25.4 M)

The 2.5 % excess on M and L is the paper's own inconsistency, not a modelling
failure — see the reconciliation recorded in ``LocalGlobalMHSA``'s docstring.
The defaults below are size S.

STFT geometry is expressed in **samples**, not tied to ``data.sample_rate``
(same deliberate choice as ``SPMambaParams``): the paper's 8 kHz recipe is
``n_fft=128 / hop_length=64`` (16 ms / 8 ms); a 16 kHz corpus wanting the same
frame rate uses ``256 / 128`` and doubles F from 65 to 129.

Precision: like ``models/spmamba.py`` this model casts to fp32 around
``torch.stft`` / ``torch.complex`` / ``torch.istft`` — ``torch.complex`` rejects
bfloat16, and the training AMP policy puts this model on bf16 autocast (see
``training/trainer.py:_setup_amp``). The transformer blocks themselves still run
in the autocast dtype.
"""

import torch
import torch.nn as nn

from .local_global_attention import LocalGlobalMHSA
from .tf_locoformer_blocks import TFLocoformerSeparator


class TFMossFormer(nn.Module):
    """TF-MossFormer: convolution-gated local & global attention in the TF domain.

    Args:
        C: Output sources (1 for enhancement, 2 for separation).
        D: Embedding dimension. Also the attention dimension (the paper's
            Table 1 reports no separate value, and TF-Locoformer's own S row
            only reproduces with ``attention_dim == D``).
        num_blocks: Number of TF blocks (each = one frequency module + one
            temporal module).
        ffn_hidden_dim: Hidden dim of the Conv-SwiGLU FFNs. The block is
            macaron-style — one FFN before and one after attention — which is
            what the paper means by "Conv-SwiGLU before and after".
        conv_kernel_size: Conv1d/Deconv1d kernel in the Conv-SwiGLU FFNs (K).
        conv_stride: Conv1d/Deconv1d stride in the Conv-SwiGLU FFNs (S).
        n_heads: Attention heads, shared by the local and the global path.
        num_groups: Groups in RMSGroupNorm.
        window_t: Sliding-window size of the temporal module's local attention.
        window_f: Sliding-window size of the frequency module's local attention.
        gate_kernel_size: Conv1d kernel of the two convolution gates.
        n_fft: STFT size in samples (window length = n_fft, Hann, center=True).
        hop_length: STFT hop in samples.
        attn_dropout: Dropout on both attention paths (Q/K/V softmax and output
            projection). The Conv-SwiGLU FFNs keep dropout 0, as in the paper's
            recipe; the sweep key ``dropout`` routes here.
    """

    def __init__(
        self,
        C: int = 2,
        D: int = 96,
        num_blocks: int = 4,
        ffn_hidden_dim: int = 256,
        conv_kernel_size: int = 4,
        conv_stride: int = 1,
        n_heads: int = 4,
        num_groups: int = 4,
        window_t: int = 31,
        window_f: int = 7,
        gate_kernel_size: int = 4,
        n_fft: int = 128,
        hop_length: int = 64,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        assert n_fft % 2 == 0, f"n_fft must be even, got {n_fft}"

        self.C = C
        self.n_fft = n_fft
        self.hop_length = hop_length

        # The single architectural difference from TF-Locoformer: each path's
        # attention becomes the gated local+global cascade. Windows differ per
        # path (w_F on frequency bins, w_T on frames), so there is one builder
        # each; the separator calls a builder once per block with that path's
        # shared RoPE module.
        def build_freq_attn(pe):
            return LocalGlobalMHSA(
                D, D, n_heads, pe, window_f, gate_kernel_size, attn_dropout
            )

        def build_time_attn(pe):
            return LocalGlobalMHSA(
                D, D, n_heads, pe, window_t, gate_kernel_size, attn_dropout
            )

        self.separator = TFLocoformerSeparator(
            num_spk=C,
            n_layers=num_blocks,
            emb_dim=D,
            norm_type="rmsgroupnorm",
            num_groups=num_groups,
            tf_order="ft",  # frequency module then temporal module
            n_heads=n_heads,
            flash_attention=False,  # unavailable on the masked local path anyway
            attention_dim=D,
            pos_enc="rope",
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],  # macaron
            ffn_hidden_dim=[ffn_hidden_dim, ffn_hidden_dim],
            conv1d_kernel=conv_kernel_size,
            conv1d_shift=conv_stride,
            dropout=0.0,  # FFN dropout; attention dropout is set per builder
            attn_builder_freq=build_freq_attn,
            attn_builder_time=build_time_attn,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate a mixture into C sources.

        Args:
            mixture: [B, 1, T] or [B, T] waveform.

        Returns:
            [B, C, T] if C > 1, else [B, T]. T is preserved exactly (the SI-SDR
            loss requires it) via ``length=`` on the inverse STFT.
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # [B, T]

        n_samples = mixture.shape[1]

        # Per-utterance standard-deviation normalisation (the recipe's
        # normalize_variance), undone on the output.
        mix_std = torch.std(mixture, dim=1, keepdim=True) + 1e-8
        mixture = mixture / mix_std

        # STFT in fp32: torch.complex does not accept bf16 and ComplexHalf is
        # experimental. .float() is a no-op outside autocast.
        window = torch.hann_window(self.n_fft, device=mixture.device)
        spectrum = torch.stft(
            mixture.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=True,
        )  # [B, F, T']

        estimates = self.separator(spectrum.transpose(1, 2))  # [B, C, T', F] complex

        estimates = estimates.transpose(2, 3)  # [B, C, F, T']
        n_batch, _, n_freqs, n_frames = estimates.shape
        if estimates.dtype == torch.complex32:
            estimates = estimates.to(torch.complex64)

        waveforms = torch.istft(
            estimates.reshape(n_batch * self.C, n_freqs, n_frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            length=n_samples,
        )  # [B * C, T]
        waveforms = waveforms.view(n_batch, self.C, n_samples)

        # De-normalise
        waveforms = waveforms * mix_std.unsqueeze(1)

        if self.C == 1:
            waveforms = waveforms.squeeze(1)  # [B, T] for ES/EB enhancement

        return waveforms


__all__ = ["TFMossFormer"]
