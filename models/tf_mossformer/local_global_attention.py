"""Convolution-gated local & global attention — the one module TF-MossFormer adds.

Re-implemented from the paper (no code was ever released):

    Zhao, Pan, Wang, Tian, Ma, Li, "TF-MossFormer: Integrating Convolution Gated
    Local-Global Attentions for Enhanced Time-Frequency Domain Monaural Speech
    Separation", arXiv:2607.21128 (2026).

Everything else in the architecture is TF-Locoformer (Saijo et al., IWAENC 2024,
arXiv:2408.03440), whose macaron block is reused verbatim from the Apache-2.0
copy in ``tf_locoformer_blocks.py``: the entire diff between the two models is
that ``LocoformerBlock.attn`` becomes :class:`LocalGlobalMHSA` instead of the
plain ``MultiHeadSelfAttention``.

This file is original code written from the paper's prose and figures; a paper
carries no code licence, so only the MERL-derived scaffolding it plugs into is
Apache-2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tf_locoformer_blocks import MultiHeadSelfAttention

# Banded masks depend only on (sequence length, window, device) — one mask is
# shared by every layer and both attention heads of a given path, so build each
# once. Two entries per model at fixed crop length (the freq path's and the time
# path's); a new input length adds one more. Sizes are trivial: 4 KiB at F=65,
# 245 KiB at T=501.
_BAND_MASK_CACHE: dict = {}


def band_mask(length: int, window: int, device) -> torch.Tensor:
    """Boolean sliding-window mask, ``True`` where a query may attend.

    Implements the neighbourhood ``N(t)`` of the paper's Eq. (3): a symmetric
    window of ``window`` positions centred on the query, truncated at the
    sequence ends (the paper leaves the boundary undefined; truncation is what a
    banded mask does naturally). Because the window is symmetric every query
    sees at least itself, so no row is fully masked and the ``-inf`` row → NaN
    failure mode of masked softmax cannot occur here.

    Args:
        length: sequence length L (frequency bins or frames).
        window: window size w; the radius is ``(w - 1) // 2``.
        device: device the mask is built on.

    Returns:
        ``[L, L]`` bool tensor, broadcast over batch and heads by SDPA.
    """
    key = (length, window, torch.device(device))
    mask = _BAND_MASK_CACHE.get(key)
    if mask is None:
        idx = torch.arange(length, device=device)
        mask = (idx[None, :] - idx[:, None]).abs() <= (window - 1) // 2
        _BAND_MASK_CACHE[key] = mask
    return mask


class ConvGate(nn.Module):
    """The "Gate" callout of Fig. 2(c): Conv1D → Swish.

    Runs along the sequence axis of the current path (frequency bins in the
    frequency module, frames in the temporal module), the same axis convention
    the block's Conv-SwiGLU uses. Stride is 1 and the padding totals K−1, so the
    sequence length is preserved. Table 1 sets the gate output dim ``O = D`` for
    every model size, which is forced anyway: the gate is multiplied elementwise
    with an attention output living at dim D.

    Padding is split ``(K // 2, K - 1 - K // 2)``. The paper does not specify it;
    offline separation is non-causal, so a near-symmetric split is the natural
    choice and it has no parameter or FLOP impact — only a ±1-position shift of
    the gate relative to its attention.
    """

    def __init__(self, dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, out_dim, kernel_size, stride=1)
        self.act = nn.SiLU()  # Swish
        self.padding = (kernel_size // 2, kernel_size - 1 - kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] → [B, L, O]."""
        y = F.pad(x.transpose(1, 2), self.padding)  # [B, D, L + K - 1]
        y = self.act(self.conv(y))  # [B, O, L]
        return y.transpose(1, 2)  # [B, L, O]


class WindowedMHSA(MultiHeadSelfAttention):
    """Sliding-window ("local") self-attention — the paper's Eq. (3).

    Identical to the vendored global attention except that the softmax is
    restricted to a band of ``window`` positions around each query, via a
    boolean ``attn_mask``. RoPE is kept on this path too (the paper is silent on
    positional encoding; RoPE is inherited from the base and costs 0 parameters,
    so it cannot be settled by counting).

    Dense-SDPA-with-a-mask rather than an ``unfold``-gathered true windowed
    attention: at these lengths (F = 65, T = 501 at the paper's 8 kHz geometry)
    the two are numerically identical to float noise, while ``unfold``
    materialises a ``[B, h, L, w, d]`` key tensor — 31× the memory of ``k`` on
    the time path — and needs its own edge mask anyway. The block's *global*
    attention already materialises the full L×L score matrix, so the local
    branch adds no new memory regime. ``tests/test_tf_mossformer.py`` keeps the
    ``unfold`` version as the reference oracle.
    """

    def __init__(self, emb_dim, attention_dim, window, n_heads=4, dropout=0.0, pe=None):
        # flash_attention=False: an attn_mask disables the flash kernel anyway
        # (PyTorch falls back to mem-efficient/math), and forcing it on would
        # make self.sdpa_backends a lie. Irrelevant at these sequence lengths.
        super().__init__(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            dropout=dropout,
            pe=pe,
            flash_attention=False,
        )
        self.window = window

    def forward(self, input):
        """[B, L, D] → [B, L, D]."""
        query, key, value = self.get_qkv(input)  # each [B, h, L, attention_dim // h]

        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        # Bool mask (True = keep) rather than a float mask: under autocast a
        # float mask would have to carry -inf, and -1e9 survives a low-precision
        # softmax. Never pass is_causal=True together with attn_mask — that
        # combination is illegal, and the window here is symmetric anyway.
        mask = band_mask(input.shape[1], self.window, input.device)
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, h, L, -1]

        output = output.transpose(1, 2)  # [B, L, h, -1]
        output = output.reshape(output.shape[:2] + (-1,))  # [B, L, attention_dim]
        return self.aggregate_heads(output)  # [B, L, D]


class LocalGlobalMHSA(nn.Module):
    """Fig. 2(c) V1: the local → global cascade with a convolution gate on each.

        y   = LocalMHSA(x)  * ConvGate(x)
        out = GlobalMHSA(y) * ConvGate(y)

    The gates are *multiplicative on the attention output* (⊗ in the figure),
    and each gate reads the same tensor that feeds its paired attention. There
    is no normalisation inside this module — the block's single pre-norm feeds
    it — and no residual across it; the block owns the one residual.

    Only V1 is implemented. Table 3's V2 (global → local), V3 (parallel branches
    summed) and V4 (V1 without the gates) exist solely as that ablation and V1 is
    the paper's configuration, so implementing them would be speculative machinery
    for a comparison this project is not running.

    Parameter reading. This is the literal figure ("reading A" of the project's
    reconciliation): the local branch has its own Q/K/V projection *and* its own
    output projection, both gates use Table 1's K = 4, and ``attention_dim = D``
    — every value straight from Table 1, no unstated deviation. It adds
    12·D² + 2·D parameters per module and yields S / M / L = 5.92 / 17.34 /
    26.00 M against the paper's 5.9 (abstract) or 6.0 (tables) / 16.9 / 25.4 M.
    Two documented alternatives fit M and L better but require an unstated
    deviation from Table 1, and neither is exposed as a flag:

      * reading C — local branch keeps its own Q/K/V but drops the output
        projection, the gate standing in for it (11·D² + 2·D): 5.84 / 17.14 /
        25.71 M, i.e. every one of the six published numbers within 2.6 %.
      * reading B — gates use K = 3 instead of Table 1's K = 4 (10·D² + 2·D):
        5.77 / 16.94 / 25.41 M, matching M and L to 0.1–0.6 %.

    The residual is the paper's own inconsistency, not a modelling failure: its
    S row implies 11.8–13.1 D² of added machinery where its M and L rows
    independently imply 9.8–10.0 D², and it prints two different S counts (5.9 M
    in the abstract, 6.0 M in Tables 1 and 4). Reading A is the only one that
    matches the S row, and it lands within 0.1 % of the paper's own
    parameter-matched control TF-Locoformer(S*, D = 112).
    """

    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads,
        pe,
        window,
        gate_kernel_size=4,
        dropout=0.0,
    ):
        super().__init__()
        self.local_attn = WindowedMHSA(
            emb_dim,
            attention_dim=attention_dim,
            window=window,
            n_heads=n_heads,
            dropout=dropout,
            pe=pe,
        )
        self.global_attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            dropout=dropout,
            pe=pe,
        )
        # Gate output dim = emb_dim (Table 1's O = D), so each gate can multiply
        # its paired attention output elementwise.
        self.local_gate = ConvGate(emb_dim, emb_dim, gate_kernel_size)
        self.global_gate = ConvGate(emb_dim, emb_dim, gate_kernel_size)

    def forward(self, x):
        """[B, L, D] → [B, L, D]."""
        y = self.local_attn(x) * self.local_gate(x)
        return self.global_attn(y) * self.global_gate(y)


__all__ = ["band_mask", "ConvGate", "WindowedMHSA", "LocalGlobalMHSA"]
