"""SPMamba3 model - SPMamba with Mamba-3 state-space blocks.

Variant of SPMamba that replaces the Mamba-1 selective-scan blocks with Mamba-3.
Mamba-3 (Lahoti et al., 2026, arXiv:2603.15569) changes the recurrence relative
to Mamba-1:
- Exponential-trapezoidal (second-order) discretization.
- Complex-valued states realized as RoPE rotations.
- The short causal conv1d is removed (so there is no `d_conv` knob).
- QK-Norm for training stability.

Architecture is otherwise identical to SPMamba: STFT encoding, GridNet blocks
with bidirectional Mamba + multi-head attention, STFT decoding. Only the
recurrent block is swapped, so this is a like-for-like comparison against
`models/spmamba.py`. We use the SISO Mamba-3 (`is_mimo=False`), which mirrors the
single-input/single-output recurrence of Mamba-1 and runs on the pure-Triton
kernel path (no TileLang/CuTe kernels needed).

Base: "SPMamba: State-space model is all you need in speech separation"
(https://github.com/JusperLee/SPMamba).

Note: Requires Mamba-3, which is not in the PyPI mamba-ssm wheels — it must be
built from source (state-spaces/mamba `main`) and needs triton>=3.5. This lives
in the dedicated `venv_mamba3/` (the main `venv/` intentionally stays on the
Mamba-1 mamba-ssm for the existing spmamba/mamba_tasnet/dpmamba runs), so the
import below is allowed to fail silently there — the hard error is deferred to
construction.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

# Mamba-3 dependencies (Linux + CUDA only, source install required — see module
# docstring). Unlike spmamba.py we do NOT warn on import: the main venv legitimately
# lacks Mamba-3 under the two-venv setup, and a warning on every `import models`
# there would be noise. Construction raises a clear, actionable error instead.
try:
    from mamba_ssm import Mamba3
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    MAMBA3_AVAILABLE = True
except ImportError:
    MAMBA3_AVAILABLE = False


_INSTALL_HINT = (
    "mamba-ssm with Mamba3 is required for SPMamba3. Mamba-3 is not in the PyPI "
    "wheels — build from source (needs triton>=3.5):\n"
    "  MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation "
    "git+https://github.com/state-spaces/mamba.git\n"
    "Use the dedicated venv_mamba3 (see CLAUDE.md → Virtual Environments)."
)


class Mamba3Block(nn.Module):
    """Bidirectional Mamba-3 block for sequence processing.

    Drop-in replacement for spmamba.py's ``MambaBlock``: same ``[B, T, C]`` in,
    ``[B, T, C*2]`` (bidirectional) / ``[B, T, C]`` (unidirectional) out. Uses a
    plain ``RMSNorm -> Mamba3 -> residual`` pattern instead of the mamba_ssm
    ``Block`` wrapper (which was tailored to Mamba-1's fused add-norm path).

    Mamba-3's SISO Triton kernel runs in bfloat16; dtype casting is handled here
    so the block is transparent to the surrounding fp32/autocast pipeline.

    Args:
        in_channels: Input feature dimension (= Mamba-3 ``d_model``).
        n_layer: Number of Mamba-3 layers (default: 1).
        bidirectional: Whether to also run a backward pass (default: False).
        d_state: SSM state dimension (default: 64; Mamba-3's own default is 128).
        headdim: Head dimension. Mamba-3's inner dim is ``expand * in_channels``
            and must be divisible by ``headdim`` (``nheads = expand*in_channels /
            headdim``) — this is the real constraint, not ``in_channels % headdim``.
        expand: Inner-dimension expansion factor (default: 2, the Mamba-3 default).
    """

    def __init__(self, in_channels, n_layer=1, bidirectional=False,
                 d_state=64, headdim=32, expand=2):
        super().__init__()

        if not MAMBA3_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        # Mamba-3 inner dim is expand*d_model; nheads = inner_dim / headdim.
        inner_dim = expand * in_channels
        if inner_dim % headdim != 0:
            raise ValueError(
                f"expand * in_channels ({expand} * {in_channels} = {inner_dim}) "
                f"must be divisible by headdim ({headdim})."
            )

        def make_mixer():
            return Mamba3(
                d_model=in_channels, d_state=d_state, headdim=headdim,
                expand=expand, is_mimo=False, is_outproj_norm=False,
                dtype=torch.bfloat16,
            )

        # Forward direction: RMSNorm + Mamba3 per layer.
        self.forward_norms = nn.ModuleList([RMSNorm(in_channels, eps=1e-5) for _ in range(n_layer)])
        self.forward_mixers = nn.ModuleList([make_mixer() for _ in range(n_layer)])

        # Backward direction (if bidirectional).
        self.backward_norms = None
        self.backward_mixers = None
        if bidirectional:
            self.backward_norms = nn.ModuleList([RMSNorm(in_channels, eps=1e-5) for _ in range(n_layer)])
            self.backward_mixers = nn.ModuleList([make_mixer() for _ in range(n_layer)])

    def _run(self, x, norms, mixers):
        """One direction: residual stream through RMSNorm -> Mamba3 layers."""
        orig_dtype = x.dtype
        residual = x
        for norm, mixer in zip(norms, mixers):
            normed = norm(residual.to(dtype=norm.weight.dtype))
            out = mixer(normed.to(torch.bfloat16))
            residual = residual + out.to(orig_dtype)
        return residual

    def forward(self, input):
        """
        Args:
            input: [B, T, C] tensor (any dtype; cast to bfloat16 inside Mamba-3).

        Returns:
            output: [B, T, C*2] if bidirectional else [B, T, C]
        """
        forward_out = self._run(input, self.forward_norms, self.forward_mixers)

        if self.backward_norms is not None:
            backward_out = self._run(torch.flip(input, [1]), self.backward_norms, self.backward_mixers)
            backward_out = torch.flip(backward_out, [1])
            return torch.cat([forward_out, backward_out], -1)

        return forward_out


class LayerNormalization4D(nn.Module):
    """4D layer normalization for [B, C, T, F] tensors."""

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")

        mu_ = x.mean(dim=1, keepdim=True)
        std_ = torch.sqrt(x.var(dim=1, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    """4D layer normalization for complex features [B, C, T, F]."""

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")

        mu_ = x.mean(dim=(1, 3), keepdim=True)
        std_ = torch.sqrt(x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class GridNetBlock(nn.Module):
    """GridNet block with Mamba-3 intra/inter frame processing and attention.

    Processes time-frequency features with:
    1. Intra-frame: Process each time frame independently (frequency modeling)
    2. Inter-frame: Process across time frames (temporal modeling)
    3. Multi-head attention: Capture long-range dependencies

    Identical to spmamba.py's GridNetBlock except the recurrent sub-blocks are
    ``Mamba3Block`` instead of ``MambaBlock``.

    Args:
        emb_dim: Embedding dimension
        emb_ks: Embedding kernel size for unfolding
        emb_hs: Embedding hop size for unfolding
        n_freqs: Number of frequency bins
        hidden_channels: Unused; kept for API symmetry with SPMamba (Mamba-3 has
            no single hidden-units knob — capacity is set by d_state/headdim/expand).
        n_head: Number of attention heads
        approx_qk_dim: Approximate Q/K dimension for attention
        activation: Activation function (default: prelu)
        eps: Epsilon for numerical stability
        d_state: Mamba-3 SSM state dimension
        headdim: Mamba-3 head dimension
        expand: Mamba-3 inner expansion factor
    """

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        d_state=64,
        headdim=32,
        expand=2,
    ):
        super().__init__()
        in_channels = emb_dim * emb_ks

        # Intra-frame processing (frequency modeling) — Mamba-3
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_mamba = Mamba3Block(
            in_channels, 1, bidirectional=True, d_state=d_state, headdim=headdim, expand=expand,
        )
        self.intra_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # Inter-frame processing (temporal modeling) — Mamba-3
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_mamba = Mamba3Block(
            in_channels, 1, bidirectional=True, d_state=d_state, headdim=headdim, expand=expand,
        )
        self.inter_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # Multi-head attention (unchanged from SPMamba)
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(
                f"attn_conv_Q_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.PReLU() if activation == "prelu" else nn.ReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_K_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.PReLU() if activation == "prelu" else nn.ReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_V_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    nn.PReLU() if activation == "prelu" else nn.ReLU(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.PReLU() if activation == "prelu" else nn.ReLU(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F] tensor

        Returns:
            output: [B, C, T, F] tensor
        """
        B, C, old_T, old_Q = x.shape

        # Pad to be compatible with embedding kernel/stride
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # Intra-frame processing (frequency modeling)
        input_ = x
        intra_rnn = self.intra_norm(input_)
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        intra_rnn = intra_rnn.transpose(1, 2)  # [B*T, N_chunks, C*K]
        intra_rnn = self.intra_mamba(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = self.intra_linear(intra_rnn)
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()
        intra_rnn = intra_rnn + input_

        # Inter-frame processing (temporal modeling)
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        inter_rnn = inter_rnn.transpose(1, 2)  # [B*Q, N_chunks, C*K]
        inter_rnn = self.inter_mamba(inter_rnn)
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = self.inter_linear(inter_rnn)
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()
        inter_rnn = inter_rnn + input_

        # Crop to original size before attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        # Multi-head attention
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(getattr(self, f"attn_conv_Q_{ii}")(batch))
            all_K.append(getattr(self, f"attn_conv_K_{ii}")(batch))
            all_V.append(getattr(self, f"attn_conv_V_{ii}")(batch))

        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)

        Q = Q.transpose(1, 2).flatten(start_dim=2)
        K = K.transpose(1, 2).flatten(start_dim=2)
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)

        V = V.reshape(old_shape)
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])
        batch = batch.transpose(0, 1)
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])
        batch = getattr(self, "attn_concat_proj")(batch)

        out = batch + inter_rnn
        return out


class SPMamba3(nn.Module):
    """SPMamba3 - SPMamba with Mamba-3 state-space blocks.

    STFT-based architecture with Mamba-3 blocks for efficient sequence modeling.
    Replaces Mamba-1 with Mamba-3 (exp-trapezoidal discretization, complex-valued
    states via RoPE, no causal conv1d). Everything around the recurrence — STFT
    front/back end, GridNet attention, RMS normalisation — matches SPMamba.

    Args:
        input_dim: STFT input dimension (not used, kept for compatibility)
        n_srcs: Number of output sources (1 for enhancement, 2 for separation)
        n_fft: FFT size (default: 256)
        stride: STFT hop length (default: 64)
        window: Window function (default: hann)
        n_layers: Number of GridNet blocks (default: 6)
        lstm_hidden_units: Unused; kept for API/config symmetry with SPMamba.
        attn_n_head: Number of attention heads (default: 4)
        attn_approx_qk_dim: Approximate Q/K dimension for attention
        emb_dim: Embedding dimension (default: 16)
        emb_ks: Embedding kernel size (default: 4)
        emb_hs: Embedding hop size (default: 1)
        activation: Activation function (default: prelu)
        eps: Epsilon for numerical stability
        d_state: Mamba-3 SSM state dimension (default: 64)
        headdim: Mamba-3 head dimension (default: 32; expand*emb_dim*emb_ks must
            be divisible by it)
        expand: Mamba-3 inner expansion factor (default: 2)
    """

    def __init__(
        self,
        input_dim=64,
        n_srcs=1,
        n_fft=256,
        stride=64,
        window="hann",
        n_layers=6,
        lstm_hidden_units=256,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=16,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        d_state=64,
        headdim=32,
        expand=2,
    ):
        super().__init__()

        if not MAMBA3_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_fft = n_fft
        self.stride = stride
        self.window = window

        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        # Initial convolution embedding
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),  # 2 channels for real/imag
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        # GridNet blocks with Mamba-3
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                )
            )

        # Output deconvolution
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Separate mixture into n_srcs sources.

        Args:
            mixture: Input mixture, shape [B, 1, T] or [B, T]

        Returns:
            Separated sources:
            - [B, T] if n_srcs=1 (single output for enhancement)
            - [B, n_srcs, T] if n_srcs>1 (multiple sources for separation)
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # [B, T]

        n_samples = mixture.shape[1]

        # Normalize by RMS
        mix_std = torch.std(mixture, dim=1, keepdim=True) + 1e-8
        mixture = mixture / mix_std

        # STFT requires float32. .float() is a no-op when already float32.
        spectrum = torch.stft(
            mixture.float(),
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=mixture.device),
            return_complex=True,
            center=True,
        )

        # Separate real and imaginary parts
        batch = torch.stack([spectrum.real, spectrum.imag], dim=1)  # [B, 2, F, T]
        batch = batch.transpose(2, 3)  # [B, 2, T, F]

        n_batch, _, n_frames, n_freqs = batch.shape

        # Process through network (runs under autocast if AMP enabled)
        batch = self.conv(batch)
        for block in self.blocks:
            batch = block(batch)
        batch = self.deconv(batch)

        # Reshape for ISTFT: [B, n_srcs, 2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])

        # Convert back to complex and apply ISTFT (cast to float32 — torch.complex
        # doesn't support bf16, matching original's stft.inverse(input.float()))
        if batch.dtype == torch.bfloat16:
            batch = batch.float()
        window_tensor = torch.hann_window(self.n_fft, device=batch.device)
        estimates = []
        for src in range(self.n_srcs):
            src_spec = torch.complex(batch[:, src, 0], batch[:, src, 1])  # [B, T, F]
            src_spec = src_spec.transpose(1, 2)  # [B, F, T]

            if src_spec.dtype in (torch.complex32,):
                src_spec = src_spec.to(torch.complex64)
            src_wav = torch.istft(
                src_spec,
                n_fft=self.n_fft,
                hop_length=self.stride,
                win_length=self.n_fft,
                window=window_tensor,
                center=True,
                length=n_samples,
            )
            estimates.append(src_wav.unsqueeze(1))  # [B, 1, T]

        estimates = torch.cat(estimates, dim=1)  # [B, n_srcs, T]

        # Denormalize
        estimates = estimates * mix_std.unsqueeze(1)

        if self.n_srcs == 1:
            estimates = estimates.squeeze(1)  # [B, T]

        return estimates
