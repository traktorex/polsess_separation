"""SPMamba model - Speech Processing Mamba for speech separation.

Based on: "SPMamba: State-space model is all you need in speech separation"
GitHub: https://github.com/JusperLee/SPMamba

Key architecture features:
- STFT-based encoding/decoding (frequency-domain processing)
- GridNet blocks with bidirectional Mamba instead of LSTM
- Multi-head attention for frequency modeling
- Linear O(N) complexity compared to quadratic attention

Note: Requires mamba-ssm library (Linux + CUDA only).
      Install: pip install mamba-ssm torch_complex
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

# Mamba dependencies (Linux + CUDA only)
_MAMBA_WARNING_SHOWN = False  # Module-level flag to show warning only once
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.block import Block
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    MAMBA_AVAILABLE = True
except ImportError as e:
    MAMBA_AVAILABLE = False
    if not _MAMBA_WARNING_SHOWN:
        print(f"Warning: mamba-ssm not available. SPMamba model will not work. Error: {e}")
        _MAMBA_WARNING_SHOWN = True


class MambaBlock(nn.Module):
    """Bidirectional Mamba block for sequence processing.

    Implements forward and backward Mamba blocks for bidirectional processing.
    Uses state-space models with O(N) complexity instead of O(N^2) attention.

    Args:
        in_channels: Input feature dimension
        n_layer: Number of Mamba layers (default: 1)
        bidirectional: Whether to use bidirectional processing (default: False)
    """

    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for SPMamba model. "
                "Install with: pip install mamba-ssm (Linux + CUDA only)"
            )

        # Forward direction Mamba blocks
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    mlp_cls=nn.Identity,  # No MLP layer needed for SPMamba
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )

        # Backward direction Mamba blocks (if bidirectional)
        self.backward_blocks = None
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        mlp_cls=nn.Identity,  # No MLP layer needed for SPMamba
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        # Initialize weights
        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        """
        Args:
            input: [B, T, C] tensor

        Returns:
            output: [B, T, C*2] if bidirectional else [B, T, C]
        """
        # Forward direction
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        # Backward direction (if enabled)
        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f
            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        return residual


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
    """GridNet block with Mamba-based intra/inter frame processing and attention.

    Processes time-frequency features with:
    1. Intra-frame: Process each time frame independently (frequency modeling)
    2. Inter-frame: Process across time frames (temporal modeling)
    3. Multi-head attention: Capture long-range dependencies

    Args:
        emb_dim: Embedding dimension
        emb_ks: Embedding kernel size for unfolding
        emb_hs: Embedding hop size for unfolding
        n_freqs: Number of frequency bins
        hidden_channels: Mamba hidden dimension
        n_head: Number of attention heads
        approx_qk_dim: Approximate Q/K dimension for attention
        activation: Activation function (default: prelu)
        eps: Epsilon for numerical stability
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
    ):
        super().__init__()
        in_channels = emb_dim * emb_ks

        # Intra-frame processing (frequency modeling)
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_mamba = MambaBlock(in_channels, 1, bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # Inter-frame processing (temporal modeling)
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_mamba = MambaBlock(in_channels, 1, bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # Multi-head attention
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


class SPMamba(nn.Module):
    """Speech Processing Mamba model for speech separation/enhancement.

    STFT-based architecture with Mamba blocks for efficient sequence modeling.
    Replaces LSTM with Mamba state-space models for O(N) complexity.

    Args:
        input_dim: STFT input dimension (not used, kept for compatibility)
        n_srcs: Number of output sources (1 for enhancement, 2 for separation)
        n_fft: FFT size (default: 256)
        stride: STFT hop length (default: 64)
        window: Window function (default: hann)
        n_layers: Number of GridNet blocks (default: 6)
        lstm_hidden_units: Hidden dimension (misleading name, for Mamba blocks)
        attn_n_head: Number of attention heads (default: 4)
        attn_approx_qk_dim: Approximate Q/K dimension for attention
        emb_dim: Embedding dimension (default: 16)
        emb_ks: Embedding kernel size (default: 4)
        emb_hs: Embedding hop size (default: 1)
        activation: Activation function (default: prelu)
        eps: Epsilon for numerical stability
        sample_rate: Audio sample rate (default: 16000)
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
        sample_rate=16000,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for SPMamba model. "
                "Install with: pip install mamba-ssm (Linux + CUDA only)"
            )

        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_fft = n_fft
        self.stride = stride
        self.window = window
        self.sample_rate = sample_rate

        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        # STFT will be done in forward pass using torch.stft
        # Input: 2 channels (real + imaginary)

        # Initial convolution embedding
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),  # 2 channels for real/imag
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        # GridNet blocks
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

        # STFT encoding
        window_tensor = torch.hann_window(self.n_fft, device=mixture.device)
        spec = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=window_tensor,
            return_complex=True,
            center=True,
        )  # [B, F, T]

        # Separate real and imaginary parts
        batch = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        batch = batch.transpose(2, 3)  # [B, 2, T, F]

        n_batch, _, n_frames, n_freqs = batch.shape

        # Process through network
        batch = self.conv(batch)
        for block in self.blocks:
            batch = block(batch)
        batch = self.deconv(batch)

        # Reshape for ISTFT: [B, n_srcs, 2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])

        # Convert back to complex and apply ISTFT
        estimates = []
        for src in range(self.n_srcs):
            src_spec = torch.complex(batch[:, src, 0], batch[:, src, 1])  # [B, T, F]
            src_spec = src_spec.transpose(1, 2)  # [B, F, T]

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
