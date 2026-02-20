"""TIGER model - Time-frequency Interleaved Gain Extraction and Reconstruction.

Based on: "TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction
for Efficient Speech Separation" (ICLR 2025)
Paper: https://arxiv.org/abs/2410.01469
GitHub: https://github.com/JusperLee/TIGER

Architecture:
    STFT → Band-split → Separator (B × shared FFI blocks) → Band-restore masks → iSTFT

FFI block:
    Frequency path: MSA (multi-scale conv U-Net) + F³A (full-freq-frame attention)
    Frame path:     MSA (multi-scale conv U-Net) + F³A (full-freq-frame attention)

Key properties:
    - ~0.70M parameters at 8kHz / ~0.82M at 16kHz (matches paper)
    - Pure PyTorch — no Mamba or other exotic dependencies
    - Band-split adapts dynamically to sample_rate (fine-grained at low freqs)
    - Same I/O convention as SPMamba: [B, T] → [B, T] or [B, n_srcs, T]
"""

import inspect
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inlined from look2hear/layers/normalizations.py
# ---------------------------------------------------------------------------

class LayerNormalization4D(nn.Module):
    """Layer norm for 4D tensors of shape [B, C, T, F] or [B, C, F, T]."""

    def __init__(self, input_dimension, eps: float = 1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu) / std) * self.gamma + self.beta


# ---------------------------------------------------------------------------
# Basic conv building blocks
# ---------------------------------------------------------------------------

def _glob_ln(n_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, n_channels, eps=1e-8)


class ConvNormAct(nn.Module):
    """Conv1d + GroupNorm + PReLU."""

    def __init__(self, n_in, n_out, k_size, stride=1, groups=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv1d(n_in, n_out, k_size, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = _glob_ln(n_out)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvNorm(nn.Module):
    """Conv1d + GroupNorm (no activation)."""

    def __init__(self, n_in, n_out, k_size, stride=1, groups=1, bias=True):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv1d(n_in, n_out, k_size, stride=stride, padding=padding,
                              bias=bias, groups=groups)
        self.norm = _glob_ln(n_out)

    def forward(self, x):
        return self.norm(self.conv(x))


class DilatedConvNorm(nn.Module):
    """Dilated Conv1d + GroupNorm."""

    def __init__(self, n_in, n_out, k_size, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, k_size, stride=stride,
                              dilation=dilation,
                              padding=((k_size - 1) // 2) * dilation,
                              groups=groups)
        self.norm = _glob_ln(n_out)

    def forward(self, x):
        return self.norm(self.conv(x))


class ATTConvActNorm(nn.Module):
    """Conv (1D or 2D) + activation + LayerNormalization4D.

    Used as Q/K/V projections inside the F³A attention module.
    """

    def __init__(self, in_chan, out_chan, kernel_size, n_freqs,
                 act_type="prelu", norm_type="LayerNormalization4D",
                 is2d=False, bias=True):
        super().__init__()
        conv_cls = nn.Conv2d if is2d else nn.Conv1d
        self.conv = conv_cls(in_chan, out_chan, kernel_size, padding="same", bias=bias)
        self.act = nn.PReLU()
        # norm_type is always LayerNormalization4D in TIGER
        self.norm = LayerNormalization4D((out_chan, n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.conv(x)))


# ---------------------------------------------------------------------------
# MSA module building blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """MLP with depthwise conv — used inside UConvBlock for global features."""

    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(hidden_size, hidden_size, 5, 1, 2, bias=True,
                                groups=hidden_size)
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InjectionMultiSum(nn.Module):
    """Selective attention (SA) gate: local * sigmoid(global) + global.

    Implements the f(x, y, z) = σ(x) ⊙ y + z function from the paper.
    Used in both the fusing and decoding stages of MSA.
    """

    def __init__(self, inp, oup, kernel=1):
        super().__init__()
        groups = inp if inp == oup else 1
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        Args:
            x_l: local features  [B, N, T_local]
            x_g: global features [B, N, T_global]
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")
        global_feat = F.interpolate(self.global_embedding(x_g), size=T, mode="nearest")
        return local_feat * sig_act + global_feat


class UConvBlock(nn.Module):
    """Multi-Scale Selective Attention (MSA) module.

    Implements the encode → fuse → decode U-Net structure from Section 3.3.1.
    Operates along either the frequency dimension K or the time dimension T.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1)
        self.depth = upsampling_depth

        # Encoding: progressive downsampling
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, k_size=5,
                                           stride=1, groups=in_channels, dilation=1))
        for _ in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, k_size=5,
                                               stride=2, groups=in_channels, dilation=1))

        # Fusing: local+global selective attention at each scale
        self.loc_glo_fus = nn.ModuleList(
            [InjectionMultiSum(in_channels, in_channels) for _ in range(upsampling_depth)]
        )

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)

        # Decoding: bottom-up reconstruction
        self.last_layer = nn.ModuleList(
            [InjectionMultiSum(in_channels, in_channels, 5)
             for _ in range(self.depth - 1)]
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, dim]  where dim is K (freq path) or T (frame path)
        Returns:
            [B, N, dim]
        """
        residual = x.clone()

        # Project to higher dim
        output1 = self.proj_1x1(x)

        # Encode: progressive downsampling
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            output.append(self.spp_dw[k](output[-1]))

        # Global features: sum all scales (pooled to coarsest resolution)
        global_f = torch.zeros_like(output[-1])
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(fea, output[-1].shape[-1])
        global_f = self.globalatt(global_f)

        # Fuse local + global at each scale
        x_fused = [self.loc_glo_fus[i](output[i], global_f) for i in range(self.depth)]

        # Decode: bottom-up
        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        return self.res_conv(expanded) + residual


# ---------------------------------------------------------------------------
# F³A module (Full-Frequency-Frame Attention)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention2D(nn.Module):
    """Full-Frequency-Frame Attention (F³A) module.

    Operates on [B, C, T, F] tensors. Merges the full time (or freq) length
    into the embedding dim so attention captures global context across all
    sub-bands (or all frames) simultaneously.

    Args:
        in_chan:   Feature channels C
        n_freqs:   Size of the last dimension (K or 1 after reshape)
        n_head:    Number of attention heads A
        hid_chan:  Hidden channel E per head (for Q and K projections)
        dim:       Which dimension to transpose before attention (3 or 4)
    """

    def __init__(self, in_chan, n_freqs, n_head=4, hid_chan=4,
                 act_type="prelu", norm_type="LayerNormalization4D", dim=3):
        super().__init__()
        assert in_chan % n_head == 0
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.dim = dim

        def _proj(out_c):
            return ATTConvActNorm(in_chan, out_c, kernel_size=1, n_freqs=n_freqs,
                                  act_type=act_type, norm_type=norm_type, is2d=True)

        self.Queries = nn.ModuleList([_proj(hid_chan) for _ in range(n_head)])
        self.Keys = nn.ModuleList([_proj(hid_chan) for _ in range(n_head)])
        self.Values = nn.ModuleList([_proj(in_chan // n_head) for _ in range(n_head)])
        self.attn_concat_proj = _proj(in_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]   # each: [B, E, T, F]
        all_K = [k(x) for k in self.Keys]
        all_V = [v(x) for v in self.Values]     # each: [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B*n_head, E, T, F]
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)  # [B*n_head, C/n_head, T, F]

        # Merge channel+freq into embedding dim → attention over time
        Q = Q.transpose(1, 2).flatten(start_dim=2)   # [B*n_head, T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)
        V = V.transpose(1, 2)                          # [B*n_head, T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)                     # [B*n_head, T, C*F/n_head]

        emb_dim = Q.shape[-1]
        attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(emb_dim)
        attn = F.softmax(attn, dim=2)
        V = torch.matmul(attn, V)                      # [B*n_head, T, C*F/n_head]
        V = V.reshape(old_shape)                        # [B*n_head, T, C/n_head, F]
        V = V.transpose(1, 2)                           # [B*n_head, C/n_head, T, F]

        emb_dim = V.shape[1]
        x = V.view([self.n_head, batch_size, emb_dim, time, freq])
        x = x.transpose(0, 1).contiguous()             # [B, n_head, C/n_head, T, F]
        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)
        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


# ---------------------------------------------------------------------------
# FFI block (Frequency-Frame Interleaved)
# ---------------------------------------------------------------------------

class Recurrent(nn.Module):
    """Separator: B iterations of shared FFI blocks.

    Each FFI block has:
      - Frequency path: UConvBlock (MSA) + MultiHeadSelfAttention2D (F³A)
      - Frame path:     UConvBlock (MSA) + MultiHeadSelfAttention2D (F³A)

    Args:
        out_channels:    Feature dim N
        in_channels:     Hidden dim in MSA U-Net
        nband:           Number of sub-bands K
        upsampling_depth: MSA downsampling depth D
        n_head:          F³A attention heads
        att_hid_chan:    F³A hidden channel E per head
        _iter:           Number of FFI block iterations B
    """

    def __init__(self, out_channels=128, in_channels=512, nband=8,
                 upsampling_depth=3, n_head=4, att_hid_chan=4,
                 kernel_size=8, stride=1, _iter=4):
        super().__init__()
        self.nband = nband

        self.freq_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head,
                                     hid_chan=att_hid_chan, dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])

        self.frame_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head,
                                     hid_chan=att_hid_chan, dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])

        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, nband, N, T]
        Returns:
            [B, nband, N, T]
        """
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, nband, T]
        mixture = x.clone()

        for i in range(self.iter):
            if i == 0:
                x = self._ffi_block(x, B, nband, N, T)
            else:
                x = self._ffi_block(self.concat_block(mixture + x), B, nband, N, T)

        return x.permute(0, 2, 1, 3).contiguous()  # [B, nband, N, T]

    def _ffi_block(self, x, B, nband, N, T):
        """One FFI block: frequency path then frame path."""
        # --- Frequency path ---
        residual = x.clone()
        # Reshape to process across sub-bands: treat T as batch
        x_t = x.permute(0, 3, 1, 2).contiguous()          # [B, T, N, nband]
        freq_fea = self.freq_path[0](x_t.view(B * T, N, nband))  # [B*T, N, nband]
        freq_fea = freq_fea.view(B, T, N, nband).permute(0, 2, 1, 3)  # [B, N, T, nband]
        freq_fea = self.freq_path[1](freq_fea)             # F³A: [B, N, T, nband]
        freq_fea = self.freq_path[2](freq_fea)             # LayerNorm
        freq_fea = freq_fea.permute(0, 1, 3, 2).contiguous()  # [B, N, nband, T]
        x = freq_fea + residual

        # --- Frame path ---
        residual = x.clone()
        # Reshape to process across frames: treat nband as batch
        x_b = x.permute(0, 2, 1, 3).contiguous()          # [B, nband, N, T]
        frame_fea = self.frame_path[0](x_b.view(B * nband, N, T))  # [B*nband, N, T]
        frame_fea = frame_fea.view(B, nband, N, T).permute(0, 2, 1, 3)  # [B, N, nband, T]
        frame_fea = self.frame_path[1](frame_fea)          # F³A: [B, N, nband, T]
        frame_fea = self.frame_path[2](frame_fea)          # LayerNorm
        x = frame_fea + residual

        return x


# ---------------------------------------------------------------------------
# Top-level TIGER model
# ---------------------------------------------------------------------------

class TIGER(nn.Module):
    """TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction.

    Args:
        out_channels:    Feature dimension N per sub-band (default: 128)
        in_channels:     Hidden dimension in MSA U-Net (default: 512)
        num_blocks:      FFI block iterations B (4=small, 8=large)
        upsampling_depth: MSA downsampling depth D (default: 4)
        att_n_head:      F³A attention heads A (default: 4)
        att_hid_chan:    F³A hidden channel E per head (default: 4)
        n_fft:           STFT window size (default: 320 → 40ms at 8kHz)
        hop_length:      STFT hop length (default: 80 → 10ms at 8kHz)
        n_srcs:          Number of output sources (1=enhancement, 2=separation)
        sample_rate:     Audio sample rate in Hz (default: 8000 for PolSESS)
    """

    def __init__(
        self,
        out_channels: int = 128,
        in_channels: int = 256,
        num_blocks: int = 4,
        upsampling_depth: int = 5,
        att_n_head: int = 4,
        att_hid_chan: int = 4,
        n_fft: int = 320,
        hop_length: int = 80,
        n_srcs: int = 1,
        sample_rate: int = 8000,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_srcs = n_srcs
        self.sample_rate = sample_rate
        self.feature_dim = out_channels
        self.eps = torch.finfo(torch.float32).eps

        # enc_dim = number of complex frequency bins = n_fft // 2 + 1
        self.enc_dim = n_fft // 2 + 1

        # Band-split scheme: LowFreqNarrowSplit from the paper.
        # Sub-band widths are computed in Hz, then converted to bin counts.
        # At 8kHz (Nyquist=4kHz): 0-1kHz narrow (25Hz), 1-2kHz (100Hz),
        # 2-4kHz (250Hz). The 500Hz bands cover 4-8kHz which is above Nyquist
        # at 8kHz, so they will have width 0 and are excluded automatically.
        nyquist = sample_rate / 2.0
        bw25  = max(1, int(np.floor(25  / nyquist * self.enc_dim)))
        bw100 = max(1, int(np.floor(100 / nyquist * self.enc_dim)))
        bw250 = max(1, int(np.floor(250 / nyquist * self.enc_dim)))
        bw500 = max(1, int(np.floor(500 / nyquist * self.enc_dim)))

        band_width = [bw25] * 40 + [bw100] * 10 + [bw250] * 8 + [bw500] * 8
        # Trim bands that would exceed enc_dim, then add remainder
        cumsum = 0
        trimmed = []
        for bw in band_width:
            if cumsum + bw > self.enc_dim:
                remainder = self.enc_dim - cumsum
                if remainder > 0:
                    trimmed.append(remainder)
                break
            trimmed.append(bw)
            cumsum += bw
        else:
            remainder = self.enc_dim - sum(trimmed)
            if remainder > 0:
                trimmed.append(remainder)

        self.band_width = trimmed
        self.nband = len(self.band_width)

        # Per-band bottleneck: GroupNorm + Conv1d(2*bw → N)
        self.BN = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(1, bw * 2, self.eps),
                nn.Conv1d(bw * 2, out_channels, 1),
            )
            for bw in self.band_width
        ])

        # Separator
        self.separator = Recurrent(
            out_channels, in_channels, self.nband, upsampling_depth,
            att_n_head, att_hid_chan, _iter=num_blocks,
        )

        # Per-band mask estimation: PReLU + Conv1d(N → 4*bw*n_srcs)
        # Output has 4× because we produce real+imag for each of n_srcs sources,
        # then split into gating pairs (2×2) for the sigmoid-gated mask.
        self.mask = nn.ModuleList([
            nn.Sequential(
                nn.PReLU(),
                nn.Conv1d(out_channels, bw * 4 * n_srcs, 1, groups=n_srcs),
            )
            for bw in self.band_width
        ])

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate mixture into n_srcs sources.

        Args:
            mixture: [B, T] or [B, 1, T]

        Returns:
            [B, T] if n_srcs=1, [B, n_srcs, T] if n_srcs>1
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # [B, T]

        n_samples = mixture.shape[1]

        # RMS normalization (same as SPMamba)
        mix_std = torch.std(mixture, dim=1, keepdim=True) + 1e-8
        mixture = mixture / mix_std

        # STFT
        window = torch.hann_window(self.n_fft, device=mixture.device,
                                   dtype=mixture.dtype)
        spec = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )  # [B, F, T_frames]

        B = spec.shape[0]

        # Stack real+imag: [B, 2, F, T_frames]
        spec_RI = torch.stack([spec.real, spec.imag], dim=1)

        # Band-split
        subband_spec_RI = []
        subband_spec = []
        idx = 0
        for bw in self.band_width:
            subband_spec_RI.append(spec_RI[:, :, idx:idx + bw].contiguous())
            subband_spec.append(spec[:, idx:idx + bw])
            idx += bw

        # Bottleneck: [B, nband, N, T_frames]
        subband_feature = torch.stack([
            self.BN[i](subband_spec_RI[i].view(B, self.band_width[i] * 2, -1))
            for i in range(self.nband)
        ], dim=1)

        # Separator
        sep_output = self.separator(subband_feature)  # [B, nband, N, T_frames]

        # Band-restore: estimate complex masks per source
        sep_subband_spec = []
        for i in range(self.nband):
            bw = self.band_width[i]
            # [B, 2, 2, n_srcs, bw, T_frames]
            this_output = self.mask[i](sep_output[:, i]).view(
                B, 2, 2, self.n_srcs, bw, -1
            )
            # Sigmoid gating: output[:,0] * sigmoid(output[:,1])
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]  # [B, n_srcs, bw, T]
            this_mask_imag = this_mask[:, 1]

            # Force masks to sum to 1 across sources (partition of unity)
            real_sum = this_mask_real.sum(1, keepdim=True)
            imag_sum = this_mask_imag.sum(1, keepdim=True)
            this_mask_real = this_mask_real - (real_sum - 1) / self.n_srcs
            this_mask_imag = this_mask_imag - imag_sum / self.n_srcs

            # Complex multiplication: mask ⊗ subband spectrum
            sb_real = subband_spec[i].real.unsqueeze(1)  # [B, 1, bw, T]
            sb_imag = subband_spec[i].imag.unsqueeze(1)
            est_real = sb_real * this_mask_real - sb_imag * this_mask_imag
            est_imag = sb_real * this_mask_imag + sb_imag * this_mask_real
            sep_subband_spec.append(torch.complex(est_real, est_imag))

        # Concatenate sub-bands back to full spectrum: [B, n_srcs, F, T_frames]
        sep_spec = torch.cat(sep_subband_spec, dim=2)

        # iSTFT per source
        T_frames = sep_spec.shape[-1]
        estimates = []
        for src in range(self.n_srcs):
            src_wav = torch.istft(
                sep_spec[:, src],  # [B, F, T_frames]
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                length=n_samples,
            )
            estimates.append(src_wav)

        estimates = torch.stack(estimates, dim=1)  # [B, n_srcs, T]

        # Denormalize
        estimates = estimates * mix_std.unsqueeze(1)

        if self.n_srcs == 1:
            return estimates.squeeze(1)  # [B, T]
        return estimates
