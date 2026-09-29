"""Amplitude/phase STFT helpers used by AP-BWE.

Lifted verbatim from upstream `datasets/dataset.py`. The model consumes the
log-magnitude (``log_amp``) and wrapped phase (``pha``) of the input
spectrogram; iSTFT reconstructs the waveform from the model's predicted
log-amplitude and phase outputs.
"""

import torch


def amp_pha_stft(audio: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = True):
    """Return (log_amp, pha, com) for a (B, T) audio tensor.

    ``log_amp`` and ``pha`` are the model's expected inputs at inference;
    ``com`` is a 2-channel real/imag stack kept for parity with upstream
    (unused by the AP-BWE inference path).
    """
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    log_amp = torch.log(torch.abs(stft_spec) + 1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack(
        (torch.exp(log_amp) * torch.cos(pha), torch.exp(log_amp) * torch.sin(pha)),
        dim=-1,
    )
    return log_amp, pha, com


def amp_pha_istft(log_amp: torch.Tensor, pha: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = True):
    """Inverse of `amp_pha_stft`. Reconstructs a (B, T) waveform from the
    model's predicted log-amplitude and phase spectrograms.
    """
    amp = torch.exp(log_amp)
    com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )
    return audio
