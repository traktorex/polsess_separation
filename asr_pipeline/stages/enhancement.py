"""Stage 3a — full-recording enhancement with pluggable backends.

The stage enhances the full input recording in one pass; the result
lives in `ctx.enhanced_full` and is sliced per speaker by the assembler
in Stage 4.

Note: the separator in Stage 3b runs on the *original* `ctx.audio`, not
on the enhanced version. Enhancers are denoisers, not source separators
— they tend to suppress the quieter speaker in overlapping speech,
which would degrade the separator's input. Keeping the two paths
independent is intentional.

Backends are selected via `EnhancementConfig.backend`:

  - `mpsenet`            : vendored MP-SENet (`asr_pipeline.vendor.mpsenet`).
                           Tiny (~2.8 M), narrow training distribution
                           (VoiceBank+DEMAND additive noise only).
  - `frcrn_se_16k`       : ClearerVoice FRCRN_SE_16K. DNS-2020 winner,
                           native 16 kHz, ~7 M params.
  - `mossformer_gan_se_16k`: ClearerVoice MossFormerGAN_SE_16K. GAN-loss
                           training but deterministic discriminative
                           inference (same category as CMGAN). Native 16
                           kHz.
  - `mossformer2_se_48k` : ClearerVoice MossFormer2_SE_48K. Strongest of
                           the three but pays a 16↔48 kHz resampling
                           round-trip when the pipeline runs at 16 kHz.

Each backend encapsulates its own chunking and SR handling; the stage
just dispatches. ClearerVoice checkpoints self-download on first use
to a HuggingFace cache.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from asr_pipeline.config import EnhancementConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage
from asr_pipeline.vendor.mpsenet import MPNet


class _AttrDict(dict):
    """Dict that supports attribute access (needed by MPNet.__init__)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# MP-SENet backend
# ---------------------------------------------------------------------------
# STFT helpers — exact port of upstream MP-SENet `mag_pha_stft` /
# `mag_pha_istft` (compressed-magnitude + raw phase representation).


def _mpsenet_stft(audio_t: torch.Tensor, h: dict) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(h["win_size"]).to(audio_t.device)
    spec = torch.stft(
        audio_t,
        h["n_fft"],
        hop_length=h["hop_size"],
        win_length=h["win_size"],
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = spec.abs().pow(h["compress_factor"])
    pha = spec.angle()
    return mag, pha


def _mpsenet_istft(mag: torch.Tensor, pha: torch.Tensor, h: dict) -> torch.Tensor:
    mag = mag.pow(1.0 / h["compress_factor"])
    spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    window = torch.hann_window(h["win_size"]).to(spec.device)
    return torch.istft(
        spec,
        h["n_fft"],
        hop_length=h["hop_size"],
        win_length=h["win_size"],
        window=window,
        center=True,
        normalized=False,
        onesided=True,
    )


class _MPSENetBackend:
    """Vendored MP-SENet — narrow training (VoiceBank+DEMAND), tiny.

    Time-axis attention is O(T^2), so long recordings are processed via
    Hann overlap-add (chunk = `max_segment_length_s`, hop = chunk / 2).
    """

    def __init__(self, config: EnhancementConfig) -> None:
        self.config = config
        self._model: MPNet | None = None
        self._h: _AttrDict | None = None
        self._device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        config_path = Path(self.config.config_path)
        ckpt_path = Path(self.config.checkpoint_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MP-SENet config.json not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MP-SENet checkpoint not found: {ckpt_path}")

        with open(config_path) as f:
            h = _AttrDict(json.load(f))

        model = MPNet(h).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["generator"])
        model.eval()

        self._model = model
        self._h = h
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._h = None
        self._device = None

    @torch.no_grad()
    def _enhance_one_shot(self, audio_np: np.ndarray) -> np.ndarray:
        """Enhance one mono region in a single MP-SENet forward pass.

        Caller is responsible for keeping the region under
        `max_segment_length_s` — longer regions blow up MP-SENet's O(T^2)
        time-axis attention.
        """
        assert self._model is not None and self._h is not None
        if len(audio_np) < self._h["win_size"]:
            return audio_np.copy()
        audio = torch.from_numpy(audio_np).to(self._device)
        norm = torch.sqrt(audio.numel() / (audio.pow(2).sum() + 1e-12))
        audio_n = (audio * norm).unsqueeze(0)
        mag, pha = _mpsenet_stft(audio_n, self._h)
        out = self._model(mag, pha)
        if isinstance(out, (tuple, list)):
            amp_out, pha_out = out[0], out[1]
        else:
            amp_out, pha_out = out, pha
        audio_out = _mpsenet_istft(amp_out, pha_out, self._h).squeeze(0)
        audio_out = (audio_out / (norm + 1e-12)).cpu().numpy().astype(np.float32)
        if len(audio_out) < len(audio_np):
            audio_out = np.pad(audio_out, (0, len(audio_np) - len(audio_out)))
        else:
            audio_out = audio_out[: len(audio_np)]
        return audio_out

    @torch.no_grad()
    def _enhance_chunked(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Hann overlap-add MP-SENet over a long region.

        Chunk size = `max_segment_length_s`; hop = chunk / 2 (canonical
        50% COLA — the Hann window sum is exactly 1.0 in the interior, so
        we get a click-free reconstruction; head/tail are divided by their
        actual accumulated weights).
        """
        chunk_n = int(self.config.max_segment_length_s * sample_rate)
        hop_n = chunk_n // 2
        n = len(audio_np)
        if n <= chunk_n:
            return self._enhance_one_shot(audio_np)

        out = np.zeros(n, dtype=np.float32)
        weights = np.zeros(n, dtype=np.float32)
        window = np.hanning(chunk_n).astype(np.float32)

        start = 0
        while start < n:
            end = min(start + chunk_n, n)
            seg = audio_np[start:end]
            if len(seg) < self._h["win_size"]:
                chunk_out = seg.astype(np.float32)
            else:
                seg_padded = (
                    np.pad(seg, (0, chunk_n - len(seg)))
                    if len(seg) < chunk_n
                    else seg
                )
                chunk_out = self._enhance_one_shot(seg_padded)[: len(seg)]
            w = window[: len(seg)]
            out[start:end] += chunk_out * w
            weights[start:end] += w
            if end == n:
                break
            start += hop_n

        weights = np.maximum(weights, 1e-6)
        return (out / weights).astype(np.float32)

    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(audio_np) < 256:
            return audio_np
        threshold_n = int(self.config.max_segment_length_s * sample_rate)
        if len(audio_np) <= threshold_n:
            return self._enhance_one_shot(audio_np)
        return self._enhance_chunked(audio_np, sample_rate)


# ---------------------------------------------------------------------------
# ClearerVoice-Studio backend
# ---------------------------------------------------------------------------


class _ClearVoiceBackend:
    """Wrapper around any of ClearerVoice-Studio's single-output SE models.

    Resamples to the model's native rate, calls `ClearVoice` in
    tensor-to-tensor mode (`call_t2t_mode`), resamples back, then
    truncates / pads to match the original input length (the underlying
    `decode_one_audio_*` helpers pad to their chunking window and don't
    truncate themselves).
    """

    def __init__(self, model_name: str, native_sample_rate: int) -> None:
        self.model_name = model_name
        self.native_sample_rate = native_sample_rate
        self._cv = None  # ClearVoice instance
        self._device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        from clearvoice import ClearVoice

        cv = ClearVoice(
            task="speech_enhancement",
            model_names=[self.model_name],
        )
        # ClearerVoice picks its own GPU at init via `get_free_gpu`. Force
        # the model + its inference helpers onto the device our pipeline
        # is using, so we don't end up running enhancement on a different
        # GPU than the rest of the stages.
        sm = cv.models[0]
        sm.device = device
        if sm.model is not None:
            if isinstance(sm.model, torch.nn.ModuleList):
                for m in sm.model:
                    m.to(device).eval()
            else:
                sm.model.to(device).eval()
        self._cv = cv
        self._device = device

    def unload(self) -> None:
        self._cv = None
        self._device = None

    @torch.no_grad()
    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._cv is None:
            raise RuntimeError("ClearVoiceBackend.enhance called before load().")
        if len(audio_np) < 256:
            return audio_np

        orig_len = len(audio_np)
        x = audio_np.astype(np.float32)
        if sample_rate != self.native_sample_rate:
            x = librosa.resample(
                x,
                orig_sr=sample_rate,
                target_sr=self.native_sample_rate,
                res_type="soxr_hq",
            )

        batched = x[np.newaxis, :].astype(np.float32)  # (1, T)
        out = self._cv(batched)  # numpy array, shape varies
        out = np.asarray(out, dtype=np.float32).squeeze()

        if sample_rate != self.native_sample_rate:
            out = librosa.resample(
                out,
                orig_sr=self.native_sample_rate,
                target_sr=sample_rate,
                res_type="soxr_hq",
            )

        if len(out) > orig_len:
            out = out[:orig_len]
        elif len(out) < orig_len:
            out = np.pad(out, (0, orig_len - len(out)))
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# DeepFilterNet backend
# ---------------------------------------------------------------------------


def _patch_torchaudio_for_deepfilternet() -> None:
    """Inject AudioMetaData stub into `torchaudio.backend.common`.

    DeepFilterNet's `df.io` module unconditionally imports
    `from torchaudio.backend.common import AudioMetaData`, but
    torchaudio ≥ 2.2 removed that module. We never call the I/O
    helpers in `df.io` (they're only used by DFN's CLI; we feed
    `enhance()` directly with tensors), so a stub is enough to make
    the package importable.
    """
    import sys
    import types

    if "torchaudio.backend.common" in sys.modules:
        return

    class _AudioMetaDataStub:  # pragma: no cover — never instantiated
        pass

    backend_mod = sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = _AudioMetaDataStub
    backend_mod.common = common_mod
    sys.modules["torchaudio.backend"] = backend_mod
    sys.modules["torchaudio.backend.common"] = common_mod


class _DeepFilterNetBackend:
    """DeepFilterNet3 wrapper. Native 48 kHz; resamples 16 kHz pipeline
    audio in/out via soxr_hq (same as the ClearerVoice 48k path).

    DFN's tensor API is in `df.enhance.enhance(model, df_state, audio)`
    where `audio` is a 1-D torch tensor (or `(channels, T)`) at the
    model's native SR. The model and df_state are returned by
    `init_df()`, which also handles checkpoint download to
    `~/.cache/DeepFilterNet/`.
    """

    def __init__(self) -> None:
        self._model = None
        self._df_state = None
        self._device: torch.device | None = None
        self._native_sr: int | None = None

    def load(self, device: torch.device) -> None:
        _patch_torchaudio_for_deepfilternet()
        from df.enhance import init_df  # local import: avoid eager torchaudio drift

        # init_df doesn't take a device arg — it picks via CUDA_VISIBLE_DEVICES.
        # The returned model is already on whichever GPU it picked; we move
        # it to our preferred device explicitly so we don't end up split
        # across two GPUs.
        model, df_state, _ = init_df()
        model.to(device).eval()
        self._model = model
        self._df_state = df_state
        self._device = device
        self._native_sr = int(df_state.sr())

    def unload(self) -> None:
        self._model = None
        self._df_state = None
        self._device = None
        self._native_sr = None

    @torch.no_grad()
    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._model is None or self._df_state is None:
            raise RuntimeError("DeepFilterNetBackend.enhance called before load().")
        if len(audio_np) < 256:
            return audio_np
        from df.enhance import enhance as df_enhance

        orig_len = len(audio_np)
        x = audio_np.astype(np.float32)
        if sample_rate != self._native_sr:
            x = librosa.resample(
                x, orig_sr=sample_rate, target_sr=self._native_sr, res_type="soxr_hq",
            )

        # DFN's `enhance` expects audio on CPU — `df_features` internally
        # calls `df.analysis(audio.numpy())` (a Rust function) for the ERB
        # band feature, then moves the features onto the model's device.
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, T), CPU
        y_t = df_enhance(self._model, self._df_state, x_t)
        y = y_t.detach().cpu().numpy().squeeze().astype(np.float32)

        if sample_rate != self._native_sr:
            y = librosa.resample(
                y, orig_sr=self._native_sr, target_sr=sample_rate, res_type="soxr_hq",
            )

        if len(y) > orig_len:
            y = y[:orig_len]
        elif len(y) < orig_len:
            y = np.pad(y, (0, orig_len - len(y)))
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------


_CLEARVOICE_BACKENDS = {
    "frcrn_se_16k": ("FRCRN_SE_16K", 16_000),
    "mossformer_gan_se_16k": ("MossFormerGAN_SE_16K", 16_000),
    "mossformer2_se_48k": ("MossFormer2_SE_48K", 48_000),
}


class EnhancementStage(Stage):
    name = "enhancement"

    def __init__(self, config: EnhancementConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._backend: _MPSENetBackend | _ClearVoiceBackend | _DeepFilterNetBackend | None = None

    def load(self, device: torch.device) -> None:
        if self.config.backend == "mpsenet":
            backend = _MPSENetBackend(self.config)
        elif self.config.backend in _CLEARVOICE_BACKENDS:
            model_name, native_sr = _CLEARVOICE_BACKENDS[self.config.backend]
            backend = _ClearVoiceBackend(model_name, native_sr)
        elif self.config.backend == "deepfilternet":
            backend = _DeepFilterNetBackend()
        else:
            raise ValueError(
                f"Unknown enhancement backend: {self.config.backend!r}"
            )
        backend.load(device)
        self._backend = backend

    def load_signature(self) -> tuple:
        # Which model gets loaded depends on `backend`. MP-SENet additionally
        # reads its weights and architecture from disk; the other backends
        # self-download by name and ignore those fields.
        if self.config.backend == "mpsenet":
            return (
                self.config.backend,
                self.config.checkpoint_path,
                self.config.config_path,
            )
        return (self.config.backend,)

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, ctx: PipelineContext) -> None:
        if self._backend is None:
            raise RuntimeError("EnhancementStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None.")
        ctx.enhanced_full = self._backend.enhance(
            ctx.audio.astype(np.float32), ctx.sample_rate
        )

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if ctx.enhanced_full is None:
            return
        sf.write(
            artifact_dir / "enhanced_full.wav",
            ctx.enhanced_full.astype(np.float32),
            ctx.sample_rate,
        )
