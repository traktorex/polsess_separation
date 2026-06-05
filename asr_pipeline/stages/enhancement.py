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
from typing import Callable

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
# Shared long-audio chunking
# ---------------------------------------------------------------------------


def _hann_overlap_add(
    audio: np.ndarray,
    window_n: int,
    process_chunk: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Process `audio` in 50%-hop Hann-windowed chunks of `window_n` samples,
    overlap-adding the per-chunk outputs (canonical COLA reconstruction).

    `process_chunk(seg)` enhances one chunk and returns audio at least as long
    as `seg` (truncated to `len(seg)` here). Input shorter than `window_n` is a
    single `process_chunk` call — for one chunk the Hann weights cancel, so the
    result equals `process_chunk(audio)[:n]`. Shared by the MP-SENet and
    ClearerVoice backends, whose long-audio chunking is otherwise identical.
    """
    n = len(audio)
    if n <= window_n:
        return process_chunk(audio)[:n].astype(np.float32)
    hop = window_n // 2
    win = np.hanning(window_n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    start = 0
    while start < n:
        end = min(start + window_n, n)
        seg = audio[start:end]
        seg_out = process_chunk(seg)[: len(seg)]
        w = win[: len(seg)]
        out[start:end] += seg_out * w
        weights[start:end] += w
        if end == n:
            break
        start += hop
    weights = np.maximum(weights, 1e-8)
    return (out / weights).astype(np.float32)


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

    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(audio_np) < 256:
            return audio_np
        chunk_n = int(self.config.max_segment_length_s * sample_rate)
        if len(audio_np) <= chunk_n:
            # Short input: single forward at exact length (no chunk padding).
            return self._enhance_one_shot(audio_np)

        # Long input: Hann overlap-add. Each chunk is padded up to `chunk_n`
        # before the forward (MP-SENet's O(T^2) attention expects the trained
        # window); the helper truncates the output back to the chunk's real
        # length. The <256-sample chunk guard mirrors `_enhance_one_shot`.
        def _process(seg: np.ndarray) -> np.ndarray:
            if len(seg) < self._h["win_size"]:
                return seg.astype(np.float32)
            seg_padded = (
                np.pad(seg, (0, chunk_n - len(seg))) if len(seg) < chunk_n else seg
            )
            return self._enhance_one_shot(seg_padded)

        return _hann_overlap_add(audio_np, chunk_n, _process)


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
        # ClearVoice's `one_time_decode_length` (seconds): audio longer than
        # this triggers its internal segmented decode — which has an upstream
        # bug (`np.zeros(b, t)` instead of `np.zeros((b, t))` in
        # decode_batch.py, crashes on long input). We keep every forward
        # below this threshold and overlap-add ourselves. Read per-model in
        # load(); 15 s is a safe floor across the SE models (FRCRN 120,
        # MossFormer2 20, …).
        self._otdl_s: float = 15.0

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
        # Respect each backend's one-pass window so we chunk just under it.
        otdl = getattr(getattr(sm, "args", None), "one_time_decode_length", None)
        if otdl:
            self._otdl_s = float(otdl)

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

        out = self._enhance_native(x)

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

    def _cv_call(self, x_native: np.ndarray) -> np.ndarray:
        """One ClearVoice forward on native-rate mono audio → mono output."""
        batched = x_native[np.newaxis, :].astype(np.float32)  # (1, T)
        out = self._cv(batched)
        return np.asarray(out, dtype=np.float32).squeeze()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        """Enhance native-rate mono audio, overlap-adding long input.

        ClearVoice's own long-audio segmentation (input longer than
        `one_time_decode_length`) is bugged, so we keep every forward under
        that threshold and stitch with a canonical 50 %-hop Hann window
        (constant overlap-add) — the same `_hann_overlap_add` scheme the
        MP-SENet backend uses. Short input reduces to a single forward.
        """
        sr = self.native_sample_rate
        window = max(int(self._otdl_s * 0.8 * sr), sr)   # stay under threshold; ≥ 1 s
        return _hann_overlap_add(x, window, self._cv_call)


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
        self._backend: _MPSENetBackend | _ClearVoiceBackend | None = None

    def load(self, device: torch.device) -> None:
        if self.config.backend == "mpsenet":
            backend = _MPSENetBackend(self.config)
        elif self.config.backend in _CLEARVOICE_BACKENDS:
            model_name, native_sr = _CLEARVOICE_BACKENDS[self.config.backend]
            backend = _ClearVoiceBackend(model_name, native_sr)
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
