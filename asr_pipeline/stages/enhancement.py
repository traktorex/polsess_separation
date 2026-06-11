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
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import torch

from asr_pipeline.config import EnhancementConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage


# Inputs shorter than this (samples) are passed through unenhanced — too short
# for the STFT window / a meaningful forward. (Not the same threshold as
# separation's like-valued _MIN_OVERLAP_SAMPLES; kept separate.)
_MIN_ENHANCE_SAMPLES = 256

# Fraction of a ClearVoice backend's one_time_decode_length we actually fill per
# forward, leaving headroom so we never trip its internal (bugged) segmenter.
_DECODE_WINDOW_SAFETY = 0.8


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
    result equals `process_chunk(audio)[:n]`.
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
    weights = np.maximum(weights, 1e-8)  # inaudible floor; only guards div-by-0 at uncovered edges
    return (out / weights).astype(np.float32)


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
        # decode_batch.py, crashes on long input). We keep every forward below
        # this threshold and overlap-add ourselves. Set per-model in load()
        # (FRCRN 120 s, MossFormerGAN 10 s, MossFormer2 20 s); None until then.
        self._decode_window_s: float | None = None

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
        # Direct attribute access (not getattr-with-default) so a renamed/moved
        # upstream field fails loud at load rather than silently using a window
        # that may exceed the real decode limit.
        self._decode_window_s = float(sm.args.one_time_decode_length)

    def unload(self) -> None:
        self._cv = None
        self._device = None

    @torch.no_grad()
    def enhance(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._cv is None:
            raise RuntimeError("ClearVoiceBackend.enhance called before load().")
        if len(audio_np) < _MIN_ENHANCE_SAMPLES:
            return audio_np.astype(np.float32)

        orig_len = len(audio_np)
        x = audio_np.astype(np.float32)
        # soxr_hq (librosa) is inherited from the batch-script lineage
        # (scripts/enhance_clarin_debleed.py) the 48 kHz checkpoint was
        # characterised against; deliberately NOT unified with the separator's
        # torchaudio resampler — the two are not bit-identical and swapping
        # would perturb the 48 kHz numbers.
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
        # ClearVoice returns (1, T) for these single-output SE models; squeeze → (T,).
        return np.asarray(out, dtype=np.float32).squeeze()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        """Enhance native-rate mono audio, overlap-adding long input.

        ClearVoice's own long-audio segmentation (input longer than
        `one_time_decode_length`) is bugged, so we keep every forward under
        that threshold and stitch with a canonical 50 %-hop Hann window
        (constant overlap-add) via `_hann_overlap_add`. Short input reduces
        to a single forward.
        """
        sr = self.native_sample_rate
        # stay under the decode threshold; ≥ 1 s
        window = max(int(self._decode_window_s * _DECODE_WINDOW_SAFETY * sr), sr)
        return _hann_overlap_add(x, window, self._cv_call)


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------


# Also imported by helper scripts (scripts/enhance_clarin_debleed.py,
# scripts/transcribe_clarin_2speakers.py) — keep the keys stable.
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
        self._backend: _ClearVoiceBackend | None = None

    def load(self, device: torch.device) -> None:
        if self.config.backend in _CLEARVOICE_BACKENDS:
            model_name, native_sr = _CLEARVOICE_BACKENDS[self.config.backend]
            backend = _ClearVoiceBackend(model_name, native_sr)
        else:
            raise ValueError(
                f"Unknown enhancement backend: {self.config.backend!r}"
            )
        backend.load(device)
        self._backend = backend

    def load_signature(self) -> tuple:
        # Which model gets loaded depends on `backend`. ClearerVoice backends
        # self-download by name, so the backend key is the whole identity.
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
