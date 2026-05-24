"""Stage 3c — bandwidth extension on separated chunks.

The separator (Stage 3b) runs at 8 kHz and its outputs reach this stage
already resampled to 16 kHz numerically but with content band-limited to
0–4 kHz. Without this stage, those narrow-band chunks are spliced into
Stage 4's per-speaker streams alongside fully-wideband 16 kHz solo audio
from Stage 3a — and the spectral discontinuity at every seam can hurt
ASR even when low-band audio is fine in isolation.

This stage is OFF by default. When ``enabled: true``, it rewrites each
``s1_gated`` / ``s2_gated`` array in ``ctx.overlap_separated`` to be
wideband. Raw arrays (``s1_raw`` / ``s2_raw``) are left untouched so the
notebook can show before/after diagnostics.

Backends are selected via ``SuperResolutionConfig.backend``:

  - ``naive``  : identity pass-through. Input is already 16 kHz
                 numerically; this backend is a no-op and serves as the
                 A/B baseline against the neural backend.
  - ``ap_bwe`` : vendored AP-BWE (``asr_pipeline.vendor.ap_bwe``) — a
                 dual-stream amplitude+phase ConvNeXt model. ~22 M
                 params, fully convolutional, claims SOTA at 8→16 kHz.
                 Requires the user to download the pretrained generator
                 (see ``asr_pipeline/vendor/ap_bwe/README.md``).

Adding a new neural backend later (NVSR, AERO, ...) follows the same
pattern as ``enhancement.py``: implement ``load``/``unload``/``process``
on a small backend class and register it in ``_BACKENDS``.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from asr_pipeline.config import SuperResolutionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage


def _log(msg: str) -> None:
    """Progress message for the super-resolution stage. Writes to both
    stdout and the debug log file (mirrors the assembly stage so the
    notebook view always shows when/whether BWE actually ran).
    """
    dlog("super_resolution", msg)


# ---------------------------------------------------------------------------
# Naive (identity) backend
# ---------------------------------------------------------------------------


class _NaiveBackend:
    """Pass-through baseline. Separator output reached this stage already
    resampled to 16 kHz numerically by ``torchaudio.functional.resample``,
    so the bandwidth-extension job for the "do nothing" variant is just
    to keep that signal as-is.

    Useful for A/B framing — turning the stage on with backend=naive
    isolates the cost of plumbing (config plumbing, ctx mutation, spill)
    from the cost of the neural model.
    """

    def load(self, device: torch.device) -> None:
        return None

    def unload(self) -> None:
        return None

    def extend(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio_np


# ---------------------------------------------------------------------------
# AP-BWE backend
# ---------------------------------------------------------------------------


_AP_BWE_LR_SR = 8_000   # AP-BWE's expected narrowband input rate.
_AP_BWE_HR_SR = 16_000  # AP-BWE's expected wideband output rate.


class _APBWEBackend:
    """AP-BWE (Lu et al. 2024) wrapper.

    The model consumes (log-amp, phase) at 16 kHz SR — the upstream
    training pipeline takes a wideband signal, downsamples to 8 kHz, and
    *then* upsamples back to 16 kHz, so that the input matches the
    "narrowband content at 16 kHz numerical rate" distribution. We feed
    Stage 3b's separated chunks (which already match that distribution)
    through the same downsample→upsample roundtrip to be safe; the cost
    is negligible compared to the model itself, and it guarantees the
    input matches the training distribution exactly.
    """

    # Bundled hyperparameter config for the 8→16 kHz model. Distributed
    # alongside the vendored generator so the user only has to fetch the
    # weights from Google Drive (and not a matching config.json).
    _CONFIG_PATH = (
        Path(__file__).resolve().parent.parent
        / "vendor" / "ap_bwe" / "config_8kto16k.json"
    )

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._h = None
        self._device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        from asr_pipeline.vendor.ap_bwe import (
            APNet_BWE_Model,
            AttrDict,
        )

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"AP-BWE checkpoint not found: {ckpt_path}\n"
                f"Download `g_8kto16k` from "
                f"https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF "
                f"and place it at the above path "
                f"(or set $AP_BWE_CHECKPOINT)."
            )
        # The upstream file is distributed inside a zip archive named after
        # the original training iteration (e.g. `g_8kto16k.zip` containing
        # `g_00500000/data.pkl`, `g_00500000/data/...`). PyTorch's `.pt`
        # format is itself a zip archive, so the .zip *is* the checkpoint —
        # users who unzip it and point at the extracted directory hit
        # `IsADirectoryError` from torch.load. Catch that early with a
        # message that explains the fix.
        if ckpt_path.is_dir():
            raise IsADirectoryError(
                f"AP-BWE checkpoint path is a directory: {ckpt_path}\n"
                f"PyTorch checkpoints are zip archives — point at the .zip / .pt "
                f"file itself, not at the extracted directory. Drop the .zip "
                f"extension if present (the file's contents are loaded by "
                f"torch.load directly)."
            )
        with open(self._CONFIG_PATH) as f:
            h = AttrDict(json.load(f))

        _log(f"load: instantiating AP-BWE generator on {device}...")
        t0 = time.perf_counter()
        model = APNet_BWE_Model(h).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["generator"])
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        _log(
            f"load: AP-BWE ready ({n_params/1e6:.1f}M params, "
            f"{time.perf_counter()-t0:.2f}s) — checkpoint={ckpt_path}"
        )

        self._model = model
        self._h = h
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._h = None
        self._device = None

    @torch.no_grad()
    def extend(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        from asr_pipeline.vendor.ap_bwe import amp_pha_istft, amp_pha_stft

        if self._model is None or self._h is None:
            raise RuntimeError("APBWEBackend.extend called before load().")
        if len(audio_np) < self._h["win_size"]:
            # iSTFT would produce empty output below one frame; pass through.
            return audio_np.astype(np.float32)

        # Mirror the upstream `inference_16k.py` preprocessing exactly: the
        # narrow-band input the network was trained on is what you get when
        # you LR-resample a wideband signal and then HR-resample it back to
        # the target rate.
        orig_len = len(audio_np)
        audio_t = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0).to(
            self._device
        )
        if sample_rate != _AP_BWE_HR_SR:
            audio_t = AF.resample(audio_t, sample_rate, _AP_BWE_HR_SR)
        audio_lr = AF.resample(audio_t, _AP_BWE_HR_SR, _AP_BWE_LR_SR)
        audio_lr = AF.resample(audio_lr, _AP_BWE_LR_SR, _AP_BWE_HR_SR)
        # Trim to match the HR-rate target length (the double-resample can
        # drift by a few samples).
        target_n = audio_t.size(1)
        audio_lr = audio_lr[:, :target_n]

        amp_nb, pha_nb, _ = amp_pha_stft(
            audio_lr, self._h["n_fft"], self._h["hop_size"], self._h["win_size"]
        )
        amp_wb, pha_wb, _ = self._model(amp_nb, pha_nb)
        audio_wb = amp_pha_istft(
            amp_wb, pha_wb, self._h["n_fft"], self._h["hop_size"], self._h["win_size"]
        )

        out = audio_wb.squeeze(0).cpu().numpy().astype(np.float32)
        if sample_rate != _AP_BWE_HR_SR:
            # Pipeline rate happens to differ from AP-BWE's HR rate (unusual).
            # Downsample back via the same polyphase path the rest of the
            # pipeline uses.
            out_t = torch.from_numpy(out).unsqueeze(0)
            out_t = AF.resample(out_t, _AP_BWE_HR_SR, sample_rate)
            out = out_t.squeeze(0).numpy().astype(np.float32)

        if len(out) > orig_len:
            out = out[:orig_len]
        elif len(out) < orig_len:
            out = np.pad(out, (0, orig_len - len(out)))
        return out


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------


class SuperResolutionStage(Stage):
    name = "super_resolution"

    def __init__(self, config: SuperResolutionConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._backend: _NaiveBackend | _APBWEBackend | None = None

    def load(self, device: torch.device) -> None:
        if self.config.backend == "naive":
            _log("load: backend=naive (identity pass-through)")
            backend = _NaiveBackend()
        elif self.config.backend == "ap_bwe":
            backend = _APBWEBackend(self.config.checkpoint_path)
        else:
            raise ValueError(
                f"Unknown super_resolution backend: {self.config.backend!r}"
            )
        backend.load(device)
        self._backend = backend

    def load_signature(self) -> tuple:
        # Naive backend has no weights — backend selector alone is enough.
        # AP-BWE additionally pulls a checkpoint from disk, so changing
        # the checkpoint path must trigger a reload too.
        if self.config.backend == "ap_bwe":
            return (self.config.backend, self.config.checkpoint_path)
        return (self.config.backend,)

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, ctx: PipelineContext) -> None:
        _log("run: entered")
        if self._backend is None:
            raise RuntimeError("SuperResolutionStage.run called before load().")
        if not ctx.overlap_separated:
            _log("run: no overlap regions — nothing to extend")
            return

        sr = ctx.sample_rate
        n = len(ctx.overlap_separated)
        backend_name = self.config.backend
        # Total chunk audio duration — useful for the real-time factor in the
        # final message, and confirms at a glance how much audio was actually
        # touched (vs. the recording's full length).
        total_chunk_s = sum(
            len(e["s1_gated"]) + len(e["s2_gated"]) for e in ctx.overlap_separated
        ) / (2 * sr)
        _log(
            f"run: applying backend={backend_name!r} to {n} overlap region(s) "
            f"× 2 streams ({total_chunk_s:.1f}s of audio total)"
        )
        t0 = time.perf_counter()
        for i_ovl, entry in enumerate(ctx.overlap_separated):
            entry["s1_gated"] = self._backend.extend(entry["s1_gated"], sr)
            entry["s2_gated"] = self._backend.extend(entry["s2_gated"], sr)
            # Progress every 10 chunks so long recordings show signs of life
            # (mirrors the assembly stage's overlap-assignment loop cadence).
            if (i_ovl + 1) % 10 == 0 or i_ovl + 1 == n:
                _log(
                    f"  extended {i_ovl+1}/{n} regions "
                    f"({time.perf_counter()-t0:.1f}s elapsed)"
                )
        elapsed = time.perf_counter() - t0
        rtf = total_chunk_s / elapsed if elapsed > 0 else float("inf")
        _log(
            f"run: done in {elapsed:.2f}s ({rtf:.1f}× real-time, "
            f"backend={backend_name!r})"
        )

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        # Separation already spills `overlap_{idx}_s{1,2}.wav`. If we ran a
        # non-identity backend, write a parallel set tagged with the
        # backend name so the user can A/B by ear without re-running.
        if self.config.backend == "naive":
            return
        if not ctx.overlap_separated:
            return
        tag = self.config.backend
        for entry in ctx.overlap_separated:
            idx = entry["idx"]
            sf.write(
                artifact_dir / f"overlap_{idx}_s1_bwe_{tag}.wav",
                entry["s1_gated"].astype(np.float32),
                ctx.sample_rate,
            )
            sf.write(
                artifact_dir / f"overlap_{idx}_s2_bwe_{tag}.wav",
                entry["s2_gated"].astype(np.float32),
                ctx.sample_rate,
            )
