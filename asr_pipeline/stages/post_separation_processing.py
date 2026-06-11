"""Stage 3c — VAD masking + optional bandwidth extension.

Inputs from Stage 3b's ``ctx.overlap_separated`` entries:
- ``s{1,2}_raw``  the separator's unmasked output (16 kHz numerical,
                  spectrally band-limited to 0–4 kHz because the
                  separator itself runs at 8 kHz)
- ``mask{1,2}``   per-stream binary VAD mask computed by 3b
- ``emit_*``      the emit-region boundaries 3b finalised

This stage is the *only* place where the VAD mask gets applied. 3b
computes the mask (it needs it anyway to decide ``snap_to_silence``
emit-region extensions) but no longer multiplies it into the streams.
Splitting compute-vs-apply lets the BWE backends see the *full*
separator output rather than the masked-with-hard-edges version — mask
edges look like discontinuities and feed ringing into BWE; running BWE
on the unmasked signal and applying the mask afterwards is cleaner.

Output: each entry's ``s{1,2}_gated`` is set to ``backend(s{1,2}_raw) *
mask{1,2}``. Stage 4 consumes only ``_gated`` arrays, so it sees
identical structure regardless of which backend ran. ``_raw`` is left
untouched so the notebook can A/B pre/post-BWE diagnostics.

Backends are selected via ``PostSeparationProcessingConfig.backend``:

  - ``naive``    : identity. ``s_gated = s_raw * mask``. The default —
                   use it when you want VAD masking only, no BWE.
  - ``ap_bwe``   : vendored AP-BWE (``asr_pipeline.vendor.ap_bwe``) — a
                   dual-stream amplitude+phase ConvNeXt model. ~22 M
                   params, fully convolutional, claims SOTA at 8→16 kHz.
                   Requires the user to download the pretrained
                   generator (see ``asr_pipeline/vendor/ap_bwe/README.md``).
  - ``flowhigh`` : FlowHigh (Yun et al., ICASSP 2025, arXiv 2501.04926).
                   Single-step flow-matching SR model from Resemble AI's
                   pip-installable fork. Native output 48 kHz —
                   downsampled to the pipeline rate inside the backend.
                   Install:  ``pip install git+https://github.com/resemble-ai/flowhigh.git@dev``
                   Checkpoints auto-download on first
                   ``FlowHighSR.from_pretrained()``.

This stage is always-on: it has no ``enabled`` flag because Stage 4
depends on ``s_gated`` being populated. Skipping it would break
assembly. To "turn off" BWE, set ``backend: naive`` — the stage still
runs, applying the mask, but no neural model touches the audio.
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

from asr_pipeline.config import PostSeparationProcessingConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage


def _log(msg: str) -> None:
    """Progress message. Writes to both stdout and the debug log file
    so notebook users see what backend ran and how long it took.
    """
    dlog("post_separation_processing", msg)


def _match_length(x: np.ndarray, n: int) -> np.ndarray:
    """Right-trim or zero-pad ``x`` to exactly ``n`` samples (tail-aligned).

    The neural backends emit audio that can drift by a handful of samples
    (STFT/iSTFT framing, resample rounding), so both the gated output (matched
    to the un-extended ``orig_len``) and the VAD mask (matched to the backend
    output length before the multiply) are reconciled through this one
    primitive. Zero-padding the mask gates out any uncovered tail.
    """
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


# ---------------------------------------------------------------------------
# Naive (identity) backend
# ---------------------------------------------------------------------------


class _NaiveBackend:
    """Pass-through baseline. Separator output reached this stage already
    resampled to 16 kHz numerically by ``torchaudio.functional.resample``,
    so the "no BWE" variant is just identity. The VAD mask is applied
    afterwards by the stage's ``run()`` loop — so ``backend: naive``
    means "VAD mask only".
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
        if len(audio_np) < self._h["n_fft"]:
            # amp_pha_stft runs torch.stft(center=True, pad_mode="reflect"),
            # which reflect-pads by n_fft//2 and so REQUIRES input length
            # > n_fft//2 — anything shorter crashes inside the STFT, not just
            # below one frame. Gate at the full n_fft (a safe superset that
            # also skips the lossy 1-2-frame region just above n_fft//2).
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

        return _match_length(out, orig_len)


# ---------------------------------------------------------------------------
# FlowHigh backend
# ---------------------------------------------------------------------------


_FLOWHIGH_TARGET_SR = 48_000  # FlowHigh always outputs at 48 kHz.
# Shortest input FlowHigh can frame. generate() upsamples to 48 kHz — a fixed
# 3× the pipeline-rate length, independent of flowhigh_input_sr — then runs
# torch.stft(n_fft=2048, center=True, pad_mode="reflect"), whose reflect-pad
# needs the 48 kHz length > n_fft//2 = 1024, i.e. pipeline-rate length > 341.
# 512 clears that with margin and matches the silero VAD all-ones boundary.
_FLOWHIGH_MIN_SAMPLES = 512


class _FlowHighBackend:
    """FlowHigh (Yun et al. 2025) wrapper.

    The Resemble AI fork exposes a single ``FlowHighSR`` class with
    ``from_pretrained(device=...)`` and ``generate(wav, sr_in, sr_out)``.
    Native output is 48 kHz; we downsample back to the pipeline rate
    inside ``extend()`` so callers see a same-rate-in/same-rate-out
    contract identical to AP-BWE.

    ``input_sr`` is configurable because the README only confirms
    12 / 16 kHz; the model accepts "any rate < 48 kHz" but 8 kHz isn't
    listed. Surfacing the knob lets the user A/B 16 (no resample-in) vs.
    8 (matches the separator's actual spectral content) by ear.

    Notes (foot-guns encoded in ``extend()``):
    - Feed ``generate()`` numpy, not a tensor. Its signature declares
      ``audio: np.ndarray`` and its scipy/librosa upsampling path operates
      on numpy; the README example's ``torchaudio.load()`` tensor triggers
      an implicit ``.numpy()`` on a CUDA tensor that fails.
    - Restore the input level. FlowHigh peak-normalises internally and emits
      output at a ~0.99 peak independent of the input, so we capture the
      input RMS and rescale the output to it — keeping the level-preserving
      "same level in, same level out" contract shared with naive / AP-BWE
      (rather than relying on Stage 4's ``overlap_rms_match_solo``). The RMS
      is measured on the *original* ``audio_np`` at ``sample_rate``, not the
      resampled-for-model copy: a 16→8 kHz downsample drops 4-8 kHz energy
      for full-band signals and would under-restore the level.
    """

    def __init__(self, input_sr: int) -> None:
        self.input_sr = input_sr
        self._model = None
        self._device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        try:
            from flowhigh import FlowHighSR
        except ImportError as e:
            raise ImportError(
                "FlowHigh backend selected but `flowhigh` package is not "
                "installed. Install with:\n"
                "    pip install git+https://github.com/resemble-ai/flowhigh.git@dev"
            ) from e

        _log(f"load: instantiating FlowHighSR on {device}...")
        t0 = time.perf_counter()
        # The README example passes the device as a string ("cuda"); pass
        # the exact torch.device string so we honour the pipeline's device
        # selection (in particular a specific GPU index).
        model = FlowHighSR.from_pretrained(device=str(device))
        try:
            n_params = sum(p.numel() for p in model.parameters())
        except Exception:
            n_params = 0
        _log(
            f"load: FlowHigh ready "
            f"({n_params/1e6:.1f}M params, {time.perf_counter()-t0:.2f}s) — "
            f"input_sr={self.input_sr}, target_sr={_FLOWHIGH_TARGET_SR}"
        )

        self._model = model
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._device = None

    @torch.no_grad()
    def extend(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("FlowHighBackend.extend called before load().")
        if len(audio_np) < _FLOWHIGH_MIN_SAMPLES:
            # Too short to frame at 48 kHz — pass through (see constant).
            return audio_np.astype(np.float32)

        orig_len = len(audio_np)

        # Feed generate() numpy, not a tensor (see class docstring Notes).
        audio_for_model = audio_np.astype(np.float32)
        if sample_rate != self.input_sr:
            # Resample on CPU via torchaudio.functional (matches the rate
            # of the rest of the pipeline's audio handling). Small input,
            # cheap.
            x_t = torch.from_numpy(audio_for_model).unsqueeze(0)
            x_t = AF.resample(x_t, sample_rate, self.input_sr)
            audio_for_model = x_t.squeeze(0).numpy().astype(np.float32)

        # Input RMS to restore on the output (FlowHigh's output level is
        # independent of the input). Measured on the original audio at
        # sample_rate — see class docstring Notes for why not the resampled copy.
        in_rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))

        wav_hr = self._model.generate(
            audio_for_model, self.input_sr, _FLOWHIGH_TARGET_SR
        )

        # Output is a torch tensor on the model's device, shape [1, T].
        if wav_hr.dim() == 2:
            wav_hr = wav_hr.squeeze(0)
        if _FLOWHIGH_TARGET_SR != sample_rate:
            wav_hr = AF.resample(
                wav_hr.unsqueeze(0), _FLOWHIGH_TARGET_SR, sample_rate
            ).squeeze(0)
        out = _match_length(wav_hr.detach().cpu().numpy().astype(np.float32), orig_len)

        # Restore the input level. Guard only out_rms (divide-by-zero): a
        # near-silent input (in_rms≈0) must map to a near-silent output, NOT be
        # left at FlowHigh's ~0.99 internal peak — so in_rms is NOT in the guard.
        out_rms = float(np.sqrt(np.mean(out ** 2))) if len(out) else 0.0
        if out_rms > 1e-8:
            out = out * (in_rms / out_rms)
            # Defensive: peak-clip protection. If the rescale pushed past
            # ±1 (shouldn't happen if input RMS < 1, but pathological
            # inputs exist), shrink to fit. Hard-clipping here would be
            # worse than a slight level miss.
            peak = float(np.max(np.abs(out)))
            if peak > 1.0:
                out = out / peak
        return out


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------


class PostSeparationProcessingStage(Stage):
    name = "post_separation_processing"

    def __init__(self, config: PostSeparationProcessingConfig) -> None:
        # Always-on stage — assembly depends on `s_gated`, so we never
        # let this be disabled at the config level. The Stage base class
        # accepts an `enabled` flag, but we don't expose it here.
        super().__init__(enabled=True)
        self.config = config
        self._backend: (
            _NaiveBackend | _APBWEBackend | _FlowHighBackend | None
        ) = None

    def load(self, device: torch.device) -> None:
        if self.config.backend == "naive":
            _log("load: backend=naive (VAD mask only, no BWE)")
            backend = _NaiveBackend()
        elif self.config.backend == "ap_bwe":
            backend = _APBWEBackend(self.config.checkpoint_path)
        elif self.config.backend == "flowhigh":
            backend = _FlowHighBackend(self.config.flowhigh_input_sr)
        else:
            raise ValueError(
                f"Unknown post_separation_processing backend: "
                f"{self.config.backend!r}"
            )
        backend.load(device)
        self._backend = backend

    def load_signature(self) -> tuple:
        # The naive backend has no per-config state beyond the backend
        # name itself. AP-BWE additionally pulls a checkpoint from disk;
        # FlowHigh's behaviour depends on the input-SR knob (changing it
        # doesn't change which weights are loaded, but it changes what gets
        # fed to the model, so include it for re-run safety).
        if self.config.backend == "ap_bwe":
            return (self.config.backend, self.config.checkpoint_path)
        if self.config.backend == "flowhigh":
            return (self.config.backend, self.config.flowhigh_input_sr)
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
            raise RuntimeError(
                "PostSeparationProcessingStage.run called before load()."
            )
        if not ctx.overlap_separated:
            _log("run: no overlap regions — nothing to do")
            return

        sr = ctx.sample_rate
        n = len(ctx.overlap_separated)
        backend_name = self.config.backend
        # Total chunk audio duration — useful for the real-time factor in the
        # final message, and confirms at a glance how much audio was actually
        # touched (vs. the recording's full length).
        total_chunk_s = sum(
            len(e["s1_raw"]) + len(e["s2_raw"]) for e in ctx.overlap_separated
        ) / (2 * sr)
        _log(
            f"run: applying backend={backend_name!r} + VAD mask to "
            f"{n} overlap region(s) × 2 streams "
            f"({total_chunk_s:.1f}s of audio total)"
        )

        # We extend the *raw* (unmasked) separator output and apply the VAD
        # mask after BWE, for two reasons:
        #   1. The mask's hard zero-to-content edges look like clicks/
        #      discontinuities to BWE, which it would try to "extend"
        #      upward into ringing.
        #   2. When a stream is entirely silent in this segment (the VAD
        #      gated everything out), feeding silence to BWE produces
        #      sawtooth-shaped artifacts because the model wasn't trained
        #      on silence. Post-BWE masking collapses those to true zeros.
        # The early-out below skips BWE entirely for fully-silent streams —
        # purely an optimisation, since the post-mask multiply would zero the
        # whole output anyway. With `backend: naive` it's also just less
        # work.
        t0 = time.perf_counter()
        n_skipped = 0
        for i_ovl, entry in enumerate(ctx.overlap_separated):
            for src_key, mask_key in (("s1", "mask1"), ("s2", "mask2")):
                raw = entry[f"{src_key}_raw"]
                mask = entry[mask_key]
                if mask.sum() == 0:
                    # Fully-silent stream for this overlap — backend output
                    # would only produce artifacts that the mask multiply
                    # would then erase. Skip and emit clean zeros.
                    entry[f"{src_key}_gated"] = np.zeros_like(raw, dtype=np.float32)
                    n_skipped += 1
                    continue
                processed = self._backend.extend(raw, sr)
                # Align mask length to backend output (STFT/iSTFT rounding
                # can drift by a handful of samples).
                mask_aligned = _match_length(mask, len(processed))
                entry[f"{src_key}_gated"] = (
                    processed * mask_aligned
                ).astype(np.float32)
            if (i_ovl + 1) % 10 == 0 or i_ovl + 1 == n:
                _log(
                    f"  processed {i_ovl+1}/{n} regions "
                    f"({time.perf_counter()-t0:.1f}s elapsed)"
                )
        elapsed = time.perf_counter() - t0
        rtf = total_chunk_s / elapsed if elapsed > 0 else float("inf")
        _log(
            f"run: done in {elapsed:.2f}s ({rtf:.1f}× real-time, "
            f"backend={backend_name!r}, {n_skipped}/{2*n} silent stream(s) skipped)"
        )

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        # Always write the final s_gated arrays — they're this stage's
        # output and the input Stage 4 will consume. For non-naive
        # backends, also tag the filename with the backend name so the
        # user can keep multiple A/B-comparison sets in one artifact dir.
        if not ctx.overlap_separated:
            return
        tag = self.config.backend
        suffix = "" if tag == "naive" else f"_bwe_{tag}"
        for entry in ctx.overlap_separated:
            idx = entry["idx"]
            sf.write(
                artifact_dir / f"overlap_{idx}_s1_gated{suffix}.wav",
                entry["s1_gated"].astype(np.float32),
                ctx.sample_rate,
            )
            sf.write(
                artifact_dir / f"overlap_{idx}_s2_gated{suffix}.wav",
                entry["s2_gated"].astype(np.float32),
                ctx.sample_rate,
            )
