"""Stage 3b — overlap separation + VAD gating.

For each overlap region in `ctx.overlap_regions`:

1. Pick a *context window* around the overlap (``context_window_mode``).
   The window is fed to the separator; ``snap_to_vad`` and ``fixed_pad``
   widen it past the original overlap so the separator gets more context.
2. Run the separator (resample 16 k -> 8 k -> separator -> 8 k -> 16 k).
   For overlaps wider than ``overlap_add_threshold_s`` we chunk + overlap-add.
3. Volume-normalise the two outputs (``volume_normalization``).
4. VAD-gate each output stream (``vad_threshold``, ``vad_mode``) to suppress
   the residual energy the separator inevitably leaks where there's no speech.
5. Pick the *emit region* (``seam_mode``) — the time range within the
   padded window that the assembler will splice into the per-speaker
   stream. ``zero_crossing`` nudges the seam to a nearby zero crossing in
   the separator output (avoiding clicks); ``overlap_boundary`` just uses
   the original overlap boundary as-is.

The full padded-window outputs are emitted to ``ctx.overlap_separated``;
the assembler in Phase 5 will slice them down to the ``emit_*`` region.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from asr_pipeline.config import SeparationConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage


_SILERO_WINDOW = 512  # silero-vad expects exactly 512 samples per call @ 16 kHz
_MIN_OVERLAP_SAMPLES = 256  # POC's lower bound for "long enough to be worth separating"


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


def _vad_mask_silero(
    audio_16k: np.ndarray,
    vad_model,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    """Per-sample VAD mask (1 = speech, 0 = silence) at 16 kHz.

    Frames are 512 samples (silero's required window size). The mask is
    expanded back to sample resolution by repeating each frame's decision.
    """
    audio = torch.from_numpy(audio_16k).to(device)
    n = int(audio.numel())
    if n < _SILERO_WINDOW:
        return np.ones(n, dtype=np.float32)
    n_windows = n // _SILERO_WINDOW
    probs = np.zeros(n_windows, dtype=np.float32)
    vad_model.reset_states()
    with torch.no_grad():
        for i in range(n_windows):
            chunk = audio[i * _SILERO_WINDOW : (i + 1) * _SILERO_WINDOW]
            p = vad_model(chunk.unsqueeze(0), 16_000)
            probs[i] = p.item() if hasattr(p, "item") else float(p)
    mask_frames = (probs > threshold).astype(np.float32)
    full_mask = np.zeros(n, dtype=np.float32)
    for i, m in enumerate(mask_frames):
        full_mask[i * _SILERO_WINDOW : (i + 1) * _SILERO_WINDOW] = m
    full_mask[n_windows * _SILERO_WINDOW :] = 1.0  # leftover tail: keep as speech
    return full_mask


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


@dataclass
class _Window:
    pad_start_s: float
    pad_end_s: float


def _window_none(start_s: float, end_s: float) -> _Window:
    """POC behaviour: no context, window = overlap region."""
    return _Window(pad_start_s=start_s, pad_end_s=end_s)


def _window_fixed_pad(
    start_s: float,
    end_s: float,
    total_duration_s: float,
    context_pad_s: float,
) -> _Window:
    return _Window(
        pad_start_s=max(0.0, start_s - context_pad_s),
        pad_end_s=min(total_duration_s, end_s + context_pad_s),
    )


def _window_snap_to_vad(
    start_s: float,
    end_s: float,
    total_duration_s: float,
    target_total_s: float,
    audio_16k: np.ndarray,
    vad_model,
    device: torch.device,
    vad_threshold: float,
    sample_rate: int,
) -> _Window:
    """Pad symmetrically toward `target_total_s`, snap each boundary to a
    nearby VAD-silent position so the separator doesn't see a cut mid-utterance.

    If no silence exists in the candidate pad range, falls back to the
    fixed-pad target boundary on that side.
    """
    overlap_dur = end_s - start_s
    available_pad = max(0.0, target_total_s - overlap_dur)
    max_pad_each_side = available_pad / 2.0
    target_pad_start_s = max(0.0, start_s - max_pad_each_side)
    target_pad_end_s = min(total_duration_s, end_s + max_pad_each_side)

    if target_pad_start_s >= start_s and target_pad_end_s <= end_s:
        return _Window(pad_start_s=start_s, pad_end_s=end_s)

    # Compute VAD on the candidate pad regions only (cheaper than the whole mix).
    lo = int(target_pad_start_s * sample_rate)
    hi = int(target_pad_end_s * sample_rate)
    region = audio_16k[lo:hi]
    mask = _vad_mask_silero(region, vad_model, device, vad_threshold)

    # Left side: candidate range = [target_pad_start, start]. Among silent
    # samples there, prefer the earliest (max pad with silent boundary).
    left_lo_in_region = 0
    left_hi_in_region = int(start_s * sample_rate) - lo
    pad_start_s = target_pad_start_s
    if left_hi_in_region > left_lo_in_region:
        left_mask = mask[left_lo_in_region:left_hi_in_region]
        silent_idx = np.where(left_mask < 0.5)[0]
        if len(silent_idx) > 0:
            pad_start_s = (lo + silent_idx[0]) / sample_rate

    # Right side: candidate range = [end, target_pad_end]. Among silent
    # samples, prefer the latest (max pad with silent boundary).
    right_lo_in_region = int(end_s * sample_rate) - lo
    right_hi_in_region = hi - lo
    pad_end_s = target_pad_end_s
    if right_hi_in_region > right_lo_in_region:
        right_mask = mask[right_lo_in_region:right_hi_in_region]
        silent_idx = np.where(right_mask < 0.5)[0]
        if len(silent_idx) > 0:
            pad_end_s = (lo + right_lo_in_region + silent_idx[-1]) / sample_rate

    return _Window(pad_start_s=pad_start_s, pad_end_s=pad_end_s)


# ---------------------------------------------------------------------------
# Separator inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _separate_single(
    mix_16k: np.ndarray,
    separator: torch.nn.Module,
    device: torch.device,
    sr_pipeline: int,
    sr_separator: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the separator on one mono 16 kHz mixture. Returns two 16 kHz streams."""
    orig_len = len(mix_16k)
    audio_t = torch.from_numpy(mix_16k).unsqueeze(0)
    audio_lo = AF.resample(audio_t, sr_pipeline, sr_separator).to(device)
    est = separator(audio_lo)  # [1, 2, T_lo]
    s1_lo = est[:, 0, :].cpu()
    s2_lo = est[:, 1, :].cpu()
    s1_hi = AF.resample(s1_lo, sr_separator, sr_pipeline).squeeze(0).numpy().astype(np.float32)
    s2_hi = AF.resample(s2_lo, sr_separator, sr_pipeline).squeeze(0).numpy().astype(np.float32)

    def _fix(arr: np.ndarray) -> np.ndarray:
        if len(arr) < orig_len:
            return np.pad(arr, (0, orig_len - len(arr)))
        return arr[:orig_len]

    return _fix(s1_hi), _fix(s2_hi)


def _pit_swap_if_needed(
    s1_prev: np.ndarray, s2_prev: np.ndarray,
    s1_cur: np.ndarray, s2_cur: np.ndarray,
    overlap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the two output streams of the current chunk with the previous
    chunk's stream ordering, using cosine similarity on the chunk-overlap region.
    """
    if overlap_samples == 0:
        return s1_cur, s2_cur
    prev_tail_1 = s1_prev[-overlap_samples:]
    prev_tail_2 = s2_prev[-overlap_samples:]
    cur_head_1 = s1_cur[:overlap_samples]
    cur_head_2 = s2_cur[:overlap_samples]

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    straight = _cos(prev_tail_1, cur_head_1) + _cos(prev_tail_2, cur_head_2)
    swapped = _cos(prev_tail_1, cur_head_2) + _cos(prev_tail_2, cur_head_1)
    if swapped > straight:
        return s2_cur, s1_cur
    return s1_cur, s2_cur


def _separate_overlap_add(
    mix_16k: np.ndarray,
    separator: torch.nn.Module,
    device: torch.device,
    sr_pipeline: int,
    sr_separator: int,
    chunk_length_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-add separation for long mixtures.

    Splits into 50%-overlapping Hann-windowed chunks of `chunk_length_s` each,
    runs the separator on each chunk, PIT-aligns successive chunks, and sums
    with the Hann window for COLA reconstruction.
    """
    chunk_samples = int(chunk_length_s * sr_pipeline)
    hop = chunk_samples // 2
    n = len(mix_16k)
    if n <= chunk_samples:
        return _separate_single(mix_16k, separator, device, sr_pipeline, sr_separator)

    out1 = np.zeros(n, dtype=np.float32)
    out2 = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    window = np.hanning(chunk_samples).astype(np.float32)

    prev_s1: Optional[np.ndarray] = None
    prev_s2: Optional[np.ndarray] = None

    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        seg = mix_16k[start:end]
        # Pad short final chunk so the separator receives the expected length.
        pad = chunk_samples - len(seg)
        if pad > 0:
            seg_padded = np.pad(seg, (0, pad))
        else:
            seg_padded = seg
        s1, s2 = _separate_single(seg_padded, separator, device, sr_pipeline, sr_separator)
        s1 = s1[: len(seg)]
        s2 = s2[: len(seg)]
        if prev_s1 is not None:
            s1, s2 = _pit_swap_if_needed(
                prev_s1, prev_s2, s1, s2, overlap_samples=hop
            )
        win = window[: len(seg)]
        out1[start:end] += s1 * win
        out2[start:end] += s2 * win
        weights[start:end] += win
        prev_s1, prev_s2 = s1, s2
        if end == n:
            break
        start += hop

    weights = np.maximum(weights, 1e-6)
    return out1 / weights, out2 / weights


# ---------------------------------------------------------------------------
# Volume normalisation
# ---------------------------------------------------------------------------


def _volume_normalise(
    s1: np.ndarray, s2: np.ndarray, mix: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the configured volume-normalisation strategy. Returns (s1, s2, applied_scale)."""
    if mode == "none":
        return s1, s2, 1.0
    if mode == "sum_equals_mix":
        combined = s1 + s2
        mix_rms = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
        combined_rms = float(np.sqrt(np.mean(combined.astype(np.float64) ** 2)))
        if combined_rms < 1e-9:
            return s1, s2, 1.0
        alpha = mix_rms / combined_rms
        return s1 * alpha, s2 * alpha, alpha
    if mode == "match_solo":
        raise NotImplementedError(
            "volume_normalization='match_solo' requires solo-region RMS, "
            "which is only known in Stage 4. Use per_piece_rms_norm in "
            "AssemblyConfig instead, or pick 'sum_equals_mix' / 'none'."
        )
    raise ValueError(f"Unknown volume_normalization mode: {mode!r}")


# ---------------------------------------------------------------------------
# Seam
# ---------------------------------------------------------------------------


def _pick_seam_zero_crossing(
    audio: np.ndarray,
    target_idx: int,
    search_radius_samples: int,
) -> int:
    """Return the index closest to `target_idx` (within ±`search_radius_samples`)
    where `audio` changes sign. Falls back to `target_idx` if none found.
    """
    n = len(audio)
    lo = max(1, target_idx - search_radius_samples)
    hi = min(n, target_idx + search_radius_samples)
    if hi <= lo:
        return target_idx
    region = audio[lo - 1 : hi]
    signs = np.sign(region)
    transitions = np.where(np.diff(signs) != 0)[0]
    if len(transitions) == 0:
        return target_idx
    abs_positions = transitions + lo
    nearest = abs_positions[np.argmin(np.abs(abs_positions - target_idx))]
    return int(nearest)


def _extend_start_to_silence(
    vad_mask: np.ndarray, zc_start_idx: int, max_extend_n: int
) -> int:
    """Walk backward from `zc_start_idx` looking for VAD silence in the
    separator output. Returns the latest silence position within the search
    range — i.e. the silence side of the most recent silence→speech transition
    before the boundary. Falls back to `zc_start_idx` if no silence found.

    By construction the returned index is ≤ `zc_start_idx`, so the resulting
    emit region is never narrower than zero_crossing's.
    """
    lo = max(0, zc_start_idx - max_extend_n)
    if zc_start_idx <= lo:
        return zc_start_idx
    region = vad_mask[lo:zc_start_idx]
    silent = np.where(region < 0.5)[0]
    if len(silent) == 0:
        return zc_start_idx
    return lo + int(silent[-1])


def _extend_end_to_silence(
    vad_mask: np.ndarray, zc_end_idx: int, max_extend_n: int
) -> int:
    """Walk forward from `zc_end_idx` looking for VAD silence in the separator
    output. Returns the first silence position within the search range — i.e.
    the silence side of the first speech→silence transition after the boundary.
    Falls back to `zc_end_idx` if no silence found.

    By construction the returned index is ≥ `zc_end_idx`, so the resulting
    emit region is never narrower than zero_crossing's.
    """
    n = len(vad_mask)
    hi = min(n, zc_end_idx + max_extend_n)
    if hi <= zc_end_idx:
        return zc_end_idx
    region = vad_mask[zc_end_idx:hi]
    silent = np.where(region < 0.5)[0]
    if len(silent) == 0:
        return zc_end_idx
    return zc_end_idx + int(silent[0])


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SeparationStage(Stage):
    name = "separation"

    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._separator: Optional[torch.nn.Module] = None
        self._vad: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        # Separator — the one seam back to the parent project. When the
        # package is lifted into CLARIN this is the only line to replace.
        from utils.model_utils import load_model_for_inference

        separator, _ckpt = load_model_for_inference(
            self.config.checkpoint_path, device=str(device)
        )
        separator.eval()

        if self.config.vad_mode == "silero":
            vad_model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            vad_model = vad_model.to(device)
        elif self.config.vad_mode == "energy":
            raise NotImplementedError(
                "vad_mode='energy' is not implemented yet; use 'silero'."
            )
        else:
            raise ValueError(f"Unknown vad_mode: {self.config.vad_mode!r}")

        self._separator = separator
        self._vad = vad_model
        self._device = device

    def unload(self) -> None:
        self._separator = None
        self._vad = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        if self._separator is None or self._vad is None:
            raise RuntimeError("SeparationStage.run called before load().")
        if ctx.overlap_regions is None or ctx.audio is None or ctx.diarization is None:
            raise RuntimeError(
                "SeparationStage.run requires audio + overlap_regions + diarization."
            )

        cfg = self.config
        sr = ctx.sample_rate
        total_dur = ctx.diarization.total_duration_s
        device = self._device
        assert device is not None

        results: list[dict] = []

        for idx, (start_s, end_s) in enumerate(ctx.overlap_regions):
            # Length sanity (mirrors the POC's < 256 sample skip).
            if int((end_s - start_s) * sr) < _MIN_OVERLAP_SAMPLES:
                continue

            window = self._pick_window(
                start_s, end_s, total_dur, ctx.audio, sr
            )
            pad_lo = int(window.pad_start_s * sr)
            pad_hi = int(window.pad_end_s * sr)
            mix = ctx.audio[pad_lo:pad_hi].astype(np.float32)

            # Separation (chunked if the padded window exceeds threshold).
            padded_dur_s = (pad_hi - pad_lo) / sr
            if padded_dur_s > cfg.overlap_add_threshold_s:
                s1_raw, s2_raw = _separate_overlap_add(
                    mix,
                    self._separator,
                    device,
                    sr_pipeline=sr,
                    sr_separator=cfg.separator_sample_rate,
                    chunk_length_s=cfg.training_chunk_length_s,
                )
                chunked = True
            else:
                s1_raw, s2_raw = _separate_single(
                    mix,
                    self._separator,
                    device,
                    sr_pipeline=sr,
                    sr_separator=cfg.separator_sample_rate,
                )
                chunked = False

            # Volume normalisation.
            s1_raw, s2_raw, vol_scale = _volume_normalise(
                s1_raw, s2_raw, mix, cfg.volume_normalization
            )

            # VAD gate on each output stream.
            mask1 = _vad_mask_silero(s1_raw, self._vad, device, cfg.vad_threshold)
            mask2 = _vad_mask_silero(s2_raw, self._vad, device, cfg.vad_threshold)
            s1_gated = s1_raw * mask1
            s2_gated = s2_raw * mask2

            # Emit region (where the assembler will splice the streams).
            emit_start_s, emit_end_s = self._pick_emit_region(
                window=window,
                overlap_start_s=start_s,
                overlap_end_s=end_s,
                combined_audio=(s1_raw + s2_raw),
                combined_vad_mask=np.maximum(mask1, mask2),
                sample_rate=sr,
            )

            results.append({
                "idx": int(idx),
                "start": start_s,
                "end": end_s,
                "pad_start": float(window.pad_start_s),
                "pad_end": float(window.pad_end_s),
                "emit_start": float(emit_start_s),
                "emit_end": float(emit_end_s),
                "chunked": bool(chunked),
                "volume_scale": float(vol_scale),
                "mix": mix,
                "s1_raw": s1_raw,
                "s2_raw": s2_raw,
                "s1_gated": s1_gated,
                "s2_gated": s2_gated,
                "mask1": mask1,
                "mask2": mask2,
            })

        ctx.overlap_separated = results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pick_window(
        self,
        start_s: float,
        end_s: float,
        total_dur: float,
        audio_16k: np.ndarray,
        sr: int,
    ) -> _Window:
        mode = self.config.context_window_mode
        if mode == "none":
            return _window_none(start_s, end_s)
        if mode == "fixed_pad":
            return _window_fixed_pad(
                start_s, end_s, total_dur, self.config.context_pad_seconds
            )
        if mode == "snap_to_vad":
            assert self._vad is not None and self._device is not None
            return _window_snap_to_vad(
                start_s,
                end_s,
                total_dur,
                target_total_s=self.config.training_chunk_length_s,
                audio_16k=audio_16k,
                vad_model=self._vad,
                device=self._device,
                vad_threshold=self.config.vad_threshold,
                sample_rate=sr,
            )
        raise ValueError(f"Unknown context_window_mode: {mode!r}")

    def _pick_emit_region(
        self,
        window: _Window,
        overlap_start_s: float,
        overlap_end_s: float,
        combined_audio: np.ndarray,
        combined_vad_mask: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, float]:
        """Return (emit_start_s, emit_end_s) in absolute recording time.

        The assembler will slice the *padded* separator output to this range.
        """
        cfg = self.config
        if cfg.seam_mode == "overlap_boundary":
            return overlap_start_s, overlap_end_s

        # Both "zero_crossing" and "snap_to_silence" start from a zero-crossing
        # nudge of the original overlap boundary. `snap_to_silence` then extends
        # outward via VAD silence (never contracts past the zc boundary).
        radius = int(cfg.seam_search_radius_s * sample_rate)
        pad_lo = int(window.pad_start_s * sample_rate)
        tgt_left = int(overlap_start_s * sample_rate) - pad_lo
        tgt_right = int(overlap_end_s * sample_rate) - pad_lo
        tgt_left = max(0, min(len(combined_audio) - 1, tgt_left))
        tgt_right = max(0, min(len(combined_audio) - 1, tgt_right))
        zc_left = _pick_seam_zero_crossing(combined_audio, tgt_left, radius)
        zc_right = _pick_seam_zero_crossing(combined_audio, tgt_right, radius)

        if cfg.seam_mode == "snap_to_silence":
            max_extend_n = int(cfg.snap_silence_max_extend_s * sample_rate)
            new_left = _extend_start_to_silence(
                combined_vad_mask, zc_left, max_extend_n
            )
            new_right = _extend_end_to_silence(
                combined_vad_mask, zc_right, max_extend_n
            )
        elif cfg.seam_mode == "zero_crossing":
            new_left = zc_left
            new_right = zc_right
        else:
            raise ValueError(f"Unknown seam_mode: {cfg.seam_mode!r}")

        emit_start_s = (pad_lo + new_left) / sample_rate
        emit_end_s = (pad_lo + new_right) / sample_rate
        # Guard: never invert.
        if emit_end_s < emit_start_s:
            return overlap_start_s, overlap_end_s
        return emit_start_s, emit_end_s

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.overlap_separated:
            return
        meta = {
            "knobs": {
                "context_window_mode": self.config.context_window_mode,
                "context_pad_seconds": self.config.context_pad_seconds,
                "training_chunk_length_s": self.config.training_chunk_length_s,
                "seam_mode": self.config.seam_mode,
                "seam_search_radius_s": self.config.seam_search_radius_s,
                "overlap_add_threshold_s": self.config.overlap_add_threshold_s,
                "volume_normalization": self.config.volume_normalization,
                "vad_threshold": self.config.vad_threshold,
                "vad_mode": self.config.vad_mode,
            },
            "overlaps": [],
        }
        for entry in ctx.overlap_separated:
            idx = entry["idx"]
            sf.write(
                artifact_dir / f"overlap_{idx}_s1.wav",
                entry["s1_gated"].astype(np.float32),
                ctx.sample_rate,
            )
            sf.write(
                artifact_dir / f"overlap_{idx}_s2.wav",
                entry["s2_gated"].astype(np.float32),
                ctx.sample_rate,
            )
            meta["overlaps"].append({
                "idx": idx,
                "start": entry["start"],
                "end": entry["end"],
                "pad_start": entry["pad_start"],
                "pad_end": entry["pad_end"],
                "emit_start": entry["emit_start"],
                "emit_end": entry["emit_end"],
                "chunked": entry["chunked"],
                "volume_scale": entry["volume_scale"],
            })
        with open(artifact_dir / "separation_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
