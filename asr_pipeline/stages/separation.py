"""Stage 3b — overlap separation + VAD gating.

For each overlap region in `ctx.overlap_regions`:

1. Pick a *context window* around the overlap (``context_window_mode``).
   The window is fed to the separator; ``expand_to_chunk`` and
   ``fixed_pad`` widen it past the original overlap so the separator
   gets more context.
2. Run the separator (resample 16 k -> 8 k -> separator -> 8 k -> 16 k).
   For overlaps wider than ``overlap_add_threshold_s`` we chunk + overlap-add.
3. Volume-normalise the two outputs (``volume_normalization``).
4. VAD-gate each output stream (``vad_threshold`` + silero) to suppress the
   residual energy the separator inevitably leaks where there's no speech.
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
from asr_pipeline.context import OverlapSeparated, PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage


def _log(msg: str) -> None:
    """Separation-stage debug log — file only.

    Used in `unload()` for low-level GPU teardown logging (sync, empty_cache);
    keeping these out of stdout avoids cluttering the notebook on every
    stage transition. They remain available via the debug log file.
    """
    dlog("separation", msg, to_stdout=False)


_SILERO_WINDOW = 512  # silero-vad expects exactly 512 samples per call @ 16 kHz
_MIN_OVERLAP_SAMPLES = 256  # POC's lower bound for "long enough to be worth separating"


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


def _soft_threshold_mask(
    probs: np.ndarray, upper: float, lower: float
) -> np.ndarray:
    """Schmitt-trigger style frame mask.

    Frames with prob > `upper` are always speech ("strong"). Frames with prob
    > `lower` are speech *only* if they're connected to a strong frame
    through an unbroken chain of also-near-threshold frames (propagated both
    forward and backward in time). Frames at or below `lower` are silence.

    Captures speech tails/onsets where the model dipped below the strict
    threshold but is still seeing some evidence of speech.
    """
    strong = probs > upper
    if not lower or lower >= upper:
        return strong.astype(np.float32)
    weak = probs > lower
    mask = strong.copy()
    # Forward propagation: extend speech rightward through weak frames.
    for i in range(1, len(probs)):
        if mask[i - 1] and weak[i]:
            mask[i] = True
    # Backward propagation: extend speech leftward through weak frames.
    for i in range(len(probs) - 2, -1, -1):
        if mask[i + 1] and weak[i]:
            mask[i] = True
    return mask.astype(np.float32)


def _dilate_mask(
    mask_frames: np.ndarray, attack_frames: int, release_frames: int
) -> np.ndarray:
    """Extend each speech run by fixed frame counts at onset/offset.

    Adds `attack_frames` of speech before each onset (0→1) and
    `release_frames` of speech after each offset (1→0). No-op when both
    counts are 0.
    """
    if attack_frames <= 0 and release_frames <= 0:
        return mask_frames
    out = mask_frames.astype(bool).copy()
    n = len(out)
    padded = np.concatenate(([False], out, [False]))
    diff = np.diff(padded.astype(np.int8))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    for onset in onsets:
        lo = max(0, onset - attack_frames)
        out[lo:onset] = True
    for offset in offsets:
        hi = min(n, offset + release_frames)
        out[offset:hi] = True
    return out.astype(np.float32)


def _vad_mask_silero(
    audio_16k: np.ndarray,
    vad_model,
    device: torch.device,
    threshold: float,
    soft_threshold: float = 0.0,
    attack_frames: int = 0,
    release_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample VAD mask + raw per-frame probabilities at 16 kHz.

    Returns ``(full_mask, probs)``:
      - ``full_mask``: shape (n_samples,), 1.0 = speech, 0.0 = silence
      - ``probs``: shape (n_frames,), raw silero output per 512-sample frame

    Frames are 512 samples (silero's required window size at 16 kHz). The
    final mask is built in three steps:

    1. Run silero per non-overlapping 512-sample frame → probability per frame.
    2. Threshold to a frame mask. When `soft_threshold > 0` and < `threshold`,
       use a Schmitt-trigger style soft threshold (see `_soft_threshold_mask`);
       otherwise a strict threshold.
    3. Optionally dilate the frame mask by `attack_frames` / `release_frames`.

    Then expand frame-resolution mask to sample resolution by repeating each
    frame's decision over its 512 samples.

    Returning the raw `probs` alongside the mask lets the notebook plot the
    actual VAD curve next to the binary mask, so the user can see *why* a
    given region was gated.
    """
    audio = torch.from_numpy(audio_16k).to(device)
    n = int(audio.numel())
    if n < _SILERO_WINDOW:
        return np.ones(n, dtype=np.float32), np.ones(0, dtype=np.float32)
    n_windows = n // _SILERO_WINDOW
    probs = np.zeros(n_windows, dtype=np.float32)
    vad_model.reset_states()
    with torch.no_grad():
        for i in range(n_windows):
            chunk = audio[i * _SILERO_WINDOW : (i + 1) * _SILERO_WINDOW]
            p = vad_model(chunk.unsqueeze(0), 16_000)
            probs[i] = p.item() if hasattr(p, "item") else float(p)

    mask_frames = _soft_threshold_mask(probs, upper=threshold, lower=soft_threshold)
    mask_frames = _dilate_mask(mask_frames, attack_frames, release_frames)

    full_mask = np.zeros(n, dtype=np.float32)
    for i, m in enumerate(mask_frames):
        full_mask[i * _SILERO_WINDOW : (i + 1) * _SILERO_WINDOW] = m
    full_mask[n_windows * _SILERO_WINDOW :] = 1.0  # leftover tail: keep as speech
    return full_mask, probs


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


def _signal_aware_pad(
    start_s: float,
    end_s: float,
    total_duration_s: float,
    target_total_s: float,
) -> tuple[float, float]:
    """Distribute pad to reach `target_total_s` total window length, biased
    toward the side with available room.

    Tries a symmetric split first. When one side hits a recording boundary
    (0 or `total_duration_s`), the leftover budget is redistributed to the
    other side. This gives the separator more useful context when the
    overlap sits near the start or end of the recording — where one side
    has effectively no signal to draw from anyway.

    Returns (pad_start_s, pad_end_s). May be narrower than `target_total_s`
    if the recording itself is shorter.
    """
    overlap_dur = end_s - start_s
    if target_total_s <= overlap_dur:
        return start_s, end_s
    extra = target_total_s - overlap_dur
    room_left = start_s
    room_right = max(0.0, total_duration_s - end_s)
    left_take = extra / 2.0
    right_take = extra / 2.0
    # Spill from saturated side to the other.
    if left_take > room_left:
        right_take += left_take - room_left
        left_take = room_left
    if right_take > room_right:
        left_take += right_take - room_right
        right_take = room_right
    # Final clamp (both sides may be saturated; rest of `extra` is lost).
    left_take = min(left_take, room_left)
    right_take = min(right_take, room_right)
    return start_s - left_take, end_s + right_take


def _window_fixed_pad(
    start_s: float,
    end_s: float,
    total_duration_s: float,
    context_pad_s: float,
    min_fragment_length_s: float = 0.0,
) -> _Window:
    """Symmetric `±context_pad_s` window, with two extensions:

    - If the resulting total window would be shorter than
      `min_fragment_length_s`, the window is widened (via `_signal_aware_pad`)
      until it reaches that floor.
    - When one side hits a recording boundary, the unused budget is
      redistributed to the other side.
    """
    overlap_dur = end_s - start_s
    target_total = max(overlap_dur + 2 * context_pad_s, min_fragment_length_s)
    pad_start, pad_end = _signal_aware_pad(
        start_s, end_s, total_duration_s, target_total
    )
    return _Window(pad_start_s=pad_start, pad_end_s=pad_end)


def _window_expand_to_chunk(
    start_s: float,
    end_s: float,
    total_duration_s: float,
    target_total_s: float,
    min_fragment_length_s: float = 0.0,
) -> _Window:
    """Pad asymmetrically so the resulting window is
    `max(target_total_s, min_fragment_length_s)` seconds wide.

    `target_total_s` is typically the separator's training chunk length
    — feeding it a window of the size it was trained on gives the
    cleanest separation. Padding distribution uses `_signal_aware_pad`,
    so when one side hits the recording boundary the leftover budget
    moves to the other side instead of being lost.

    No utterance-aware boundary adjustment: the separator was trained
    on mid-utterance crops, so cutting an utterance at the pad boundary
    doesn't hurt separation quality. Utterance-aware boundary handling
    belongs in `seam_mode` (which controls the *emit* region — the
    part actually spliced into the per-speaker stream).
    """
    effective_target = max(target_total_s, min_fragment_length_s)
    pad_start, pad_end = _signal_aware_pad(
        start_s, end_s, total_duration_s, effective_target
    )
    return _Window(pad_start_s=pad_start, pad_end_s=pad_end)


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

        vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        vad_model = vad_model.to(device)

        self._separator = separator
        self._vad = vad_model
        self._device = device

    def load_signature(self) -> tuple:
        # Only the separator checkpoint controls which weights end up on the
        # GPU. All other knobs (context window mode, seam mode, VAD
        # thresholds, volume normalisation, etc.) are runtime behaviour —
        # re-read on every call, no reload needed.
        return (self.config.checkpoint_path,)

    def unload(self) -> None:
        # Detailed logging here because this method is the prime suspect for
        # long-recording CUDA deadlocks: a stuck stream from stage 3b will
        # block `torch.cuda.empty_cache()` indefinitely. The explicit
        # synchronize() turns a silent hang into a (still long, but visible)
        # block — we log immediately before and after so the user can see
        # whether sync or empty_cache is the one that doesn't return.
        _log("unload: dropping separator + VAD references...")
        self._separator = None
        self._vad = None
        self._device = None
        _log("unload: gc.collect()...")
        gc.collect()
        if torch.cuda.is_available():
            _log("unload: torch.cuda.synchronize() ...")
            torch.cuda.synchronize()
            _log("unload: torch.cuda.empty_cache() ...")
            torch.cuda.empty_cache()
        _log("unload: done")

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

            # VAD gate on each output stream. Soft-threshold + attack/release
            # dilation applied to keep onset/tail frames in the mask without
            # admitting between-utterance noise.
            mask1, probs1 = _vad_mask_silero(
                s1_raw, self._vad, device, cfg.vad_threshold,
                soft_threshold=cfg.vad_soft_threshold,
                attack_frames=cfg.vad_attack_frames,
                release_frames=cfg.vad_release_frames,
            )
            mask2, probs2 = _vad_mask_silero(
                s2_raw, self._vad, device, cfg.vad_threshold,
                soft_threshold=cfg.vad_soft_threshold,
                attack_frames=cfg.vad_attack_frames,
                release_frames=cfg.vad_release_frames,
            )
            # NB: we DON'T apply the mask here — that's Stage 3c's job
            # (`post_separation_processing`). 3c needs the unmasked
            # streams as input to its BWE backend (mask edges look like
            # discontinuities to BWE). The mask is still computed here
            # because `seam_mode == "snap_to_silence"` uses it to extend
            # emit-region boundaries via VAD silence detection.

            # Emit region (where the assembler will splice the streams).
            emit_start_s, emit_end_s = self._pick_emit_region(
                window=window,
                overlap_start_s=start_s,
                overlap_end_s=end_s,
                combined_audio=(s1_raw + s2_raw),
                combined_vad_mask=np.maximum(mask1, mask2),
                sample_rate=sr,
            )

            # The TypedDict schema is defined in `asr_pipeline/context.py`.
            # `s{1,2}_gated` is added by Stage 3c (post_separation_processing).
            entry: OverlapSeparated = {
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
                "mask1": mask1,
                "mask2": mask2,
                # Raw per-frame VAD probabilities (16 kHz audio -> 512-sample
                # frames -> 32 ms per frame). Lets the notebook plot the actual
                # silero curve next to the binary mask.
                "probs1": probs1,
                "probs2": probs2,
            }
            results.append(entry)

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
                start_s, end_s, total_dur,
                context_pad_s=self.config.context_pad_seconds,
                min_fragment_length_s=self.config.min_fragment_length_s,
            )
        if mode == "expand_to_chunk":
            return _window_expand_to_chunk(
                start_s,
                end_s,
                total_dur,
                target_total_s=self.config.training_chunk_length_s,
                min_fragment_length_s=self.config.min_fragment_length_s,
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
                "min_fragment_length_s": self.config.min_fragment_length_s,
                "training_chunk_length_s": self.config.training_chunk_length_s,
                "seam_mode": self.config.seam_mode,
                "seam_search_radius_s": self.config.seam_search_radius_s,
                "overlap_add_threshold_s": self.config.overlap_add_threshold_s,
                "volume_normalization": self.config.volume_normalization,
                "vad_threshold": self.config.vad_threshold,
            },
            "overlaps": [],
        }
        for entry in ctx.overlap_separated:
            idx = entry["idx"]
            # Spill the *unmasked* separator outputs — they're what this
            # stage actually produces. Stage 3c spills the masked
            # `_gated` versions in turn.
            sf.write(
                artifact_dir / f"overlap_{idx}_s1_raw.wav",
                entry["s1_raw"].astype(np.float32),
                ctx.sample_rate,
            )
            sf.write(
                artifact_dir / f"overlap_{idx}_s2_raw.wav",
                entry["s2_raw"].astype(np.float32),
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
