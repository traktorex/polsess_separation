"""Stage 4 — per-speaker stream assembly via ECAPA anchors + TimestampMap.

Inputs from upstream stages:
- `ctx.enhanced_full`     full-recording enhancer output (Stage 3a). When
                          `enhancement.enabled: false` this stays None and
                          the assembler falls back to `ctx.audio`.
- `ctx.diarization`       per-speaker pyannote segments + overlap timeline
- `ctx.overlap_regions`   the intervals SepFormer was run on (Stage 2)
- `ctx.overlap_separated` per-region SepFormer outputs + emit boundaries

For each speaker:

1. Derive that speaker's solo intervals on the fly: pyannote's segments
   for the speaker, minus `ctx.overlap_regions`. No padding — pyannote's
   boundaries are used as-is.
2. Build an ECAPA-TDNN *anchor* embedding from a concatenation of
   `enhanced_full` sliced at those solo intervals. If any speaker has
   less than `min_solo_for_anchor_s` of solo audio, set the diagnostic
   flag `ctx.weak_anchor = True`. This flag is informational — it lets
   the caller know the anchor quality is shaky but ECAPA assignment is
   still attempted. The fixed straight-through fallback only kicks in
   per-overlap when ECAPA actually fails to produce an anchor (input
   below the 0.25 s ECAPA floor); see step 3.
3. For each overlap, ECAPA-embed each of the two SepFormer outputs (the
   full padded gated stream) and pick the speaker pairing
   (s1->A,s2->B vs s1->B,s2->A) with the higher summed cosine similarity
   against the anchors. **No-separation mode** (`separation.enabled: false`,
   or 3b otherwise produced nothing): fill overlap regions from the
   enhanced mixture, attributing the same slice to all speakers.
4. Slice each overlap's gated stream to its `emit_start`/`emit_end`
   region (from Stage 3b), so we only emit the assembler-relevant slice.
5. Concatenate per speaker in `output_mode`:
     - "shortened"   -> speech-only concat with `silence_separator_s` gaps
     - "full_length" -> stream length = input length; pieces placed at
                        their original timestamps; gaps filled with silence
6. Build a `TimestampMap` that records the (concat_*) -> (orig_*) mapping
   for every piece in every speaker's stream.

Speaker-assignment offers three strategies via `assembly.overlap_assignment`:
the POC's per-overlap ECAPA argmax (`ecapa_argmax`, default); `continuity_tiebreak`,
which re-decides near-ties with local bracketing-solo anchors (`_continuity_decision`);
and `consensus_2means`, a global constrained-2-means pre-pass (`_consensus_pairings`)
that decides all overlaps jointly so consensus can overrule a lone confident-but-wrong
one. All dispatch inside `_assign_overlaps`.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from asr_pipeline.config import AssemblyConfig
from asr_pipeline.context import (
    Interval,
    PipelineContext,
    TimestampMap,
    TimestampMapEntry,
)
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage


def _log(msg: str) -> None:
    """Progress message for the assembly stage. Writes to both stdout and
    the debug log file (so messages survive VSCode WSL kernel disconnects).
    """
    dlog("assembly", msg)


# ECAPA needs a minimum amount of audio for a usable speaker embedding.
# Inputs shorter than this are zero-padded up to it (`_ecapa_embed`) or
# skipped entirely (`_compute_anchors` leaves the anchor None). Now exposed as
# `AssemblyConfig.anchor_min_duration_s` (default 0.25); the free functions take
# it as a parameter (no config access) and default to that value.
_ANCHOR_MIN_DURATION_S_DEFAULT = 0.25
# Overlap separator streams shorter than this are too brief for a *stable*
# ECAPA embedding, so `_assign_overlaps` falls back to fixed assignment
# rather than trusting a noisy cosine on a fraction of a syllable. Now exposed
# as `AssemblyConfig.overlap_min_duration_s` (default 0.1); threaded as a
# parameter with that default.
_OVERLAP_MIN_DURATION_S_DEFAULT = 0.1


# ---------------------------------------------------------------------------
# Solo-interval derivation
# ---------------------------------------------------------------------------


def _coalesce(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping intervals into disjoint ones, sorted by start.

    pyannote can in principle emit two overlapping segments for the same
    speaker; without coalescing, `_subtract` would process each independently
    and the shared region would survive twice — duplicating that speech into
    the assembled stream (and the transcript). Empirically pyannote-3.1's
    binarised output is already disjoint per speaker (a scan of every CLARIN
    diarization found zero same-speaker self-overlap), so this is
    disjoint-by-construction insurance that keeps assembly consistent with the
    rest of the pipeline — every other interval set is coalesced before use.
    Touching intervals (`s == prev_end`) are left separate: they double-count
    nothing.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    out: list[list[float]] = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


def _subtract(
    regions_a: list[Interval], regions_b: list[Interval]
) -> list[Interval]:
    """Set difference `regions_a − regions_b` on intervals."""
    result: list[Interval] = []
    regions_b = sorted(regions_b)
    for s, e in sorted(regions_a):
        current: list[Interval] = [(s, e)]
        for bs, be in regions_b:
            next_cur: list[Interval] = []
            for cs, ce in current:
                if be <= cs or bs >= ce:
                    next_cur.append((cs, ce))
                    continue
                if bs > cs:
                    next_cur.append((cs, bs))
                if be < ce:
                    next_cur.append((be, ce))
            current = next_cur
        result.extend(current)
    return [(s, e) for s, e in result if e - s > 1e-3]


def _speaker_solo_intervals(
    seg_df, speaker: str, overlap_regions: list[Interval]
) -> list[Interval]:
    """Per-speaker pyannote segments minus the overlap regions."""
    raw = [
        (float(r.start), float(r.end))
        for r in seg_df[seg_df["speaker"] == speaker].itertuples()
    ]
    return _subtract(_coalesce(raw), overlap_regions)


# ---------------------------------------------------------------------------
# ECAPA embedding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ecapa_embed(
    audio_16k: np.ndarray, ecapa, device: torch.device, sample_rate: int,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> torch.Tensor:
    """Return a unit-norm speaker embedding for the audio (1-D float32).

    Dispatches on the embedder kind so the anchor_embedding=ecapa2 knob
    (Option 4) and the default SpeechBrain ECAPA1 share one call site:

      - SpeechBrain ``EncoderClassifier`` exposes ``encode_batch`` and is fed
        ``(1, T)`` → ``(1, 1, dim)``. The current/default path; byte-identical.
      - The custom ECAPA2 wrapper (``BaseCustomSpeakerEmbedding``) is callable as
        ``embedder((1, 1, T)) -> (1, dim) numpy``; we wrap it to a torch tensor.

    ``anchor_min_duration_s`` (AssemblyConfig.anchor_min_duration_s, default
    0.25 s) is the pad floor: shorter input is zero-padded up to it. The default
    is >= both embedders' minimum (SB's floor and the wrapper's ~25 ms
    ``min_num_samples``), so neither path under-feeds.
    """
    min_len = int(sample_rate * anchor_min_duration_s)
    if len(audio_16k) < min_len:
        audio_16k = np.pad(audio_16k, (0, min_len - len(audio_16k)))
    if hasattr(ecapa, "encode_batch"):
        audio = torch.from_numpy(audio_16k).unsqueeze(0).to(device)
        emb = ecapa.encode_batch(audio).squeeze(0).squeeze(0)
    else:
        # Custom wrapper: (batch=1, channel=1, T) → (1, dim) numpy.
        wav = torch.from_numpy(audio_16k.astype(np.float32)).reshape(1, 1, -1)
        emb = torch.from_numpy(np.asarray(ecapa(wav)[0])).to(device)
    emb = emb / (emb.norm() + 1e-8)
    return emb


def _cap_anchor_audio(
    audio_16k: np.ndarray, sample_rate: int, max_duration_s: Optional[float]
) -> np.ndarray:
    """Cap anchor audio length by uniformly subsampling 1-second chunks.

    ECAPA is fed in a single forward pass; on a long recording the per-speaker
    solo concat (e.g. 400 s) blows up GPU memory or stalls the kernel. This
    keeps the concat under `max_duration_s` by selecting evenly-spaced
    1-second chunks across the original concat, so the sample still covers
    the speaker's full timeline rather than just the start.

    `max_duration_s=None` (or audio already short enough) returns the input
    unchanged.
    """
    if max_duration_s is None:
        return audio_16k
    cap_samples = int(max_duration_s * sample_rate)
    if len(audio_16k) <= cap_samples:
        return audio_16k
    chunk_samples = sample_rate            # 1 s chunks
    n_chunks = max(1, cap_samples // chunk_samples)
    total_chunks = max(1, len(audio_16k) // chunk_samples)
    if n_chunks >= total_chunks:
        return audio_16k[:cap_samples]
    # Evenly-spaced start indices for the chunks.
    starts = np.linspace(0, len(audio_16k) - chunk_samples, n_chunks).astype(int)
    pieces = [audio_16k[s : s + chunk_samples] for s in starts]
    return np.concatenate(pieces).astype(np.float32)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


def _local_anchor(
    solo_intervals: list[Interval],
    ovl_start: float,
    ovl_end: float,
    window_s: float,
    audio: np.ndarray,
    ecapa,
    device: torch.device,
    sr: int,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> Optional[torch.Tensor]:
    """ECAPA embedding from a speaker's solo audio bracketing one overlap.

    Concatenates the speaker's solo intervals clipped to ``[ovl_start -
    window_s, ovl_end + window_s]`` and embeds them. Returns None when there is
    less than ``anchor_min_duration_s`` of local solo — too little for a stable
    embedding, so the caller falls back to the global anchor. This is the
    "speech continuity" signal: the speaker's voice right next to the overlap,
    which tracks local channel/SNR conditions better than the global anchor.
    """
    lo_t = ovl_start - window_s
    hi_t = ovl_end + window_s
    slices: list[np.ndarray] = []
    for s, e in solo_intervals:
        cs = max(s, lo_t)
        ce = min(e, hi_t)
        if ce > cs:
            lo = int(cs * sr)
            hi = int(ce * sr)
            if hi > lo:
                slices.append(audio[lo:hi].astype(np.float32))
    if not slices:
        return None
    concat = np.concatenate(slices)
    if len(concat) < int(sr * anchor_min_duration_s):
        return None
    return _ecapa_embed(concat, ecapa, device, sr, anchor_min_duration_s).cpu()


def _continuity_decision(
    argmax_pairing: str,
    straight: float,
    swapped: float,
    margin: float,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    a: str,
    b: str,
    ovl: dict,
    window_s: float,
    solo_intervals_by_spk: Optional[dict[str, list[Interval]]],
    audio: Optional[np.ndarray],
    ecapa,
    device: torch.device,
    sr: int,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> tuple[str, str]:
    """Continuity tie-break for one overlap. Returns ``(chosen, pairing_label)``.

    When the global-anchor straight/swapped gap clears ``margin`` the global
    argmax stands (decisive — no need to second-guess). On a near-tie it
    re-decides using *local* anchors (:func:`_local_anchor`) from the solo audio
    bracketing the overlap, falling back to the global argmax whenever a local
    anchor is unavailable or non-finite. Stateless: depends only on this
    overlap's neighbourhood (no carry-forward chain), so it cannot propagate one
    wrong decision the way the ``overlap_assign_min_margin`` prior can.
    """
    if abs(straight - swapped) >= margin:
        return argmax_pairing, argmax_pairing
    if not solo_intervals_by_spk or audio is None:
        return argmax_pairing, f"{argmax_pairing} (continuity unavailable)"
    ovl_start = float(ovl["emit_start"])
    ovl_end = float(ovl["emit_end"])
    la = _local_anchor(solo_intervals_by_spk.get(a, []), ovl_start, ovl_end,
                       window_s, audio, ecapa, device, sr, anchor_min_duration_s)
    lb = _local_anchor(solo_intervals_by_spk.get(b, []), ovl_start, ovl_end,
                       window_s, audio, ecapa, device, sr, anchor_min_duration_s)
    if la is None or lb is None:
        return argmax_pairing, f"{argmax_pairing} (continuity unavailable)"
    straight_l = _cos(emb1, la) + _cos(emb2, lb)
    swapped_l = _cos(emb1, lb) + _cos(emb2, la)
    if not (np.isfinite(straight_l) and np.isfinite(swapped_l)):
        return argmax_pairing, f"{argmax_pairing} (continuity non-finite)"
    chosen = "straight" if straight_l >= swapped_l else "swapped"
    return chosen, f"{chosen} (continuity tie-break)"


def _consensus_pairings(
    overlap_separated: list,
    anchors: dict[str, Optional[torch.Tensor]],
    speakers: list[str],
    ecapa,
    device: torch.device,
    sr: int,
    max_iter: int = 10,
    overlap_min_duration_s: float = _OVERLAP_MIN_DURATION_S_DEFAULT,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> dict[int, str]:
    """Global per-overlap pairings by constrained 2-means (consensus).

    Seeds two centroids from the solo anchors, then alternates until the pairings
    stop changing: (1) assign each overlap's two streams to the centroids by best
    summed cosine, (2) recompute each centroid from the solo anchor PLUS the
    overlap embeddings now assigned to it. Because every overlap votes into the
    centroids, a lone confident-but-wrong per-overlap decision can be overruled
    by the consensus of the rest — the lever that ecapa_argmax / continuity (both
    per-overlap) cannot reach. The per-overlap constraint (the two streams go to
    different speakers) is intrinsic to the straight/swapped choice.

    Returns ``{i_ovl: 'straight'|'swapped'}`` for ECAPA-eligible overlaps only;
    an empty dict when an anchor is missing or no overlap is eligible (the caller
    then falls back to the per-overlap argmax path, never dropping a region).
    """
    if len(speakers) < 2 or any(anchors.get(s) is None for s in speakers[:2]):
        return {}
    a, b = speakers[0], speakers[1]
    min_len = int(sr * overlap_min_duration_s)
    embs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, ovl in enumerate(overlap_separated):
        if "s1_gated" not in ovl or len(ovl["s1_gated"]) < min_len:
            continue
        e1 = _ecapa_embed(ovl["s1_gated"], ecapa, device, sr, anchor_min_duration_s).cpu()
        e2 = _ecapa_embed(ovl["s2_gated"], ecapa, device, sr, anchor_min_duration_s).cpu()
        if torch.isfinite(e1).all() and torch.isfinite(e2).all():
            embs[i] = (e1, e2)
    if not embs:
        return {}
    cA, cB = anchors[a].clone(), anchors[b].clone()
    pairings: dict[int, str] = {}
    for _ in range(max_iter):
        new = {
            i: ("straight"
                if _cos(e1, cA) + _cos(e2, cB) >= _cos(e1, cB) + _cos(e2, cA)
                else "swapped")
            for i, (e1, e2) in embs.items()
        }
        if new == pairings:
            break
        pairings = new
        va, vb = [anchors[a]], [anchors[b]]
        for i, (e1, e2) in embs.items():
            hi, lo = (e1, e2) if pairings[i] == "straight" else (e2, e1)
            va.append(hi)
            vb.append(lo)
        cA = torch.stack(va).mean(0); cA = cA / (cA.norm() + 1e-8)
        cB = torch.stack(vb).mean(0); cB = cB / (cB.norm() + 1e-8)
    return pairings


# ---------------------------------------------------------------------------
# Emit-region slicing
# ---------------------------------------------------------------------------


def _slice_emit(
    gated_audio: np.ndarray,
    pad_start_s: float,
    emit_start_s: float,
    emit_end_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Slice the gated padded separator output to the assembler emit region.

    Round, don't truncate: the separation stage derives `emit_*` as
    `pad_start_s + k/sr`, so the offset here is an integer number of samples
    up to float error — truncation would turn a −ε rounding residue into a
    full one-sample shift off the seam the separation stage chose.
    """
    offset = int(round((emit_start_s - pad_start_s) * sample_rate))
    length = int(round((emit_end_s - emit_start_s) * sample_rate))
    if offset < 0:
        offset = 0
    if offset + length > len(gated_audio):
        length = len(gated_audio) - offset
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    return gated_audio[offset : offset + length].astype(np.float32)


# ---------------------------------------------------------------------------
# Optional per-piece RMS normalisation
# ---------------------------------------------------------------------------


def _rms(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _rms_normalise(
    pieces: list[np.ndarray], target_rms: Optional[float]
) -> list[np.ndarray]:
    """Rescale each piece so its RMS matches `target_rms`.

    When `target_rms` is None, the median RMS across non-empty pieces is
    used as the target. Empty pieces are returned untouched.
    """
    rms_values = [_rms(p) for p in pieces if p.size > 0]
    if not rms_values:
        return pieces
    if target_rms is None:
        target_rms = float(np.median(rms_values))
    out: list[np.ndarray] = []
    for p in pieces:
        r = _rms(p)
        if r < 1e-9 or p.size == 0:
            out.append(p)
        else:
            out.append(p * (target_rms / r))
    return out


def _match_overlap_rms_to_solo(events_for_spk: list[dict]) -> None:
    """Scale each `overlap` event's audio so its RMS matches the median RMS
    of the speaker's `solo` events. Mutates in place.
    """
    solo_rms = [
        r
        for e in events_for_spk
        if e["kind"] == "solo" and (r := _rms(e["audio"])) > 1e-9
    ]
    if not solo_rms:
        return
    target = float(np.median(solo_rms))
    for e in events_for_spk:
        if e["kind"] != "overlap":
            continue
        r = _rms(e["audio"])
        if r > 1e-9:
            e["audio"] = (e["audio"] * (target / r)).astype(np.float32)


def _apply_fade(audio: np.ndarray, in_n: int, out_n: int) -> np.ndarray:
    """Apply a half-Hann fade-in (`in_n` samples) and fade-out (`out_n`
    samples) to the audio's edges. `in_n` and `out_n` may differ — the
    assembler uses a longer fade for piece-to-piece seams (crossfade_ms)
    and a shorter one at the stream's outermost edges (edge_fade_ms).

    Returns a new array; input untouched. If a piece is too short to fit
    both fades, each one is capped at half the piece length.
    """
    if (in_n <= 0 and out_n <= 0) or len(audio) == 0:
        return audio
    out = audio.copy()
    in_n = min(in_n, len(out) // 2)
    out_n = min(out_n, len(out) // 2)
    # A 1-sample ramp is a click, not a fade: np.hanning(2) == [0, 0], so the
    # half-Hann would zero the single boundary sample instead of leaving it
    # ~unchanged. Skip ramps shorter than 2 samples (a no-op is correct there).
    if in_n >= 2:
        ramp_in = np.hanning(2 * in_n)[:in_n].astype(np.float32)
        out[:in_n] *= ramp_in
    if out_n >= 2:
        ramp_out = np.hanning(2 * out_n)[out_n:].astype(np.float32)
        out[-out_n:] *= ramp_out
    return out


# ---------------------------------------------------------------------------
# Phase helpers (called by AssemblyStage.run)
# ---------------------------------------------------------------------------


def _derive_solo_intervals(
    ctx: PipelineContext, speakers: list[str]
) -> dict[str, list[Interval]]:
    """Per-speaker pyannote segments minus the emit regions from Stage 3b.

    When 3b extends emit boundaries past pyannote's overlap (seam_mode=
    "snap_to_silence"), the extension would otherwise appear in both the
    overlap event and the adjacent solo event. Subtracting *emit* regions
    (not the raw routing regions) prevents that duplication. Falls back to
    `ctx.overlap_regions` when 3b didn't run.
    """
    assert ctx.diarization is not None  # checked by AssemblyStage.run
    if ctx.overlap_separated:
        blocked: list[Interval] = [
            (float(o["emit_start"]), float(o["emit_end"]))
            for o in ctx.overlap_separated
        ]
    else:
        blocked = ctx.overlap_regions or []
    return {
        spk: _speaker_solo_intervals(ctx.diarization.segments_df, spk, blocked)
        for spk in speakers
    }


def _compute_anchors(
    speakers: list[str],
    intervals_by_spk: dict[str, list[Interval]],
    enhanced: np.ndarray,
    ecapa,
    device: torch.device,
    sr: int,
    anchor_cap_s: Optional[float],
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> tuple[dict[str, Optional[torch.Tensor]], dict[str, float]]:
    """Compute one ECAPA anchor per speaker from their solo concat.

    The concat is capped at `anchor_cap_s` via `_cap_anchor_audio` because
    on long recordings (e.g. 949 s with 400+ s of solo per speaker) feeding
    ECAPA's `encode_batch` in a single forward OOMs the GPU or stalls the
    kernel.

    Returns `(anchors_by_spk, solo_duration_by_spk)`. `anchors[spk]` is None
    when the speaker doesn't have enough solo audio to embed
    (< `anchor_min_duration_s`, default 0.25 s).
    """
    anchors: dict[str, Optional[torch.Tensor]] = {}
    solo_durations: dict[str, float] = {}
    min_anchor_len = int(sr * anchor_min_duration_s)
    _log(f"computing speaker anchors via ECAPA (anchor_max={anchor_cap_s}s)...")
    for spk in speakers:
        t0 = time.perf_counter()
        slices: list[np.ndarray] = []
        for s, e in intervals_by_spk[spk]:
            lo = int(s * sr)
            hi = int(e * sr)
            if hi > lo:
                slices.append(enhanced[lo:hi].astype(np.float32))
        concat = (
            np.concatenate(slices) if slices else np.zeros(16, dtype=np.float32)
        )
        solo_durations[spk] = len(concat) / sr
        anchor_input = _cap_anchor_audio(concat, sr, anchor_cap_s)
        capped = len(anchor_input) < len(concat)
        _log(
            f"  speaker {spk!r}: {len(intervals_by_spk[spk])} solo intervals, "
            f"concat {len(concat)/sr:.1f}s"
            + (f", capped to {len(anchor_input)/sr:.1f}s" if capped else "")
            + " — calling ECAPA..."
        )
        if len(anchor_input) >= min_anchor_len:
            anchors[spk] = _ecapa_embed(
                anchor_input, ecapa, device, sr, anchor_min_duration_s
            ).cpu()
        else:
            anchors[spk] = None
        _log(
            f"  speaker {spk!r}: anchor done in {time.perf_counter()-t0:.2f}s "
            f"(anchor={'set' if anchors[spk] is not None else 'None (too short)'})"
        )
    return anchors, solo_durations


def _mixture_fill_overlaps(
    overlap_regions: list[Interval],
    speakers: list[str],
    audio: np.ndarray,
    sr: int,
) -> list[dict]:
    """No-separation mode: fill overlap regions with the (enhanced) mixture.

    Used when ``separation.enabled: false`` (or when stage 3b otherwise
    produced no separated streams). The same audio slice is attributed to
    every speaker — Whisper will double-transcribe the overlapping content,
    and cpWER pays the cost honestly via the per-speaker GT.

    Returns assignments with the same shape as :func:`_assign_overlaps`.
    """
    assignments: list[dict] = []
    for start_s, end_s in overlap_regions:
        # Clamp instead of dropping: pyannote turns can overshoot the
        # actual audio end by a fraction of a second (frame quantisation),
        # and an overlap region inheriting that overshoot would otherwise
        # lose its entire audio here.
        lo = int(start_s * sr)
        hi = min(int(end_s * sr), len(audio))
        if hi <= lo:
            continue
        clip = audio[lo:hi].astype(np.float32)
        emit_pieces = {spk: clip.copy() for spk in speakers}
        assignments.append({
            "orig_start": float(start_s),
            "orig_end": float(end_s),
            "pairing": "no_separation",
            "emit_pieces": emit_pieces,
        })
    return assignments


def _assign_overlaps(
    overlap_separated: list,
    anchors: dict[str, Optional[torch.Tensor]],
    speakers: list[str],
    ecapa,
    device: torch.device,
    sr: int,
    min_margin: float = 0.0,
    *,
    strategy: str = "ecapa_argmax",
    continuity_margin: float = 0.0,
    continuity_window_s: float = 10.0,
    solo_intervals_by_spk: Optional[dict[str, list[Interval]]] = None,
    assembly_audio: Optional[np.ndarray] = None,
    external_pairings: Optional[dict[int, str]] = None,
    overlap_min_duration_s: float = _OVERLAP_MIN_DURATION_S_DEFAULT,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
) -> list[dict]:
    """For each overlap, ECAPA-embed s1/s2 and pick the pairing with higher
    summed cosine similarity to the anchors. Falls back to fixed assignment
    when an anchor is missing or the streams are too short to embed
    (< 0.1 s) — never drops a region. Slices each picked stream to the
    emit region.

    ``strategy`` selects how a finite ECAPA decision is finalised:

    - ``"ecapa_argmax"`` (default): argmax(straight, swapped), with the optional
      ``min_margin`` carry-forward prior described below.
    - ``"continuity_tiebreak"``: argmax, but a near-tie (gap <
      ``continuity_margin``) is re-decided by :func:`_continuity_decision` using
      *local* anchors from the solo audio bracketing the overlap (needs
      ``solo_intervals_by_spk`` + ``assembly_audio``). ``min_margin`` is ignored
      in this mode.
    - ``"consensus_2means"``: a pre-pass (:func:`_consensus_pairings`) decides
      every overlap jointly via constrained 2-means over all overlap embeddings +
      the solo anchors, so the consensus can overrule a lone confident-but-wrong
      per-overlap decision. The loop then looks up each overlap's pairing.

    ``external_pairings`` (B+ handoff, ``ctx.overlap_speaker_assignment``):
    `{i_ovl -> "straight"|"swapped"}` decided up front by the RelabelStage global
    clustering. When present, an ECAPA-eligible overlap whose ``i_ovl`` is in the
    dict uses the external pairing directly (NO per-overlap re-embed), labelled
    ``"<chosen> (relabel_global)"`` — it takes precedence over ALL three
    ``strategy`` modes for the overlaps it covers (a strictly stronger global
    decision). Overlaps NOT in the dict (B+ dropped them as sub-min / leaked /
    degenerate, or they are ECAPA-ineligible here) fall through to the existing
    ladder unchanged — the SCOPE-compliant fall-soft to current behaviour.

    ``min_margin`` (default 0.0 = off, ``ecapa_argmax`` only) gates a
    carry-forward prior on the ECAPA path: when ``abs(straight - swapped) <
    min_margin`` the ECAPA decision is too close to trust (short overlaps where
    the cosines near-tie), so the overlap inherits ``last_pairing`` — the most
    recent *confident* (margin-clearing) ECAPA decision — instead of the noisy
    argmax. With the default margin of 0, every finite decision clears the (zero)
    gap and the prior never fires, so behaviour is the pure-argmax baseline. The
    first confident decision seeds the prior; ambiguous overlaps before any
    confident one (or when the gate is off) fall through to plain argmax.

    Returns one assignment dict per overlap: `{orig_start, orig_end, pairing,
    emit_pieces: {speaker: audio_np}}`.
    """
    n = len(overlap_separated)
    _log(f"assigning {n} overlaps to speakers via ECAPA cosine...")
    if len(speakers) > 2:
        # Overlap attribution only ever assigns the two separator streams to
        # speakers[:2]; speakers 3..n would silently get no overlap audio.
        # num_speakers=2 is pinned end-to-end (diarization, SepFormer, CLARIN),
        # so a third speaker is off-distribution — but warn rather than drop
        # audio in silence.
        _log(
            f"WARNING: {len(speakers)} speakers but overlap assignment handles "
            f"only 2 — speakers {speakers[2:]} get no overlap audio."
        )
    t_start = time.perf_counter()
    min_overlap_len = int(sr * overlap_min_duration_s)
    assignments: list[dict] = []
    # Carry-forward prior for the margin gate: the last confident (margin-clearing)
    # ECAPA pairing, "straight" or "swapped". None until the first confident
    # decision. Only consulted when min_margin > 0 (the gate is off at default).
    last_pairing: Optional[str] = None
    # Consensus 2-means decides every overlap jointly up front (one pre-pass over
    # all overlaps); the per-overlap loop below then just looks up its pairing.
    consensus = (
        _consensus_pairings(
            overlap_separated, anchors, speakers, ecapa, device, sr,
            overlap_min_duration_s=overlap_min_duration_s,
            anchor_min_duration_s=anchor_min_duration_s,
        )
        if strategy == "consensus_2means" else None
    )
    for i_ovl, ovl in enumerate(overlap_separated):
        if "s1_gated" not in ovl or "s2_gated" not in ovl:
            raise RuntimeError(
                "overlap_separated entries have no 's1_gated'/'s2_gated' — "
                "run the post_separation_processing stage before assembly."
            )
        too_short = len(ovl["s1_gated"]) < min_overlap_len
        have_both_anchors = (
            len(speakers) >= 2
            and all(anchors.get(s) is not None for s in speakers[:2])
        )
        if not too_short and have_both_anchors:
            a, b = speakers[0], speakers[1]
            if external_pairings is not None and i_ovl in external_pairings:
                # B+ global-clustering handoff: use the pre-decided pairing
                # directly (no per-overlap re-embed). Strictly stronger than the
                # per-overlap strategies, so it overrides them for this overlap.
                chosen = external_pairings[i_ovl]
                pairing = f"{chosen} (relabel_global)"
                stream_for = (
                    {a: ovl["s1_gated"], b: ovl["s2_gated"]} if chosen == "straight"
                    else {a: ovl["s2_gated"], b: ovl["s1_gated"]}
                )
            elif consensus is not None and i_ovl in consensus:
                # Global consensus (2-means) pre-decided this overlap from ALL
                # overlaps jointly; use it directly (its embeddings were computed
                # in the pre-pass, so no per-overlap re-embed is needed here).
                chosen = consensus[i_ovl]
                pairing = f"{chosen} (consensus)"
                stream_for = (
                    {a: ovl["s1_gated"], b: ovl["s2_gated"]} if chosen == "straight"
                    else {a: ovl["s2_gated"], b: ovl["s1_gated"]}
                )
            else:
                # ECAPA path: embed both streams and pick the pairing with the
                # higher *summed* cosine similarity to the two anchors. (Embedding
                # is deferred to here so a too-short / anchor-missing overlap pays
                # no ECAPA forward.)
                emb1 = _ecapa_embed(
                    ovl["s1_gated"], ecapa, device, sr, anchor_min_duration_s
                ).cpu()
                emb2 = _ecapa_embed(
                    ovl["s2_gated"], ecapa, device, sr, anchor_min_duration_s
                ).cpu()
                straight = _cos(emb1, anchors[a]) + _cos(emb2, anchors[b])
                swapped = _cos(emb1, anchors[b]) + _cos(emb2, anchors[a])
                if not (np.isfinite(straight) and np.isfinite(swapped)):
                    # ECAPA is an external model fed degenerate gated input; a
                    # non-finite cosine (e.g. a NaN embedding) would make
                    # `straight >= swapped` evaluate False and silently pick
                    # "swapped" as if it were a real decision. Drop to the fixed
                    # fallback instead — the one system-boundary check here.
                    _log(
                        f"  overlap {ovl['idx']}: non-finite ECAPA cosine "
                        f"(straight={straight}, swapped={swapped}) — fixed assignment"
                    )
                    stream_for = {a: ovl["s1_gated"], b: ovl["s2_gated"]}
                    pairing = "arbitrary (non-finite cosine)"
                else:
                    # Argmax pairing (the `>=` tie-break keeps stream order).
                    argmax_pairing = "straight" if straight >= swapped else "swapped"
                    if strategy == "continuity_tiebreak":
                        # Near-ties re-decided by local (bracketing-solo) anchors;
                        # decisive gaps keep the global argmax. Stateless.
                        chosen, pairing = _continuity_decision(
                            argmax_pairing, straight, swapped, continuity_margin,
                            emb1, emb2, a, b, ovl, continuity_window_s,
                            solo_intervals_by_spk, assembly_audio,
                            ecapa, device, sr, anchor_min_duration_s,
                        )
                    # Margin gate (off when min_margin == 0): a near-tie ECAPA
                    # decision is unreliable on short overlaps, so when the gap is
                    # below the margin AND we already have a confident prior, inherit
                    # it instead of trusting the argmax. A confident (margin-clearing)
                    # decision updates the prior for later ambiguous overlaps.
                    elif (
                        min_margin > 0
                        and abs(straight - swapped) < min_margin
                        and last_pairing is not None
                    ):
                        pairing = f"{last_pairing} (carry-forward prior)"
                        chosen = last_pairing
                    else:
                        pairing = argmax_pairing
                        chosen = argmax_pairing
                        if min_margin > 0 and abs(straight - swapped) >= min_margin:
                            last_pairing = argmax_pairing
                    if chosen == "straight":
                        stream_for = {a: ovl["s1_gated"], b: ovl["s2_gated"]}
                    else:
                        stream_for = {a: ovl["s2_gated"], b: ovl["s1_gated"]}
        else:
            # Fixed assignment when the streams are too short to embed or an
            # anchor is missing. Never drop the region — a drop would lose the
            # audio for BOTH speakers, because the emit region has already
            # been subtracted from the solo intervals.
            # Author ruling (2026-06-11, SCOPE §10 q3): these degenerate cases
            # (sub-0.25 s solo anchors, sub-0.1 s overlaps, non-finite cosines)
            # are too rare and information-poor to act on better — the fixed
            # positional pairing is the accepted behavior, kept as-is.
            if too_short:
                _log(
                    f"  overlap {ovl['idx']}: too short for ECAPA "
                    f"({len(ovl['s1_gated'])/sr:.3f}s) — fixed assignment"
                )
            stream_for = {}
            if len(speakers) >= 1:
                stream_for[speakers[0]] = ovl["s1_gated"]
            if len(speakers) >= 2:
                stream_for[speakers[1]] = ovl["s2_gated"]
            pairing = (
                "arbitrary (too short)" if too_short
                else "arbitrary (weak anchor)"
            )

        emit_pieces = {
            spk: _slice_emit(
                audio, ovl["pad_start"], ovl["emit_start"], ovl["emit_end"], sr
            )
            for spk, audio in stream_for.items()
        }
        assignments.append({
            "orig_start": float(ovl["emit_start"]),
            "orig_end": float(ovl["emit_end"]),
            "pairing": pairing,
            "emit_pieces": emit_pieces,
        })
        if (i_ovl + 1) % 10 == 0 or i_ovl + 1 == n:
            _log(
                f"  assigned {i_ovl+1}/{n} overlaps "
                f"({time.perf_counter()-t_start:.1f}s elapsed)"
            )
    return assignments


def _build_events(
    speakers: list[str],
    solo_intervals_by_spk: dict[str, list[Interval]],
    assignments: list[dict],
    enhanced: np.ndarray,
    sr: int,
) -> dict[str, list[dict]]:
    """Combine per-speaker solos (sliced from enhanced full) with overlap
    assignments into one sorted event list per speaker.

    Each event: `{orig_start, orig_end, audio: np.ndarray, kind: "solo"|"overlap"}`.
    """
    events: dict[str, list[dict]] = {spk: [] for spk in speakers}
    for spk in speakers:
        for s, e in solo_intervals_by_spk[spk]:
            lo = int(s * sr)
            hi = int(e * sr)
            if hi <= lo:
                continue
            events[spk].append({
                "orig_start": float(s),
                "orig_end": float(e),
                "audio": enhanced[lo:hi].astype(np.float32),
                "kind": "solo",
            })
    for a in assignments:
        for spk, audio in a["emit_pieces"].items():
            if audio.size == 0:
                continue
            events[spk].append({
                "orig_start": a["orig_start"],
                "orig_end": a["orig_end"],
                "audio": audio,
                "kind": "overlap",
            })
    for spk in speakers:
        events[spk].sort(key=lambda e: e["orig_start"])
    return events


def _apply_per_piece_post(
    events_by_spk: dict[str, list[dict]],
    cfg: AssemblyConfig,
    sr: int,
) -> None:
    """Apply per-piece post-processing in place: overlap→solo RMS match,
    optional aggressive RMS normalisation, then edge / crossfade.
    """
    if cfg.overlap_rms_match_solo:
        for spk in events_by_spk:
            _match_overlap_rms_to_solo(events_by_spk[spk])
    if cfg.per_piece_rms_norm:
        for spk in events_by_spk:
            rescaled = _rms_normalise(
                [e["audio"] for e in events_by_spk[spk]], cfg.target_rms
            )
            for e, r in zip(events_by_spk[spk], rescaled):
                e["audio"] = r.astype(np.float32)
    if cfg.crossfade_ms > 0 or cfg.edge_fade_ms > 0:
        # crossfade_ms applies at internal piece-to-piece seams; edge_fade_ms
        # at the very start of the first piece and very end of the last piece
        # (only neighbour there is silence outside the stream, so the fade can
        # be shorter).
        crossfade_n = int(cfg.crossfade_ms * sr / 1000)
        edge_fade_n = int(cfg.edge_fade_ms * sr / 1000)
        for spk in events_by_spk:
            spk_events = events_by_spk[spk]
            n_events = len(spk_events)
            for i, e in enumerate(spk_events):
                in_n = edge_fade_n if i == 0 else crossfade_n
                out_n = edge_fade_n if i == n_events - 1 else crossfade_n
                e["audio"] = _apply_fade(e["audio"], in_n, out_n)


def _concat_shortened(
    events_spk: list[dict], cfg: AssemblyConfig, sr: int
) -> tuple[np.ndarray, list[TimestampMapEntry]]:
    """Speech-only concat with `silence_separator_s` gaps. Returns
    `(stream, timestamp_map_entries)`.
    """
    gap_samples = int(cfg.silence_separator_s * sr)
    pieces: list[np.ndarray] = []
    tmap: list[TimestampMapEntry] = []
    cursor = 0.0
    for ev in events_spk:
        if pieces:
            pieces.append(np.zeros(gap_samples, dtype=np.float32))
            cursor += gap_samples / sr
        pieces.append(ev["audio"])
        dur = len(ev["audio"]) / sr
        tmap.append(TimestampMapEntry(
            concat_start=cursor,
            concat_end=cursor + dur,
            orig_start=ev["orig_start"],
            orig_end=ev["orig_end"],
            kind=ev["kind"],
        ))
        cursor += dur
    audio = (
        np.concatenate(pieces).astype(np.float32)
        if pieces
        else np.zeros(sr, dtype=np.float32)
    )
    return audio, tmap


def _concat_full_length(
    events_spk: list[dict], total_dur_s: float, sr: int
) -> tuple[np.ndarray, list[TimestampMapEntry]]:
    """Total stream length = input recording length; gaps filled with silence.
    Pieces placed at their original timestamps.
    """
    total_samples = int(total_dur_s * sr)
    stream = np.zeros(total_samples, dtype=np.float32)
    tmap: list[TimestampMapEntry] = []
    for ev in events_spk:
        start_idx = int(ev["orig_start"] * sr)
        if start_idx >= total_samples:
            continue
        audio = ev["audio"]
        if start_idx < 0:
            # Mirror `_slice_emit`'s lower clamp: a negative orig_start would
            # otherwise index from the array end and leave a 0-length
            # destination slice against a positive-length source — a hard
            # broadcast ValueError. Trim the audio head so shapes stay aligned.
            audio = audio[-start_idx:]
            start_idx = 0
        end_idx = min(total_samples, start_idx + len(audio))
        audio = audio[: end_idx - start_idx]
        stream[start_idx:end_idx] = audio
        tmap.append(TimestampMapEntry(
            concat_start=ev["orig_start"],
            concat_end=ev["orig_start"] + len(audio) / sr,
            orig_start=ev["orig_start"],
            orig_end=ev["orig_end"],
            kind=ev["kind"],
        ))
    return stream, tmap


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class AssemblyStage(Stage):
    name = "assembly"

    def __init__(self, config: AssemblyConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._ecapa = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        t0 = time.perf_counter()
        if self.config.anchor_embedding == "ecapa2":
            # Option 4: route the anchor / per-overlap embedder through the same
            # custom-embedder wrapper the diarization stage uses. `_ecapa_embed`
            # dispatches on the absence of `encode_batch` (the wrapper is
            # call-style), so no other call site changes.
            from asr_pipeline.stages.custom_embeddings import build_custom_embedding

            _log(f"load: building custom anchor embedder 'ecapa2' on {device}...")
            embedder = build_custom_embedding("ecapa2", device)
            if embedder is None:  # defensive — "ecapa2" is a known custom name
                raise RuntimeError(
                    "build_custom_embedding('ecapa2') returned None — the custom "
                    "embedder name is not recognised."
                )
            self._ecapa = embedder
        else:
            from speechbrain.inference.speaker import EncoderClassifier

            _log(f"load: instantiating ECAPA encoder on {device}...")
            cache_dir = Path.cwd() / ".cache" / "ecapa"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(device)},
                savedir=str(cache_dir),
            )
            ecapa.eval()
            self._ecapa = ecapa
        self._device = device
        _log(
            f"load: anchor embedder ready "
            f"({self.config.anchor_embedding}, {time.perf_counter()-t0:.2f}s)"
        )

    def load_signature(self) -> tuple:
        # Was () (model never depended on config). Now the embedder choice picks
        # WHICH model loads, so it must trigger an interactive-API reload when
        # the knob flips.
        return (self.config.anchor_embedding,)

    def unload(self) -> None:
        self._ecapa = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        # Logged unconditionally so we can confirm we entered run() at all
        # even when something later hangs. (Used to diagnose long-recording
        # stalls where the kernel froze before any other logging printed.)
        _log("run: entered")
        if self._ecapa is None or self._device is None:
            raise RuntimeError("AssemblyStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError(
                "AssemblyStage.run requires ctx.audio (load_audio must run)."
            )
        if ctx.diarization is None:
            raise RuntimeError(
                "AssemblyStage.run requires ctx.diarization (DiarizationStage must run first)."
            )
        if ctx.overlap_regions is None:
            raise RuntimeError(
                "AssemblyStage.run requires ctx.overlap_regions (RoutingStage must run first)."
            )
        if not ctx.speakers:
            # Nothing to assemble — keep ctx fields at their defaults.
            ctx.timestamp_map = TimestampMap(per_speaker={})
            return

        cfg = self.config
        sr = ctx.sample_rate
        # Pick the audio source for ECAPA anchors, solo slicing, and (in
        # no-separation mode) overlap fills. Enhanced when Stage 3a ran;
        # the raw mixture otherwise — this is what `enhancement.enabled:
        # false` looks like to the assembler.
        if ctx.enhanced_full is not None:
            assembly_audio = ctx.enhanced_full
            audio_source = "enhanced_full"
        else:
            assembly_audio = ctx.audio
            audio_source = "raw mixture (enhancement disabled)"
        speakers = ctx.speakers
        spk_to_label = {spk: chr(ord("A") + i) for i, spk in enumerate(speakers)}

        n_overlaps = len(ctx.overlap_separated) if ctx.overlap_separated else 0
        _log(
            f"start: {len(speakers)} speakers, {n_overlaps} overlap regions, "
            f"recording {len(assembly_audio)/sr:.1f}s, audio source={audio_source}"
        )

        # Phase 1: per-speaker solo intervals (pyannote segments minus emits).
        solo_intervals_by_spk = _derive_solo_intervals(ctx, speakers)

        # Phase 2: ECAPA anchor per speaker.
        anchors, solo_durations = _compute_anchors(
            speakers, solo_intervals_by_spk, assembly_audio,
            self._ecapa, self._device, sr, cfg.anchor_max_duration_s,
            anchor_min_duration_s=cfg.anchor_min_duration_s,
        )
        weak_anchor = any(
            d < cfg.min_solo_for_anchor_s for d in solo_durations.values()
        )
        if weak_anchor:
            _log(
                f"weak_anchor=True (min solo duration "
                f"{min(solo_durations.values()):.2f}s "
                f"< min_solo_for_anchor_s={cfg.min_solo_for_anchor_s})"
            )

        # Phase 3: per-overlap speaker assignment.
        # Two modes:
        #   - Normal: stage 3b produced separated streams; ECAPA picks the
        #     pairing per overlap.
        #   - No-separation (`separation.enabled: false` in config, or 3b
        #     otherwise produced nothing): fill overlap regions from the
        #     enhanced mixture, attributed to all speakers.
        if ctx.overlap_separated:
            if ctx.overlap_speaker_assignment is not None:
                _log(
                    f"B+ handoff: relabel pre-decided "
                    f"{len(ctx.overlap_speaker_assignment)}/"
                    f"{len(ctx.overlap_separated)} overlap(s) "
                    f"(relabel_global); the rest use the anchor ladder."
                )
            assignments = _assign_overlaps(
                ctx.overlap_separated, anchors, speakers,
                self._ecapa, self._device, sr,
                min_margin=cfg.overlap_assign_min_margin,
                strategy=cfg.overlap_assignment,
                continuity_margin=cfg.continuity_tiebreak_margin,
                continuity_window_s=cfg.continuity_window_s,
                solo_intervals_by_spk=solo_intervals_by_spk,
                assembly_audio=assembly_audio,
                external_pairings=ctx.overlap_speaker_assignment,
                overlap_min_duration_s=cfg.overlap_min_duration_s,
                anchor_min_duration_s=cfg.anchor_min_duration_s,
            )
        elif ctx.overlap_regions:
            _log(
                f"no separated streams; filling "
                f"{len(ctx.overlap_regions)} overlap region(s) from the "
                f"{audio_source} (all speakers)"
            )
            assignments = _mixture_fill_overlaps(
                ctx.overlap_regions, speakers, assembly_audio, sr,
            )
        else:
            assignments = []

        # Phase 4: combine solo + overlap events.
        _log("building per-speaker event lists + post-processing...")
        events = _build_events(
            speakers, solo_intervals_by_spk, assignments, assembly_audio, sr
        )

        # Phase 5: per-piece RMS match / norm / fades (in place).
        _apply_per_piece_post(events, cfg, sr)

        # Phase 6: concatenate per speaker (mode-dependent).
        _log(f"concatenating per-speaker streams (mode={cfg.output_mode})...")
        total_dur = ctx.diarization.total_duration_s if ctx.diarization else 0.0
        assembled: dict[str, np.ndarray] = {}
        timestamp_map = TimestampMap(per_speaker={})
        for spk in speakers:
            if cfg.output_mode == "shortened":
                stream, tmap = _concat_shortened(events[spk], cfg, sr)
            elif cfg.output_mode == "full_length":
                stream, tmap = _concat_full_length(events[spk], total_dur, sr)
            else:
                # Should be unreachable: PipelineConfig.__post_init__ validates.
                raise ValueError(f"Unknown output_mode: {cfg.output_mode!r}")
            assembled[spk] = stream
            timestamp_map.per_speaker[spk] = tmap

        ctx.assembled = assembled
        ctx.spk_to_label = spk_to_label
        ctx.timestamp_map = timestamp_map
        ctx.weak_anchor = weak_anchor
        for spk, audio in assembled.items():
            _log(
                f"  speaker {spk!r} ({spk_to_label.get(spk, spk)}): "
                f"{len(audio)/sr:.2f}s assembled, "
                f"{len(timestamp_map.per_speaker.get(spk, []))} pieces"
            )
        _log("done.")

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.assembled:
            return
        for spk, audio in ctx.assembled.items():
            label = ctx.spk_to_label.get(spk, spk)
            sf.write(
                artifact_dir / f"assembled_{label}.wav",
                audio.astype(np.float32),
                ctx.sample_rate,
            )
        if ctx.timestamp_map is not None:
            payload = {
                "weak_anchor": ctx.weak_anchor,
                "spk_to_label": ctx.spk_to_label,
                "output_mode": self.config.output_mode,
                "per_speaker": {
                    spk: [
                        {
                            "concat_start": e.concat_start,
                            "concat_end": e.concat_end,
                            "orig_start": e.orig_start,
                            "orig_end": e.orig_end,
                            "kind": e.kind,
                        }
                        for e in entries
                    ]
                    for spk, entries in ctx.timestamp_map.per_speaker.items()
                },
            }
            with open(artifact_dir / "timestamp_map.json", "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
