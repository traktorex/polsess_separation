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
   for the speaker, minus `ctx.overlap_regions`. No padding by default —
   pyannote's boundaries are used as-is. (`solo_onset_pad_s` > 0 extends
   piece STARTS at extraction time only, clamped to never enter overlap
   regions or adjacent pieces; see `_pad_solo_onsets`.)
2. Build an ECAPA-TDNN *anchor* embedding from a concatenation of
   `enhanced_full` sliced at those solo intervals. If any speaker has
   less than `weak_anchor_warn_below_s` of solo audio, set the diagnostic
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

Speaker assignment uses per-overlap ECAPA argmax against the solo anchors,
with the RelabelStage's B+ global-clustering handoff (`external_pairings`)
taking precedence for the overlaps it covers. All dispatch inside
`_assign_overlaps`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

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


def _diag_cos(value: Optional[float]) -> Optional[float]:
    """A cosine sum as a plain float for the diagnostic, else None.

    ``None`` in means the pairing was decided without cosines (B+ handoff or a
    fixed fallback). Non-finite in means ECAPA produced a degenerate embedding —
    which `json.dumps` would write as a bare `NaN`/`Infinity` literal that
    strict JSON parsers reject, so it becomes `null` too.
    """
    if value is None or not np.isfinite(value):
        return None
    return float(value)


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


def _emit_blocked_regions(ctx: PipelineContext) -> list[Interval]:
    """The intervals solo audio must not come from: Stage 3b's emit regions when
    they exist, else the raw routing overlap regions. Shared by
    `_derive_solo_intervals` (which subtracts them from the speaker segments)
    and `_pad_solo_onsets` (whose onset pad must never reach into them), so the
    two can't drift apart.
    """
    if ctx.overlap_separated:
        return [
            (float(o["emit_start"]), float(o["emit_end"]))
            for o in ctx.overlap_separated
        ]
    return ctx.overlap_regions or []


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
    blocked = _emit_blocked_regions(ctx)
    return {
        spk: _speaker_solo_intervals(ctx.diarization.segments_df, spk, blocked)
        for spk in speakers
    }


def _pad_solo_onsets(
    solo_intervals_by_spk: dict[str, list[Interval]],
    blocked: list[Interval],
    pad_s: float,
) -> dict[str, list[Interval]]:
    """Extend each solo piece's START earlier by up to ``pad_s`` seconds.

    Motivation (ear pass, 2026-07): pyannote turn starts lag true speech onsets
    slightly, and the assembler slices exactly at the diarization boundary — so
    first phonemes get shaved ("szefie" audible as "efie") or onset slivers get
    split into / duplicated in the other stream. A small bounded pad recovers
    them. Onset side only: piece ENDS are never moved (offsets did not show the
    failure, and padding ends would double speech against the next piece's
    onset). Kept deliberately simpler than the seam logic's extend-toward-
    silence: a fixed bounded pad, no VAD.

    Hard clamps — the padded start is the LATEST of:

    - ``start − pad_s``;
    - 0.0;
    - the end of any ``blocked`` (overlap emit) region before the start — that
      audio contains BOTH speakers, so injecting it into a single-speaker
      stream would leak the other speaker past separation;
    - the end of ANY adjacent solo piece, in EITHER stream. Same-stream: the
      previous piece must not be overwritten (full_length mode places pieces at
      their padded original times). Other-stream: routing can drop overlaps
      shorter than ``min_overlap_dur``, so the other speaker's solo span can sit
      (or even reach) directly before this piece — padding into it would inject
      their voice, the exact failure the overlap clamp prevents.

    ``pad_s <= 0`` returns the input unchanged (the byte-identical no-op that
    the 0.0 default pins). Pure and deterministic — plain interval arithmetic.
    """
    if pad_s <= 0:
        return solo_intervals_by_spk
    # Everything the pad must not reach into: all speakers' solo pieces plus
    # the blocked (overlap) regions.
    occupied: list[Interval] = list(blocked)
    for intervals in solo_intervals_by_spk.values():
        occupied.extend(intervals)
    out: dict[str, list[Interval]] = {}
    for spk, intervals in solo_intervals_by_spk.items():
        padded: list[Interval] = []
        for s, e in intervals:
            new_start = max(0.0, s - pad_s)
            for o_s, o_e in occupied:
                # `o_s < s` skips the piece itself (its own start IS `s`) and
                # anything starting at/after our start (irrelevant to an onset
                # pad). An occupied interval overlapping our start (o_e > s —
                # possible for the other stream when routing dropped a short
                # overlap) clamps the pad away entirely (min(o_e, s) == s).
                if o_s < s and o_e > new_start:
                    new_start = max(new_start, min(float(o_e), s))
            padded.append((float(new_start), float(e)))
        out[spk] = padded
    return out


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
    *,
    external_pairings: Optional[dict[int, str]] = None,
    overlap_min_duration_s: float = _OVERLAP_MIN_DURATION_S_DEFAULT,
    anchor_min_duration_s: float = _ANCHOR_MIN_DURATION_S_DEFAULT,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """For each overlap, ECAPA-embed s1/s2 and pick the pairing with higher
    summed cosine similarity to the anchors (``ecapa_argmax``). Falls back to
    fixed assignment when an anchor is missing or the streams are too short to
    embed (< 0.1 s) — never drops a region. Slices each picked stream to the
    emit region.

    ``external_pairings`` (B+ handoff, ``ctx.overlap_speaker_assignment``):
    `{i_ovl -> "straight"|"swapped"}` decided up front by the RelabelStage global
    clustering. When present, an ECAPA-eligible overlap whose ``i_ovl`` is in the
    dict uses the external pairing directly (NO per-overlap re-embed), labelled
    ``"<chosen> (relabel_global)"`` — a strictly stronger global decision that
    takes precedence over the per-overlap argmax. Overlaps NOT in the dict (B+
    dropped them as sub-min / leaked / degenerate, or they are ECAPA-ineligible
    here) fall through to the per-overlap argmax unchanged — the SCOPE-compliant
    fall-soft to current behaviour.

    ``progress`` is an optional ``(done, total)`` sink called once per overlap
    (the orchestrator's `stage_progress` plumbing); ``None`` = silent.

    Returns one assignment dict per overlap: `{orig_start, orig_end, pairing,
    emit_pieces: {speaker: audio_np}, diag}`. ``diag`` is the JSON-safe
    attribution record `{idx, pairing, cos_straight, cos_swapped}` that
    `AssemblyStage.run` lifts onto `ctx.assembly_diag` (cosines are `None`
    whenever the pairing was not decided by an ECAPA argmax).
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
    for i_ovl, ovl in enumerate(overlap_separated):
        if "s1_gated" not in ovl or "s2_gated" not in ovl:
            raise RuntimeError(
                "overlap_separated entries have no 's1_gated'/'s2_gated' — "
                "run the post_separation_processing stage before assembly."
            )
        # Diagnostic cosines: filled in on the ECAPA path only, so they stay
        # None for the B+ handoff and the fixed fallbacks.
        cos_straight: Optional[float] = None
        cos_swapped: Optional[float] = None
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
                cos_straight, cos_swapped = straight, swapped
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
                    pairing = "straight" if straight >= swapped else "swapped"
                    if pairing == "straight":
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
            # Attribution diagnostic — the decision without the audio, JSON-safe.
            # `idx` is the routing-region index (joins to routing.json's
            # `overlap_regions`), not this loop's dense position.
            "diag": {
                "idx": int(ovl["idx"]),
                "pairing": pairing,
                "cos_straight": _diag_cos(cos_straight),
                "cos_swapped": _diag_cos(cos_swapped),
            },
        })
        if (i_ovl + 1) % 10 == 0 or i_ovl + 1 == n:
            _log(
                f"  assigned {i_ovl+1}/{n} overlaps "
                f"({time.perf_counter()-t_start:.1f}s elapsed)"
            )
        if progress is not None:
            progress(i_ovl + 1, n)
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
        # Local import mirrors `load` — keeps custom_embeddings' pyannote import
        # off assembly's module-load path.
        from asr_pipeline.stages.custom_embeddings import release_gpu_memory

        self._ecapa = None
        self._device = None
        release_gpu_memory()

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
            d < cfg.weak_anchor_warn_below_s for d in solo_durations.values()
        )
        if weak_anchor:
            _log(
                f"weak_anchor=True (min solo duration "
                f"{min(solo_durations.values()):.2f}s "
                f"< weak_anchor_warn_below_s={cfg.weak_anchor_warn_below_s})"
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
                external_pairings=ctx.overlap_speaker_assignment,
                overlap_min_duration_s=cfg.overlap_min_duration_s,
                anchor_min_duration_s=cfg.anchor_min_duration_s,
                progress=self._progress,
            )
            # Attribution diagnostics: one compact record per overlap (pairing +
            # the two ECAPA cosine sums, no audio). Mirrors `ctx.diarization_diag`
            # — io.write_pipeline_outputs writes it verbatim into metadata.json.
            ctx.assembly_diag = [a["diag"] for a in assignments]
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
        # Onset pad (solo_onset_pad_s > 0 only; 0.0 default returns the input
        # unchanged): extend solo piece STARTS slightly earlier at extraction so
        # shaved first phonemes are recovered. Applied HERE only — the anchors
        # and the continuity intervals computed above keep the unpadded
        # diarization boundaries.
        solo_event_intervals = _pad_solo_onsets(
            solo_intervals_by_spk, _emit_blocked_regions(ctx),
            cfg.solo_onset_pad_s,
        )
        events = _build_events(
            speakers, solo_event_intervals, assignments, assembly_audio, sr
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
