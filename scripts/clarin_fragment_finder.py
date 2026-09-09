"""Find candidate test fragments in CLARIN 2-speaker recordings.

Goal: produce 0.5–2.5 minute regions where (a) the separator has work to
do (real overlap), (b) both speakers contribute (conversational content,
not 95 % one speaker), and (c) the boundaries don't cut mid-utterance
(snapped to silence where possible).

Algorithm (intentionally simple — designed to be inspected and tuned in
``asr/clarin_fragments.ipynb``):

1. Sweep-line over the pyannote diarization → *overlap timeline*
   (intervals where ≥ 2 speakers are simultaneously active).
2. Merged speech timeline → silence gaps (≥ ``silence_snap_min_s`` long).
3. Slide a window of ``target_length_s`` across the recording with stride
   ``stride_s``. For each position, snap each boundary outward to the
   nearest silence within ``silence_snap_search_s`` (no snap if no silence
   in range). Discard if the resulting length leaves [min_length_s, max_length_s].
4. Filter: total overlap ≥ ``min_overlap_s``, weaker-speaker share
   ≥ ``min_speaker_balance``, speech density ≥ ``min_speech_density``.
5. Non-max-suppress overlapping candidates (IoU > ``nms_iou_threshold``),
   keeping the one with the most overlap.

This module has zero audio dependencies — it operates purely on diarization
JSON. The notebook adds visualisation + audio playback on top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Turn(NamedTuple):
    """One pyannote diarization segment."""
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class FragmentParams:
    """All knobs in one place. Defaults are sensible starting points;
    tune in the notebook before batch extraction."""

    # Length envelope (fragment.duration always lands in this range).
    target_length_s: float = 90.0
    min_length_s: float = 30.0
    max_length_s: float = 150.0

    # Sliding-window stride (smaller = denser candidates, slower).
    stride_s: float = 5.0

    # Silence handling.
    silence_snap_min_s: float = 0.5      # min gap to count as silence
    silence_snap_search_s: float = 5.0   # boundary snap search radius

    # Per-fragment filters.
    min_overlap_s: float = 3.0
    min_speaker_balance: float = 0.15    # weaker speaker's share of speech time
    min_speech_density: float = 0.40     # speech time / fragment duration

    # Non-max-suppress IoU threshold (lower = more fragments per recording).
    nms_iou_threshold: float = 0.30


@dataclass
class Fragment:
    """One identified test fragment.

    ``speech_s_by_spk`` records each speaker's talking time within the
    fragment — these can sum to > duration when speakers overlap.
    ``union_speech_s`` is the time *any* speaker is active (≤ duration),
    used for the speech-density filter so values stay interpretable as a
    fraction.
    """

    start: float
    end: float
    overlap_s: float                            # total overlap duration in [start, end]
    n_overlap_events: int                       # distinct overlap intervals
    max_event_s: float                          # duration of the longest single overlap event
    union_speech_s: float                       # union of speaker timelines (≤ duration)
    speech_s_by_spk: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_density(self) -> float:
        """Fraction of fragment containing any speech (always ≤ 1.0)."""
        return self.union_speech_s / self.duration if self.duration > 0 else 0.0

    @property
    def overlap_density(self) -> float:
        return self.overlap_s / self.duration if self.duration > 0 else 0.0

    @property
    def speaker_balance(self) -> float:
        """Weaker speaker's share of per-speaker talking time (0–0.5).

        Computed from `speech_s_by_spk`, so a fragment where both speakers
        talk equally returns 0.5; one where SPEAKER_00 dominates returns
        a small number.
        """
        total = sum(self.speech_s_by_spk.values())
        if not self.speech_s_by_spk or total == 0:
            return 0.0
        return min(self.speech_s_by_spk.values()) / total

    def summary_dict(self) -> dict:
        """Flat dict for CSV / DataFrame display."""
        return {
            "start_s": round(self.start, 2),
            "end_s": round(self.end, 2),
            "duration_s": round(self.duration, 2),
            "overlap_s": round(self.overlap_s, 2),
            "overlap_density": round(self.overlap_density, 3),
            "n_overlap_events": self.n_overlap_events,
            "max_event_s": round(self.max_event_s, 2),
            "speech_density": round(self.speech_density, 3),
            "speaker_balance": round(self.speaker_balance, 3),
            **{
                f"spk_{k}_s": round(v, 2)
                for k, v in sorted(self.speech_s_by_spk.items())
            },
        }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_diarization(path: str | Path) -> tuple[list[Turn], float]:
    """Read a pyannote diarization JSON. Returns (turns_sorted_by_start, total_duration_s).

    Expected schema::

        {"total_duration_s": 1234.5,
         "segments": [{"start": 0.0, "end": 1.2, "duration": 1.2,
                       "speaker": "SPEAKER_00"}, ...]}
    """
    data = json.loads(Path(path).read_text())
    turns = [
        Turn(start=float(s["start"]), end=float(s["end"]),
             speaker=str(s["speaker"]))
        for s in data.get("segments", [])
    ]
    turns.sort(key=lambda t: t.start)
    total = float(data.get("total_duration_s") or (turns[-1].end if turns else 0.0))
    return turns, total


# ---------------------------------------------------------------------------
# Timeline computations
# ---------------------------------------------------------------------------


def overlap_intervals(turns: list[Turn]) -> list[tuple[float, float]]:
    """Sweep-line: find intervals where ≥ 2 distinct speakers are active.

    Each pyannote turn contributes a +1 event at its start and a -1 at its
    end (per-speaker counter). Whenever the number of distinct active
    speakers crosses 2 (going up), we open an overlap interval; whenever
    it drops below 2 (going down), we close one.
    """
    events: list[tuple[float, int, str]] = []
    for t in turns:
        events.append((t.start, +1, t.speaker))
        events.append((t.end,   -1, t.speaker))
    # Ties: process ends before starts at the same time so a 0-duration
    # transition doesn't spuriously open an overlap.
    events.sort(key=lambda e: (e[0], -e[1]))

    active: dict[str, int] = {}
    overlaps: list[tuple[float, float]] = []
    overlap_start: float | None = None

    for t, delta, spk in events:
        if delta == +1:
            active[spk] = active.get(spk, 0) + 1
        else:
            active[spk] = active.get(spk, 0) - 1
            if active[spk] <= 0:
                active.pop(spk, None)
        n_distinct = len(active)
        if n_distinct >= 2 and overlap_start is None:
            overlap_start = t
        elif n_distinct < 2 and overlap_start is not None:
            if t > overlap_start:
                overlaps.append((overlap_start, t))
            overlap_start = None
    return overlaps


def speech_intervals(turns: list[Turn]) -> list[tuple[float, float]]:
    """Union of all turns (≥ 1 speaker active). Returns merged intervals."""
    if not turns:
        return []
    sorted_by_start = sorted(turns, key=lambda t: t.start)
    merged: list[list[float]] = [[sorted_by_start[0].start, sorted_by_start[0].end]]
    for t in sorted_by_start[1:]:
        if t.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t.end)
        else:
            merged.append([t.start, t.end])
    return [(s, e) for s, e in merged]


def silence_gaps(
    speech: list[tuple[float, float]], total_dur: float, min_gap_s: float
) -> list[tuple[float, float]]:
    """Silence gaps (no speaker active) at least ``min_gap_s`` long."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in speech:
        if s - cursor >= min_gap_s:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if total_dur - cursor >= min_gap_s:
        gaps.append((cursor, total_dur))
    return gaps


# ---------------------------------------------------------------------------
# Fragment scoring
# ---------------------------------------------------------------------------


def _measure(
    turns: list[Turn],
    overlaps: list[tuple[float, float]],
    speech: list[tuple[float, float]],
    start: float, end: float,
) -> tuple[float, int, float, dict[str, float], float]:
    """Compute (overlap_s, n_events, max_event_s, speech_s_by_spk,
    union_speech_s) restricted to [start, end].

    `max_event_s` is the duration of the longest single overlap event that
    intersects the fragment (clipped to the fragment bounds). It's the
    discriminator between "backchannel only" fragments (many short events,
    max < 1s) and "substantive overlap" fragments (at least one ≥ 1s event
    where both speakers genuinely talk over each other).
    """
    overlap_s = 0.0
    n_events = 0
    max_event_s = 0.0
    for os_, oe in overlaps:
        clip = max(0.0, min(end, oe) - max(start, os_))
        if clip > 0:
            overlap_s += clip
            n_events += 1
            if clip > max_event_s:
                max_event_s = clip
    by_spk: dict[str, float] = {}
    for t in turns:
        clip = max(0.0, min(end, t.end) - max(start, t.start))
        if clip > 0:
            by_spk[t.speaker] = by_spk.get(t.speaker, 0.0) + clip
    union_speech_s = 0.0
    for ss, se in speech:
        clip = max(0.0, min(end, se) - max(start, ss))
        union_speech_s += clip
    return overlap_s, n_events, max_event_s, by_spk, union_speech_s


def _snap_boundary_outward(
    boundary: float,
    silences: list[tuple[float, float]],
    direction: str,           # "left" or "right"
    search_s: float,
) -> float:
    """Move ``boundary`` outward (away from the fragment center) to the
    nearest silence midpoint within ``search_s``. Returns ``boundary``
    unchanged if no silence in range."""
    best = boundary
    best_dist = float("inf")
    for s, e in silences:
        mid = (s + e) / 2
        if direction == "left":
            if mid <= boundary and boundary - mid <= search_s:
                dist = boundary - mid
                if dist < best_dist:
                    best_dist = dist
                    best = mid
        else:  # right
            if mid >= boundary and mid - boundary <= search_s:
                dist = mid - boundary
                if dist < best_dist:
                    best_dist = dist
                    best = mid
    return best


def _iou(a: Fragment, b: Fragment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _nms(fragments: list[Fragment], iou_threshold: float) -> list[Fragment]:
    """Greedy non-max suppression — sort by overlap_s desc, drop any
    candidate whose IoU with an already-kept fragment exceeds the
    threshold."""
    fragments = sorted(fragments, key=lambda f: f.overlap_s, reverse=True)
    kept: list[Fragment] = []
    for f in fragments:
        if all(_iou(f, k) <= iou_threshold for k in kept):
            kept.append(f)
    kept.sort(key=lambda f: f.start)
    return kept


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_fragments(
    turns: list[Turn],
    total_dur: float,
    params: FragmentParams,
) -> list[Fragment]:
    """Identify candidate test fragments in one recording.

    Returns a list of ``Fragment`` sorted by start time. Empty list if the
    recording has no overlap or no candidate passes all filters.
    """
    overlaps = overlap_intervals(turns)
    if not overlaps:
        return []

    speech = speech_intervals(turns)
    silences = silence_gaps(speech, total_dur, params.silence_snap_min_s)

    candidates: list[Fragment] = []
    t = 0.0
    while t + params.target_length_s <= total_dur:
        raw_start = t
        raw_end = t + params.target_length_s

        snap_start = _snap_boundary_outward(
            raw_start, silences, "left", params.silence_snap_search_s
        )
        snap_end = _snap_boundary_outward(
            raw_end, silences, "right", params.silence_snap_search_s
        )
        snap_start = max(0.0, snap_start)
        snap_end = min(total_dur, snap_end)

        dur = snap_end - snap_start
        if dur < params.min_length_s or dur > params.max_length_s:
            t += params.stride_s
            continue

        overlap_s, n_events, max_event_s, by_spk, union_speech_s = _measure(
            turns, overlaps, speech, snap_start, snap_end
        )
        per_speaker_total = sum(by_spk.values())
        if per_speaker_total == 0:
            t += params.stride_s
            continue

        balance = min(by_spk.values()) / per_speaker_total
        speech_density = union_speech_s / dur

        if (overlap_s < params.min_overlap_s
                or balance < params.min_speaker_balance
                or speech_density < params.min_speech_density):
            t += params.stride_s
            continue

        candidates.append(Fragment(
            start=snap_start, end=snap_end,
            overlap_s=overlap_s, n_overlap_events=n_events,
            max_event_s=max_event_s,
            union_speech_s=union_speech_s,
            speech_s_by_spk=by_spk,
        ))
        t += params.stride_s

    return _nms(candidates, params.nms_iou_threshold)


# ---------------------------------------------------------------------------
# CLI smoke (single recording)
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("diarization", type=Path,
                   help="Path to a pyannote diarization JSON file.")
    p.add_argument("--target-length-s", type=float, default=90.0)
    p.add_argument("--min-overlap-s",   type=float, default=3.0)
    args = p.parse_args()

    turns, total = load_diarization(args.diarization)
    params = FragmentParams(
        target_length_s=args.target_length_s,
        min_overlap_s=args.min_overlap_s,
    )
    frags = find_fragments(turns, total, params)

    print(f"{args.diarization.name}: {total:.1f}s, {len(turns)} turns, "
          f"{len(frags)} fragment(s) found")
    for i, f in enumerate(frags):
        print(f"  [{i}] {f.start:7.1f} → {f.end:7.1f} "
              f"({f.duration:5.1f}s)  "
              f"overlap={f.overlap_s:5.1f}s ({f.overlap_density*100:.1f}%)  "
              f"max_event={f.max_event_s:4.1f}s  "
              f"balance={f.speaker_balance:.2f}  "
              f"speech={f.speech_density*100:.0f}%")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
