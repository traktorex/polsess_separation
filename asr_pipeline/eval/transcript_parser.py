"""Parsers for the two transcript file formats in this project.

- `parse_transcript_file` — the pipeline's per-recording output:

    === Speaker A (SPEAKER_00) ===
    [ 21.46 →  26.22]  Nie wiem czy widok nie jest w słuchawkach.

    === Speaker B (SPEAKER_01) ===
    [ 28.46 →  29.96]  Dziękuje za uwagę.

- `parse_gt_txt` — the GT correction format (one file per oracle
  channel, one segment per line):

    [00:00:01.20 → 00:00:03.45] To jest pierwszy segment.
    [00:00:03.50 → 00:00:05.80] To jest drugi segment.

Both return lists of `Utterance(start, end, text)` named tuples with
times in seconds.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, NamedTuple


class Utterance(NamedTuple):
    """One timestamped utterance."""

    start: float
    end: float
    text: str


# Pipeline output: `=== Speaker A (SPEAKER_00) ===` header + `[s.cc → s.cc] text` lines.
_HEADER_RE = re.compile(r"^===\s*Speaker\s+(\S+).*?===\s*$")
_PIPELINE_LINE_RE = re.compile(
    r"^\s*\[\s*(\d+\.\d+)\s*(?:→|->|—)\s*(\d+\.\d+)\s*\]\s*(.*)$"
)

# GT format — two flavours we accept:
#   `[HH:MM:SS.cc → HH:MM:SS.cc] text`   (older script-generated GT)
#   `[ s.cc → s.cc]            text`     (newer hand-corrected GT — copied
#                                         from the pipeline / explore_pipeline
#                                         stage 5 output, which prints
#                                         decimal seconds rather than HH:MM:SS).
_GT_LINE_HMS_RE = re.compile(
    r"^\s*\[\s*(\d+):(\d+):(\d+\.\d+)\s*(?:→|->|—)\s*(\d+):(\d+):(\d+\.\d+)\s*\]\s*(.*)$"
)
_GT_LINE_SECONDS_RE = re.compile(
    r"^\s*\[\s*(\d+(?:\.\d+)?)\s*(?:→|->|—)\s*(\d+(?:\.\d+)?)\s*\]\s*(.*)$"
)


def parse_transcript_file(path: str | Path) -> dict[str, list[Utterance]]:
    """Parse a pipeline transcript file → {speaker_label: [Utterance, ...]}.

    Unrecognised lines (blank, sub-headers, etc.) are skipped so minor
    formatting drift doesn't break the parser.
    """
    path = Path(path)
    current: str | None = None
    out: dict[str, list[Utterance]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _HEADER_RE.match(line)
            if m:
                current = m.group(1)
                out.setdefault(current, [])
                continue
            if current is None:
                continue
            m = _PIPELINE_LINE_RE.match(line)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                text = m.group(3).strip()
                if text:
                    out[current].append(Utterance(start, end, text))
    return out


def parse_gt_txt(path: str | Path) -> list[Utterance]:
    """Parse one GT .txt file (one channel) → list of utterances.

    Accepts either timestamp format (`[HH:MM:SS.cc → ...]` or `[s.cc → s.cc]`);
    blank / non-matching lines (sub-headers, comments) are skipped.
    """
    path = Path(path)
    utts: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_timed_line(raw)
        if parsed is None:
            continue
        start, end, text = parsed
        if text:
            utts.append(Utterance(start, end, text))
    return utts


def _parse_timed_line(raw: str) -> tuple[float, float, str] | None:
    """Try both `[HH:MM:SS.cc → ...]` and `[s.cc → s.cc]` forms.

    Returns (start_s, end_s, text) or None if the line doesn't match
    either form. Order matters: try HMS first because the seconds regex
    would otherwise match the seconds field of an HMS timestamp.
    """
    m = _GT_LINE_HMS_RE.match(raw)
    if m:
        h1, mi1, s1, h2, mi2, s2, text = m.groups()
        return (
            int(h1) * 3600 + int(mi1) * 60 + float(s1),
            int(h2) * 3600 + int(mi2) * 60 + float(s2),
            text.strip(),
        )
    m = _GT_LINE_SECONDS_RE.match(raw)
    if m:
        s1, s2, text = m.groups()
        return float(s1), float(s2), text.strip()
    return None


def parse_eaf(path: str | Path) -> dict[str, list[Utterance]]:
    """Parse an ELAN ``.eaf`` file → ``{speaker_label: [Utterance, ...]}``.

    This is the reader for the **hand-corrected GT** — the EAF is the source
    of truth, so the eval never needs a separate ``.txt`` export.

    Each ``TIER`` becomes one speaker. The ``Speaker_`` prefix is stripped
    from ``TIER_ID`` so ``Speaker_A`` → ``A`` (matching the pipeline's A/B
    labels). Only ``ALIGNABLE_ANNOTATION``s with non-empty text and
    resolvable start/end time slots are kept; utterances are returned sorted
    by start time. Tiers ELAN may add on save (empty tiers, symbolic
    subdivisions) yield empty lists, which downstream code drops.
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    slots: dict[str, float] = {}
    for ts in root.iter("TIME_SLOT"):
        v = ts.get("TIME_VALUE")
        if v is not None:
            slots[ts.get("TIME_SLOT_ID")] = int(v) / 1000.0

    out: dict[str, list[Utterance]] = {}
    for tier in root.iter("TIER"):
        tier_id = tier.get("TIER_ID") or ""
        label = tier_id[len("Speaker_"):] if tier_id.startswith("Speaker_") else tier_id
        utts: list[Utterance] = []
        for aa in tier.iter("ALIGNABLE_ANNOTATION"):
            start = slots.get(aa.get("TIME_SLOT_REF1"))
            end = slots.get(aa.get("TIME_SLOT_REF2"))
            val = aa.find("ANNOTATION_VALUE")
            text = (val.text or "").strip() if val is not None else ""
            if text and start is not None and end is not None:
                utts.append(Utterance(start, end, text))
        utts.sort(key=lambda u: u.start)
        out[label] = utts
    return out


def concat_utterances(utts: Iterable[Utterance]) -> str:
    """Join the text of all utterances with single spaces."""
    return " ".join(u.text for u in utts if u.text)
