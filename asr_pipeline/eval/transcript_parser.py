"""Parsers for the two transcript file formats in this project.

- `parse_transcript_file` — the pipeline's per-recording output:

    === Speaker A (SPEAKER_00) ===
    [ 21.46 →  26.22]  Nie wiem czy widok nie jest w słuchawkach.

    === Speaker B (SPEAKER_01) ===
    [ 28.46 →  29.96]  Dziękuje za uwagę.

- `parse_gt_txt` — the GT correction format produced by
  `scripts/transcribe_clarin_debleed.py` (one file per oracle channel,
  one segment per line):

    [00:00:01.20 → 00:00:03.45] To jest pierwszy segment.
    [00:00:03.50 → 00:00:05.80] To jest drugi segment.

Both return lists of `Utterance(start, end, text)` named tuples with
times in seconds.
"""

from __future__ import annotations

import re
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

# GT format: `[HH:MM:SS.cc → HH:MM:SS.cc] text`.
_GT_LINE_RE = re.compile(
    r"^\s*\[\s*(\d+):(\d+):(\d+\.\d+)\s*(?:→|->|—)\s*(\d+):(\d+):(\d+\.\d+)\s*\]\s*(.*)$"
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

    The file format has one segment per line with `[HH:MM:SS.cc →
    HH:MM:SS.cc]` prefixes; non-matching lines (blank, comments) are
    skipped. After human correction the timestamps are still preserved,
    so we keep them for tcpWER's per-word time alignment.
    """
    path = Path(path)
    utts: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _GT_LINE_RE.match(raw)
        if not m:
            continue
        h1, mi1, s1, h2, mi2, s2, text = m.groups()
        start = int(h1) * 3600 + int(mi1) * 60 + float(s1)
        end = int(h2) * 3600 + int(mi2) * 60 + float(s2)
        text = text.strip()
        if text:
            utts.append(Utterance(start, end, text))
    return utts


def concat_utterances(utts: Iterable[Utterance]) -> str:
    """Join the text of all utterances with single spaces."""
    return " ".join(u.text for u in utts if u.text)
