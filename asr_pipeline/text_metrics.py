"""Text-level transcript metrics shared across the pipeline and its tooling.

Deliberately stdlib-only (no torch / whisperx / numpy) so the offline
repetition-loop scanner (``docs/sweep_plan/scan_repetition_loops.py``) can import
it from the CLI without dragging in the heavy ASR stack — and so the in-pipeline
loop-retry detector (``stages/transcription.py``) and that post-run monitor score
transcripts with ONE implementation. They must agree: a window the pipeline
splices back in after a loop retry must not be re-flagged by the scanner.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

# A transcript segment needs at least this many word tokens before its
# dominant-token fraction is trusted as a loop signal — below it, ordinary short
# repeats ("no tak, no tak") would score misleadingly high.
LOOP_MIN_TOKENS = 10
# ...and the single most-common token must itself recur at least this many times.
# 12 sits in the observed gap between genuine disfluency and hallucination: every
# real loop in the corpus repeats ×14-×112 (REPETITION_LOOP_HALLUCINATIONS.md;
# tightest is `takie`×14), while the longest GENUINE repeat WhisperX transcribes
# is ×8-9 ("Tak, tak, ..." — dev 4f9251fe, which an earlier ×8 gate clipped into
# real deletions).
LOOP_MIN_TOP_COUNT = 12
# Default dominant-token-fraction threshold above which a segment is treated as a
# repetition-loop hallucination. This is the scanner's validated operating point
# (its former ``MIN_TOP_FRAC``); ``TranscriptionConfig.loop_score_threshold``
# defaults to it.
LOOP_SCORE_THRESHOLD = 0.4

_WORD_RE = re.compile(r"\w+")


class LoopScore(NamedTuple):
    """Repetition-loop metric for one transcript segment.

    ``score`` is the dominant-token fraction — occurrences of the single most
    common word token divided by the total word-token count — forced to ``0.0``
    when the segment is too short (< ``LOOP_MIN_TOKENS``) or the top token too
    infrequent (< ``LOOP_MIN_TOP_COUNT``) to be a loop. Those guards are folded
    into ``score`` so a bare ``score >= threshold`` test is a complete detector.
    ``top_token`` / ``top_count`` / ``n_tokens`` are exposed for the scanner's
    report and stay populated even when ``score`` is gated to ``0.0``.
    """

    score: float
    top_token: str
    top_count: int
    n_tokens: int


def repetition_loop_score(text: str) -> LoopScore:
    """Score how much a transcript segment looks like a repetition loop.

    A repetition-loop hallucination (``No tak, tak, tak, ...`` ×112) is one
    merged window dominated by a single repeated token; it scores near ``1.0``.
    Natural speech — even mild Polish disfluency repeats — scores low (or is
    gated to ``0.0`` by the token-count guards). See ``LoopScore``.
    """
    toks = _WORD_RE.findall((text or "").lower())
    n = len(toks)
    if n == 0:
        return LoopScore(0.0, "", 0, 0)
    top_token, top_count = Counter(toks).most_common(1)[0]
    if n < LOOP_MIN_TOKENS or top_count < LOOP_MIN_TOP_COUNT:
        return LoopScore(0.0, top_token, top_count, n)
    return LoopScore(top_count / n, top_token, top_count, n)
