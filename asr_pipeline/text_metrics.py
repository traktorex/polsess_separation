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


# ---------------------------------------------------------------------------
# Multi-token PHRASE-loop detection
# ---------------------------------------------------------------------------

# The multi-token mirror of the dominant-token metric above. A repeated 3-token
# phrase caps every single token's dominant fraction at ~1/3 < LOOP_SCORE_
# THRESHOLD, so `repetition_loop_score` is structurally blind to it; this scans
# for a repeated n-gram instead. Detection = the longest consecutive repeated
# n-gram run over the token stream, for n in [PHRASE_RUN_NGRAM_MIN,
# PHRASE_RUN_NGRAM_MAX], gated at PHRASE_RUN_MIN repeats.
#
# Calibration (the PHRASE_RUN_MIN gate): across all 236 hand-corrected GT
# reference texts the longest GENUINE consecutive repeated 3–8-gram run is 2,
# while the two observed hallucinations run 6 (152ed870__seg00 transcript_A,
# "Tak, to jest..." ×6 inside ONE segment) and 14 (5bab2c34__seg00
# transcript_mixture, "Jak pojedziemy do Dekathlonu... O!" ×14 across 14
# consecutive segments). Gating at run >= 4 sits in the wide two-sided margin —
# mirrors the single-token gate study (genuine ≤×9 vs real loops ×14+, gate 12).
PHRASE_RUN_NGRAM_MIN = 3
PHRASE_RUN_NGRAM_MAX = 8
PHRASE_RUN_MIN = 4


class PhraseRun(NamedTuple):
    """Longest consecutive repeated-phrase run in a token stream.

    ``run`` = number of consecutive exact repeats of ``phrase`` (an n-gram of
    PHRASE_RUN_NGRAM_MIN..MAX tokens carrying >= 2 distinct tokens); ``start`` /
    ``end`` are the token-index span it covers (end exclusive). A stream with no
    qualifying repeat yields ``run == 1`` with an empty ``phrase`` and a zero
    span; an empty token stream yields ``run == 0``.
    """

    run: int
    phrase: str
    start: int
    end: int


def find_phrase_runs(
    tokens: list[str], min_run: int = PHRASE_RUN_MIN
) -> list[PhraseRun]:
    """Every merged repeated-phrase run of at least ``min_run`` repeats.

    Scans every n-gram length in [PHRASE_RUN_NGRAM_MIN, PHRASE_RUN_NGRAM_MAX]
    and every start index (no skip-optimisation — transcripts are small and
    correctness beats speed). An n-gram is a candidate ONLY if it holds >= 2
    distinct tokens: a uniform run ("no no no no") is the single-token detector's
    jurisdiction (``repetition_loop_score``), and genuine one-word Polish
    backchannel repeats are common. The same loop is detected at its period AND
    at multiples of the period, so overlapping / adjacent token spans are merged;
    each merged span carries the max run and the phrase of its strongest
    (highest-run) contributing detection. Returned sorted by start.

    ``tokens`` are opaque strings — a caller can forbid a run from crossing a
    boundary by injecting a UNIQUE sentinel token there (e.g. ``"\\x00gap0"``,
    ``"\\x00gap1"``); uniqueness alone stops any n-gram from matching across it.
    """
    n_tokens = len(tokens)
    detections: list[PhraseRun] = []
    for n in range(PHRASE_RUN_NGRAM_MIN, PHRASE_RUN_NGRAM_MAX + 1):
        for i in range(0, n_tokens - n + 1):
            base = tokens[i:i + n]
            if len(set(base)) < 2:
                continue                        # uniform n-gram — not our job
            run = 1
            j = i + n
            while j + n <= n_tokens and tokens[j:j + n] == base:
                run += 1
                j += n
            if run >= min_run:
                detections.append(
                    PhraseRun(run, " ".join(base), i, i + run * n)
                )
    if not detections:
        return []
    # Merge overlapping/adjacent spans (one loop shows up at its period and every
    # multiple of it); keep the strongest detection's run + phrase per span.
    detections.sort(key=lambda d: (d.start, d.end))
    merged: list[PhraseRun] = []
    cur_start, cur_end, best = (
        detections[0].start, detections[0].end, detections[0]
    )
    for d in detections[1:]:
        if d.start <= cur_end:                  # overlap or touch → same loop
            cur_end = max(cur_end, d.end)
            if d.run > best.run:
                best = d
        else:
            merged.append(PhraseRun(best.run, best.phrase, cur_start, cur_end))
            cur_start, cur_end, best = d.start, d.end, d
    merged.append(PhraseRun(best.run, best.phrase, cur_start, cur_end))
    return merged


def max_phrase_run(tokens: list[str]) -> PhraseRun:
    """The single strongest repeated-phrase run, with NO min-run gate.

    For the offline scanner's report — like ``find_phrase_runs`` but ungated
    (min_run = 2) and reduced to the highest-run span. Returns the no-repeat
    sentinel ``PhraseRun(1, "", 0, 0)`` when nothing repeats, and
    ``PhraseRun(0, "", 0, 0)`` for an empty token stream.
    """
    if not tokens:
        return PhraseRun(0, "", 0, 0)
    runs = find_phrase_runs(tokens, min_run=2)
    if not runs:
        return PhraseRun(1, "", 0, 0)
    return max(runs, key=lambda r: r.run)


def phrase_run_score(text: str) -> PhraseRun:
    """Tokenise ``text`` (lowercased, ``_WORD_RE``) and return ``max_phrase_run``.

    Text-in convenience for the offline scanner and tests; mirrors
    ``repetition_loop_score``'s signature.
    """
    return max_phrase_run(_WORD_RE.findall((text or "").lower()))
