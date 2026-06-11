"""EdAcc-specific hypothesis filtering — Stella-passage excision.

**Why this exists.** EdAcc (Edinburgh International Accents of English Corpus)
opens every conversation with both speakers reading the Speech Accent Archive
elicitation passage ("Please call Stella…") aloud. That reading is *not*
conversational speech and the official EdAcc evaluation excludes it — via
sclite time-gating against the per-utterance STM timestamps. EdAcc ships **no
per-utterance timestamps** to us (the HF mirror has none; the DataShare
`segments/` are uniform 30 s chunks; the official STM isn't distributed), so
NIST-style time-gating is impossible here. Author ruling 2026-06-12: excise the
passage from the *hypothesis* by **text matching** instead, mirroring how the
reference already drops it (the `IGNORE_TIME_SEGMENT_IN_SCORING` utterances are
removed when the EdAcc reference is prepared).

**Scope (hard).** This is dataset-specific, opt-in, and lives in the eval layer
ONLY. It is never wired into `asr_pipeline/` proper, never default-on, and
never touches any non-EdAcc dataset. The EdAcc scoring path passes the filter
in explicitly (`compute_layer3(..., hyp_filter=excise_stella_passage)`); every
other dataset leaves `hyp_filter=None` and is byte-identical to before.

**What it does.** Slides a window over the normalized hypothesis token stream,
finds the contiguous block whose token-level edit distance to the canonical
passage is lowest, and excises it when that block matches closely enough. It
iterates so both speakers' readings can be removed from a single stream (the
un-separated mixture transcript contains the passage twice; diarization error
can also land both readings in one separated stream). A conversational *mention*
of the passage — "I wonder who came up with this, it doesn't make any sense" —
is far from the 69-token canonical block and stays well above threshold, so it
survives (regression-tested against the real EAEC-C02 streams).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rapidfuzz.distance import Levenshtein

from asr_pipeline.debug_log import dlog
from asr_pipeline.eval.metrics import _normalize_text
from asr_pipeline.eval.transcript_parser import Utterance


# The Speech Accent Archive elicitation passage, canonical wording. Tuned and
# validated against the real EAEC-C02 WhisperX output (recognized variants seen:
# "slaps" for "slabs", missing/extra commas) — fuzzy matching absorbs those.
_STELLA_PASSAGE = (
    "Please call Stella. Ask her to bring these things with her from the store: "
    "six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a "
    "snack for her brother Bob. We also need a small plastic snake and a big toy "
    "frog for the kids. She can scoop these things into three red bags, and we "
    "will go meet her Wednesday at the train station."
)

# Normalized canonical token list, computed once. `_normalize_text` is the same
# normalizer L3 scoring uses, so the window comparison is apples-to-apples with
# how the hypothesis is scored. en: digits stay digits here (none in passage).
_PASSAGE_TOKENS: List[str] = _normalize_text(_STELLA_PASSAGE, "en").split()
_PASSAGE_LEN = len(_PASSAGE_TOKENS)

# Match acceptance: a window is excised when its token-level WER vs the canonical
# passage is <= this. Tuned on the real EAEC-C02 streams (A/B/mixture): the true
# passage readings score ~0.05-0.20 WER (a few recognition slips), while the
# nearest conversational block ("I wonder who came up with this…") scores well
# above 0.8. 0.45 sits in the wide empty gap between them — generous enough for
# heavy-accent recognition error, far from any conversational false positive.
_DEFAULT_THRESHOLD = 0.45

# Window sizes swept around the canonical length, to absorb insertions/deletions
# in the recognized reading (WhisperX may merge "her brother Bob" / split a
# sentence). +/-40% of the passage length brackets every real case seen.
_MIN_WINDOW = max(1, int(_PASSAGE_LEN * 0.6))
_MAX_WINDOW = int(_PASSAGE_LEN * 1.4)

# Hard cap on excisions per stream — both speakers' readings can land in one
# stream (mixture has 2; a diarization error can put 2 in a separated stream),
# but never more. The cap stops a pathological loop from eating real speech.
_MAX_EXCISIONS = 2


@dataclass
class ExcisionSpan:
    """One excised window: token span (half-open) + its match score (WER)."""

    start_token: int
    end_token: int          # half-open
    score: float            # token-level WER vs the canonical passage
    n_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_tokens = self.end_token - self.start_token


@dataclass
class ExcisionReport:
    """What the filter removed from one stream.

    `n_excised` windows were cut (0..2); `spans` carries each window's token
    span and match score; `n_tokens_before`/`after` bracket the token count.
    Surfaced by the caller (dlog + return value) so an excision is never silent.
    """

    n_excised: int
    spans: List[ExcisionSpan]
    n_tokens_before: int
    n_tokens_after: int
    threshold: float


def _utt_tokens(utts: List[Utterance], lang: str) -> List[tuple[int, str]]:
    """Flatten utterances → [(utt_index, normalized_token), ...].

    Carries each token's source-utterance index so an excised token span maps
    back to whole utterances (we excise at utterance granularity — a window
    that overlaps an utterance removes that utterance entirely; partial-
    utterance excision would leave fragments that mis-score).
    """
    out: List[tuple[int, str]] = []
    for i, u in enumerate(utts):
        for tok in _normalize_text(u.text, lang).split():
            out.append((i, tok))
    return out


def _best_window(tokens: List[str]) -> Optional[tuple[int, int, float]]:
    """Lowest-WER contiguous window vs the canonical passage.

    Returns (start, end_exclusive, wer) over the token list, or None if the
    stream is too short to hold even the minimum window. Sweeps window sizes
    in `[_MIN_WINDOW, _MAX_WINDOW]`; WER = edit_distance / passage_len so scores
    are comparable across window sizes (normalizing by the *reference* passage
    length, the quantity an excision is trying to match).
    """
    n = len(tokens)
    if n < _MIN_WINDOW:
        return None
    best: Optional[tuple[int, int, float]] = None
    max_w = min(_MAX_WINDOW, n)
    for w in range(_MIN_WINDOW, max_w + 1):
        for start in range(0, n - w + 1):
            window = tokens[start:start + w]
            dist = Levenshtein.distance(window, _PASSAGE_TOKENS)
            wer = dist / _PASSAGE_LEN
            if best is None or wer < best[2]:
                best = (start, start + w, wer)
    return best


def excise_stella_passage(
    utts: List[Utterance],
    lang: str = "en",
    threshold: float = _DEFAULT_THRESHOLD,
    session_id: str = "",
) -> tuple[List[Utterance], ExcisionReport]:
    """Remove up to 2 Stella-passage readings from one hypothesis stream.

    Iterates: find the lowest-WER window vs the canonical passage; if it clears
    `threshold`, drop the utterances it spans and search the remainder; stop at
    `_MAX_EXCISIONS` or when the best window no longer clears threshold.

    Excision is at **utterance** granularity: any utterance with a token inside
    the matched window is removed whole. Returns (filtered_utterances, report).
    The report must be surfaced by the caller (it is, via dlog below + the
    returned value) — SCOPE §4.1/§4.3, no silent removal.

    `lang` selects the normalizer used for matching (EdAcc is English). For
    non-English callers the canonical passage wouldn't match anyway; this filter
    is EdAcc-only by construction.
    """
    remaining = list(utts)
    n_before = sum(len(_normalize_text(u.text, lang).split()) for u in remaining)
    spans: List[ExcisionSpan] = []

    for _ in range(_MAX_EXCISIONS):
        flat = _utt_tokens(remaining, lang)
        tokens = [t for _, t in flat]
        best = _best_window(tokens)
        if best is None:
            break
        start_tok, end_tok, wer = best
        if wer > threshold:
            break
        # Map the matched token span back to the utterances it touches.
        touched = sorted({flat[i][0] for i in range(start_tok, end_tok)})
        spans.append(ExcisionSpan(start_token=start_tok, end_token=end_tok, score=wer))
        dlog("edacc",
             f"excise_stella_passage[{session_id}]: window tokens "
             f"[{start_tok}:{end_tok}] (n={end_tok - start_tok}) wer={wer:.3f} "
             f"<= {threshold} → dropping utterances {touched}")
        keep = [u for i, u in enumerate(remaining) if i not in set(touched)]
        if len(keep) == len(remaining):
            # No utterance actually removed (shouldn't happen) — stop to avoid
            # an infinite loop on a window that maps to nothing.
            break
        remaining = keep

    n_after = sum(len(_normalize_text(u.text, lang).split()) for u in remaining)
    report = ExcisionReport(
        n_excised=len(spans),
        spans=spans,
        n_tokens_before=n_before,
        n_tokens_after=n_after,
        threshold=threshold,
    )
    if spans:
        dlog("edacc",
             f"excise_stella_passage[{session_id}]: removed {len(spans)} "
             f"passage window(s), {n_before} → {n_after} tokens")
    return remaining, report


# Type alias for the L3 wire-up hook (a per-speaker-stream hypothesis filter).
HypFilter = Callable[[List[Utterance]], tuple[List[Utterance], ExcisionReport]]
