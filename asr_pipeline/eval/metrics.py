"""Metrics for the eval layers (L2 audio quality, L3 ASR error rates).

L1/DER is retired (SCOPE §10 q8): no valid reference diarization exists for
any dataset, so DER is not computed anywhere.

- **Layer 2 (separation)** — no helper here; the notebook uses
  `torchmetrics.functional.audio.*` directly for SI-SDR / PESQ-WB /
  STOI (intrusive) and `torchaudio.pipelines.SQUIM_OBJECTIVE` for the
  non-intrusive estimates. Both libraries are already used elsewhere
  in the project (e.g. evaluate.py), so we stay
  consistent with them rather than rolling our own.
- **Layer 3 (ASR error rates)** — the cpWER / ORC / MIMO WER & CER family,
  all wrapping the MeetEval package (the CHiME-7/8 evaluation toolkit), which
  owns permutation, alignment, and time-constrained scoring:
    - `cpwer_meeteval` — cpWER + tcpWER, speaker-attributed (charges
      attribution errors); the main per-speaker score.
    - `orc_wer_meeteval` / `mimo_wer_meeteval` — speaker-agnostic WER of the
      single-stream mixture baseline (one Whisper pass over the raw mix);
      MIMO additionally forgives the reference-interleaving order.
    - `orc_wer_multistream` — attribution-blind WER of the multi-stream
      (per-speaker) hypothesis; `cpWER - ORC` is the attribution penalty.
    - `cp_cer_meeteval` / `orc_cer_meeteval` / `mimo_cer_meeteval` —
      character-error-rate analogs of cpWER / ORC-WER / MIMO-WER. Each reuses
      its WER metric's word-level assignment and scores chars under it (never
      re-optimising the routing at the character level — that would read more
      permissively); `orc_/mimo_cer_meeteval` take either the single-stream
      mixture or a multi-stream per-speaker hypothesis, like `mimo_wer_meeteval`.

We strip punctuation and lowercase before scoring (preserving Polish
diacritics), since both Whisper and the pipeline emit casing /
punctuation that doesn't reflect actual ASR errors. MeetEval's built-in
normalizers either over-strip (`lower,rm([^a-z0-9 ])` removes Polish
letters) or under-strip (`lower,rm(.?!,)` misses `;:—…`), so we
pre-normalize the SegLST `words` field ourselves.

We also fold digit tokens to their spoken words (`2024` →
`dwa tysiące dwadzieścia cztery`) so the GT (written as words, the way
they're spoken) and Whisper (which sometimes emits digits) land in the
same surface form instead of scoring as substitutions. The conversion is
cardinal-only — an ordinal written `5.` reads as `pięć`, not `piąty`, so
the rare cardinal/ordinal mismatch survives; everything else collapses.
The number speller is **language-aware** (E13): every scoring entry point
takes `lang` (default `"pl"`), threaded down to `_digits_to_words`, so an
English hypothesis spells `3` as `three`, not the Polish `trzy`. Layer 3
resolves `lang` from the pipeline's `metadata.json` config snapshot.
Mixed alphanumeric tokens (`C3P2`, `A24`) are split into letter/digit runs
with the digit runs spelled out (`c three p two`, `a twenty-four` for en /
`a dwadzieścia cztery` for pl), so the compact recognizer form matches a
spelled-out reference (`C THREE P TWO`). This is language-uniform and applied
symmetrically to both sides, so it can only fix a spurious mismatch, never
introduce one.

Finally, following the CHiME normalizer, we drop non-verbal material that
neither side should be scored on: bracketed non-speech markup (`[śmiech]`,
`<muzyka>`) and a conservative list of non-lexical filler vocalizations
(`yyy`, `eee`, `mmm`, `hmm`, `mhm`). Lexical backchannels that *are*
words — `no`, `tak`, `aha`, `yhy` — are deliberately kept.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

from asr_pipeline.debug_log import dlog
from asr_pipeline.eval.transcript_parser import Utterance

try:
    from num2words import num2words as _num2words
except ImportError:  # eval-only dep; scorer still runs, digits just stay digits
    _num2words = None


# ---------------------------------------------------------------------------
# Layer 3 — cpWER / tcpWER via MeetEval
# ---------------------------------------------------------------------------

# Strip ASCII + Unicode punctuation Whisper actually emits in Polish output.
# Keeps Polish letters (ą ć ę ł ń ó ś ź ż) because `\w` in Python's `re` is
# Unicode-aware by default and matches them.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)

# Bracketed non-speech markup: `[śmiech]`, `[muzyka]`, `<śmiech>`. Removed
# before punctuation stripping so the inner word doesn't leak as a token.
_BRACKET_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")

# Non-lexical filler vocalizations (whole-token match). Conservative: only
# clear hesitation sounds — lexical backchannels (`no`, `tak`, `aha`, `yhy`) are
# kept (the author's GT convention treats `yhy` as a lexical backchannel, unlike
# the `mhm`/`eee`/`yyy`/`mmm`/`hmm` hesitations which both sides drop).
_FILLER_RE = re.compile(r"(?:y{2,}|e{2,}|m{2,}|hm+|mhm+)")

# Interchangeable spelling variants mapped to one canonical token, applied to
# BOTH reference and hypothesis so the choice never costs WER. Extend this as
# more equivalences turn up (keep only genuinely free variants — same word,
# different spelling — not different words).
_CANON = {"okej": "ok", "noo": "no"}

# Collapse expressive letter elongation: a letter repeated 3+ times -> once
# ("nooo"->"no", "taaak"->"tak"). Safe because Polish orthography never repeats a
# letter 3+ times (max gemination is 2: "lekko", "kooperacja", "zoo"), so this can
# only fire on expressive lengthening. Applied to plain word tokens on BOTH sides;
# 2-char interjection elongations ("noo") go through _CANON instead, since a
# blanket 2->1 collapse would corrupt legitimate doubles.
_ELONG_RE = re.compile(r"(.)\1{2,}")

# Apostrophes are removed WITHOUT inserting a space (unlike other punctuation,
# which becomes a space) so a Polish genitive of a foreign proper noun stays one
# token: "War'a" -> "wara" (matching a GT written "Wara"), not "war a". Covers
# ASCII ' plus the curly/modifier variants Whisper/Cohere emit.
_APOS_RE = re.compile("['’‘ʼ]")

# Whole-phrase equivalences folded on the normalized string (both sides), for
# abbreviation <-> spelled-out forms the per-token pass can't see. Canonical form
# is the abbreviation. "i tak dalej" is overwhelmingly the "etc." idiom in
# conversational Polish, so the fold is safe.
_PHRASE_FOLDS = [
    (re.compile(r"\bi tak dalej\b"), "itd"),
    (re.compile(r"\bi tym podobne\b"), "itp"),
]

# Split a token on its maximal digit runs, keeping the runs as separate groups:
# "c3p2" → ["c", "3", "p", "2"]; "a24" → ["a", "24"]; "10x" → ["10", "x"].
# Used to render alphanumeric tokens letter/digit-run-wise so a digit-run hyp
# ("C3P2") matches a spelled-out reference ("C THREE P TWO") — see _alnum_split.
_DIGIT_RUN_RE = re.compile(r"(\d+)")


@lru_cache(maxsize=4096)
def _digits_to_words(token: str, lang: str = "pl") -> str:
    """`'2024'` → spoken cardinal words in `lang` (e.g. pl: `'dwa tysiące
    dwadzieścia cztery'`, en: `'two thousand and twenty-four'`).

    `lang` is a num2words language code; it selects the spoken form so a
    digit hypothesis lands in the same surface form as the words-spelled-out
    GT for *that* recording's language (E13 — scoring English hypotheses with
    the Polish number speller turned `3` into `trzy`, a fabricated error).

    Returns the token unchanged when num2words is unavailable, or when the
    integer is outside num2words' supported range — including the absurdly
    long digit runs Whisper sometimes hallucinates on silence/music, where
    num2words raises `KeyError`/`IndexError` (not `ValueError`) from its
    magnitude table. Cached because the same small integers recur across
    thousands of utterances.
    """
    if _num2words is None:
        return token
    try:
        return _num2words(int(token), lang=lang)
    except (ValueError, OverflowError, NotImplementedError, KeyError, IndexError):
        return token


def _alnum_split(token: str, lang: str) -> str:
    """Render a mixed letter/digit token as space-separated letter/digit runs,
    spelling each digit run in `lang`.

    `'c3p2'` → `'c three p two'` (en) / `'c trzy p dwa'` (pl), so a recognizer
    that emits the compact form (`C3P2`) lands in the same surface form as a
    reference written out (`C THREE P TWO`). Letter runs are kept verbatim;
    digit runs go through `_digits_to_words`. The rule is language-uniform
    (uses the per-language number speller).

    Pure-letter and pure-digit tokens never reach here (the caller dispatches
    pure digits to `_digits_to_words` directly and leaves pure letters alone),
    so this only fires on genuinely mixed tokens (`a24`, `10x`, `ck3`).
    """
    parts = [p for p in _DIGIT_RUN_RE.split(token) if p]
    return " ".join(
        _digits_to_words(p, lang) if p.isdigit() else p for p in parts
    )


def _is_alnum_mixed(token: str) -> bool:
    """True iff the token has at least one digit AND at least one non-digit
    character (the case `_alnum_split` handles). Pure-digit / pure-alpha → False."""
    has_digit = any(c.isdigit() for c in token)
    has_other = any(not c.isdigit() for c in token)
    return has_digit and has_other


def _normalize_text(s: str, lang: str = "pl") -> str:
    """Lowercase, drop non-speech markup + fillers, fold digits to spoken
    words in `lang`, split alphanumeric tokens, collapse letter elongation,
    drop apostrophes (no space), fold abbreviation phrases (itd/itp),
    strip punctuation, collapse whitespace.

    Preserves diacritics (phonemic in Polish — `ł` vs `l` is a real
    substitution and should count as a WER error). Digit tokens become their
    spoken cardinal form *in `lang`* so they match GT written as words; mixed
    alphanumeric tokens (`C3P2`, `A24`) are split into letter/digit runs with
    the digit runs spelled out, so the compact recognizer form matches a
    spelled-out reference. Bracketed non-speech and non-lexical fillers are
    removed from both sides so they never count as errors. `lang="pl"` (the
    default) reproduces the pre-E13 behaviour byte-for-byte *except* on mixed
    alphanumeric tokens, which previously stayed glued (e.g. `a24`) and now
    split (`a dwadzieścia cztery`) — applied symmetrically to ref and hyp.
    """
    s = _BRACKET_RE.sub(" ", s.lower())
    s = _APOS_RE.sub("", s)   # apostrophes drop with no space (see _APOS_RE)
    tokens = _PUNCT_RE.sub(" ", s).split()
    out = []
    for tok in tokens:
        if _FILLER_RE.fullmatch(tok):
            continue
        tok = _CANON.get(tok, tok)
        if tok.isdigit():
            out.append(_digits_to_words(tok, lang))
        elif _is_alnum_mixed(tok):
            out.append(_alnum_split(tok, lang))
        else:
            out.append(_ELONG_RE.sub(r"\1", tok))
    result = " ".join(out)
    for pat, repl in _PHRASE_FOLDS:   # abbreviation <-> spelled-out (see _PHRASE_FOLDS)
        result = pat.sub(repl, result)
    return result


# Word-boundary sentinel for character-level (CER) tokenization: spaces become
# this token so they are scored as edits, matching cp_cer's string-Levenshtein
# convention (spaces count). U+2581 does not occur in normalized transcript text.
_CER_SPACE = "▁"


def _char_words(text_norm: str) -> str:
    """Char-tokenize a normalized 'words' string for character-level (CER)
    scoring: each character becomes its own whitespace-separated token (spaces →
    the _CER_SPACE sentinel). A WER routine run on this yields CER, reusing the
    exact permutation/assignment machinery of the word-level metrics."""
    return " ".join(_CER_SPACE if ch == " " else ch for ch in text_norm)


def _seglst_from_dict(
    utts_by_spk: Dict[str, List[Utterance]], session_id: str, lang: str = "pl",
    char_level: bool = False,
):
    """SegLST rows from per-speaker utterances, with normalization applied.

    Shared by every metric below — one row per non-empty utterance.
    `lang` selects the number speller (see ``_normalize_text``). When
    ``char_level`` is set the normalized text is char-tokenized (``_char_words``)
    so a WER routine computes CER. meeteval is imported lazily so the module
    stays importable without it.
    """
    from meeteval.io.seglst import SegLST

    def _words(u):
        w = _normalize_text(u.text, lang)
        return _char_words(w) if char_level else w

    # Untimed utterances (start/end = None) carry 0.0 placeholders here. This
    # is safe ONLY because the metrics that consume this SegLST — cpWER, ORC,
    # MIMO — ignore time; the time-aware metric (tcpWER) must be gated upstream
    # in layer3 so it never scores an untimed reference on these placeholders
    # (SCOPE §4.1: no fake-time scoring presented as real).
    return SegLST([
        {
            "session_id": session_id,
            "speaker": spk,
            "start_time": float(u.start) if u.start is not None else 0.0,
            "end_time": float(u.end) if u.end is not None else 0.0,
            "words": _words(u),
        }
        for spk, utts in utts_by_spk.items()
        for u in utts
        if u.text.strip()
    ])


def _seglst_from_list(
    utterances: List[Utterance], session_id: str, speaker: str = "mixture",
    lang: str = "pl",
):
    """SegLST rows from a flat utterance list under one pseudo-speaker."""
    return _seglst_from_dict({speaker: utterances}, session_id, lang)


def _ensure_nonempty_hyp(hyp, session_id: str, context: str):
    """Make an all-empty hypothesis scoreable as 100 % deletions.

    meeteval aborts the whole batch ("Missing ... recordings in hypothesis")
    when a session has zero hypothesis segments — but an empty hypothesis is
    a real pipeline outcome (observed: LibriCSS OV40_session8_seg2,
    pipeline_nosep — WhisperX heard nothing in either sparse stream).
    meeteval's own remedy is to emit an empty transcript, so we insert one
    empty-text segment (→ WER 1.0, all deletions) and say so visibly
    (SCOPE §4.1: no silent substitution)."""
    if len(hyp) > 0:
        return hyp
    from meeteval.io.seglst import SegLST

    dlog("metrics",
         f"{context}: hypothesis for session {session_id!r} is empty — "
         "inserting one empty-text segment so it scores as all-deletions "
         "(WER 1.0) instead of aborting the batch")
    return SegLST([{
        "session_id": session_id, "speaker": "A",
        "start_time": 0.0, "end_time": 0.0, "words": "",
    }])


def _rate(obj) -> float:
    """A meeteval result's error rate as a float, scoring a 0/0 session as a
    perfect match.

    meeteval sets ``error_rate = None`` when the reference length is zero —
    i.e. every utterance normalized to empty on both sides, as in an
    all-filler ``yyy / mhm`` backchannel exchange. Scoring that as 0.0 rather
    than crashing on ``float(None)`` stops one such recording from aborting a
    whole evaluation batch.
    """
    return float(obj.error_rate) if obj.error_rate is not None else 0.0


def _wer_result(obj, key: str) -> Dict[str, object]:
    """``{key: rate, "errors": ..., "length": ...}`` — the shared return shape
    of the single-stream ORC/MIMO and multi-stream ORC wrappers."""
    return {key: _rate(obj), "errors": int(obj.errors), "length": int(obj.length)}


def cpwer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    tcp_collar_s: float = 5.0,
    lang: str = "pl",
    skip_tcp: bool = False,
) -> Dict[str, object]:
    """cpWER (+ tcpWER unless skipped) via MeetEval, language-aware normalization.

    Inputs are dicts mapping speaker label → list of
    `Utterance(start, end, text)` (as produced by
    `parse_transcript_file` and `parse_gt_txt`).

    Returns a dict with::

        {
            "cpwer": float,                  # 0..1 fraction
            "cp_assignment": tuple,          # (ref_spk, hyp_spk) pairs
            "tcpwer": float or None,         # None when skip_tcp
            "tcp_assignment": tuple or None,
            "cp_errors": int, "cp_length": int,
            "tcp_errors": int or None, "tcp_length": int or None,
            "tcp_skipped": bool,             # True iff tcpWER not computed
        }

    `tcp_collar_s` is the per-word time tolerance for tcpWER (CHiME-7
    default is 5.0). MeetEval places each word at its segment midpoint
    by default — fine for our use since GT segments come from Whisper's
    own segmentation.

    `skip_tcp` exists for untimed references (datasets that ship no
    per-utterance timing): tcpWER on fake/placeholder times would be a
    fabricated number (SCOPE §4.1), so the caller sets `skip_tcp=True` and
    tcpWER is returned as `None` with `tcp_skipped=True`. cpWER, which ignores
    time, is unaffected.
    """
    from meeteval.wer import cpwer

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    hyp = _ensure_nonempty_hyp(
        _seglst_from_dict(hyp_utts_by_spk, session_id, lang),
        session_id, "cpwer_meeteval",
    )

    cp = cpwer(ref, hyp)[session_id]
    out: Dict[str, object] = {
        "cpwer": _rate(cp),
        "cp_assignment": tuple(cp.assignment),
        "cp_errors": int(cp.errors),
        "cp_length": int(cp.length),
        "tcp_collar_s": float(tcp_collar_s),
        "tcp_skipped": bool(skip_tcp),
    }
    if skip_tcp:
        out.update(tcpwer=None, tcp_assignment=None,
                   tcp_errors=None, tcp_length=None)
        return out

    from meeteval.wer import tcpwer

    tcp = tcpwer(ref, hyp, collar=tcp_collar_s)[session_id]
    out.update(
        tcpwer=_rate(tcp),
        tcp_assignment=tuple(tcp.assignment),
        tcp_errors=int(tcp.errors),
        tcp_length=int(tcp.length),
    )
    return out


def orc_wer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utterances: List[Utterance],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """ORC-WER for the single-stream mixture baseline (one Whisper pass).

    This is the **one-hypothesis-stream** special case of MeetEval's ORC-WER.
    MeetEval's ORC-WER is *not* inherently single-stream — it routes each
    reference utterance to one of an arbitrary number of hypothesis streams
    (``hypothesis.groupby('speaker')`` internally); the multi-stream sibling
    that scores the per-speaker pipeline output is :func:`orc_wer_multistream`.
    Here the mixture transcript is passed as one pseudo-speaker, so there is
    exactly one stream and the routing is **forced** (every reference utterance
    lands on it). With no routing freedom left, ORC reduces to comparing the
    hypothesis against *all reference utterances concatenated in segment/time
    order* (``reference_sort='segment_if_available'``).

    Consequence: for this K=1 case ORC still **charges the reference
    interleaving order**. If the GT time-order is ``A1 B1 A2 B2`` but Whisper
    emits ``A1 A2 B1 B2``, ORC pays for the reordering. It is
    :func:`mimo_wer_meeteval` — not this function — that forgives that order
    (it keeps the per-speaker reference streams separate and optimises their
    interleaving). Report MIMO alongside ORC for the mixture baseline; the
    ``ORC - MIMO`` gap is exactly the cost of the fixed time-order merge.

    Inputs:
      - ``ref_utts_by_spk``: same shape as for ``cpwer_meeteval`` — the
        per-speaker GT.
      - ``hyp_utterances``: flat list of ``Utterance`` from the mixture
        transcript (parsed via ``parse_gt_txt``).

    Note: ORC/MIMO scores depend on GT *segmentation* granularity (the
    utterance is the atomic assignment unit), unlike plain cpWER/cpCER which
    concatenate each speaker before scoring. See :func:`mimo_wer_meeteval`.

    Returns ``{"orc_wer", "errors", "length"}``.
    """
    from meeteval.wer import orcwer

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    # One pseudo-speaker for the mixture hypothesis.
    hyp = _ensure_nonempty_hyp(
        _seglst_from_list(hyp_utterances, session_id, lang=lang),
        session_id, "orc_wer_meeteval",
    )
    orc = orcwer(ref, hyp)[session_id]
    return _wer_result(orc, "orc_wer")


def mimo_wer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp: "List[Utterance] | Dict[str, List[Utterance]]",
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """MIMO-WER, for either the single-stream mixture or a multi-stream hyp.

    ``hyp`` may be a flat ``List[Utterance]`` (the single-stream mixture
    baseline — one Whisper pass over the raw mix) or a ``Dict[str,
    List[Utterance]]`` of per-speaker streams (the pipeline's per-speaker
    output). MeetEval's MIMO-WER is multiple-input multiple-output by design,
    so both are valid hypothesis shapes; the dict form gives the
    speaker-agnostic MIMO-WER of the pipeline output (charges no attribution).

    Like :func:`orc_wer_meeteval` it uses MeetEval's MIMO-WER instead of
    ORC-WER. The difference matters for a single hypothesis stream (the
    un-separated mixture transcript):

    - **ORC** keeps the reference as one pool and fixes the merge order by
      utterance time, then assigns to the hypothesis stream.
    - **MIMO** keeps the per-speaker reference streams separate and *optimises
      their interleaving* into the single hypothesis (preserving each
      speaker's internal order). It therefore does not penalise the
      unpredictable order in which Whisper interleaves the two speakers
      inside overlaps.

    Both are speaker-agnostic (neither charges attribution errors; MeetEval
    paper Fig 1a). Crucially, MIMO is **more robust to faulty reference
    annotations** — the paper's Fig 1c shows ORC over-estimating the WER
    when the reference is imperfect, where MIMO does not. That makes MIMO
    the better mixture-baseline score for recordings whose GT timestamps are
    unreliable (e.g. the pre-ELAN 442dd69e GT). For clean single-output data
    the gap to ORC is usually small; report which one you used.

    MIMO-WER costs more than ORC-WER (it keeps the dependency on the number
    of reference speakers I that ORC drops), but is polynomial and trivial
    for our two-speaker, few-hundred-utterance transcripts.

    Returns ``{"mimo_wer", "errors", "length"}``.
    """
    from meeteval.wer import mimower

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    hyp_seglst = (
        _seglst_from_dict(hyp, session_id, lang)
        if isinstance(hyp, dict)
        else _seglst_from_list(hyp, session_id, lang=lang)
    )
    hyp_seglst = _ensure_nonempty_hyp(hyp_seglst, session_id, "mimo_wer_meeteval")
    m = mimower(ref, hyp_seglst)[session_id]
    return _wer_result(m, "mimo_wer")


def orc_wer_multistream(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """ORC-WER on a *multi-stream* hypothesis — attribution-blind WER.

    Optimally assigns each reference utterance to one of the hypothesis
    streams (the per-speaker pipeline outputs), ignoring reference speaker
    grouping. ORC-WER <= cpWER always; the gap ``cpWER - ORC-WER`` is the
    speaker-attribution penalty — how much error comes from routing words
    to the wrong speaker rather than mis-recognising them.

    Same shape as ``cpwer_meeteval`` inputs; same language-aware normalization.
    """
    from meeteval.wer import orcwer

    orc = orcwer(
        _seglst_from_dict(ref_utts_by_spk, session_id, lang),
        _ensure_nonempty_hyp(
            _seglst_from_dict(hyp_utts_by_spk, session_id, lang),
            session_id, "orc_wer_multistream",
        ),
    )[session_id]
    return _wer_result(orc, "orc_wer")


def orc_cer_multistream(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """ORC-CER on a multi-stream hypothesis — the character analog of
    :func:`orc_wer_multistream`. Char-tokenizes both sides (spaces scored, as in
    :func:`cp_cer_meeteval`) so meeteval's ORC routine returns CER. The gap
    ``cp-CER - ORC-CER`` is the attribution penalty in characters; ORC-CER <=
    cp-CER always. Returns ``{"orc_cer", "errors", "length"}``."""
    from meeteval.wer import orcwer

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang, char_level=True)
    hyp = _ensure_nonempty_hyp(
        _seglst_from_dict(hyp_utts_by_spk, session_id, lang, char_level=True),
        session_id, "orc_cer_multistream",
    )
    return _wer_result(orcwer(ref, hyp)[session_id], "orc_cer")


def mimo_cer_multistream(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """MIMO-CER on a multi-stream hypothesis — the character analog of
    :func:`mimo_wer_meeteval` on a per-speaker dict. The granularity-robust
    content floor in characters. Returns ``{"mimo_cer", "errors", "length"}``.

    SLOW: meeteval's MIMO assignment search blows up on char-token sequences
    (~150x the word-level cost, ~11 s on a 90 s fragment), so it is NOT used in
    routine scoring (rescore_stratified uses ORC-CER, which is ~0.7 s and
    coincides with MIMO-CER on this data). Kept for one-off granularity checks."""
    from meeteval.wer import mimower

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang, char_level=True)
    hyp = _ensure_nonempty_hyp(
        _seglst_from_dict(hyp_utts_by_spk, session_id, lang, char_level=True),
        session_id, "mimo_cer_multistream",
    )
    return _wer_result(mimower(ref, hyp)[session_id], "mimo_cer")


def cp_cer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """Character error rate under the cpWER speaker assignment.

    We score cpWER first to obtain the optimal (ref_spk → hyp_spk)
    permutation, then compute character-level edit distance on the
    concatenated, normalized text of each matched pair. Reusing cpWER's
    assignment (rather than letting CER pick its own permutation) keeps
    the two numbers directly comparable: same speaker matching, same
    Polish-aware normalization, only the unit (chars vs words) differs.

    Spaces between tokens count as characters, but since both sides pass
    through identical normalization (single spaces between tokens) that's
    symmetric and doesn't bias the rate. Returns ``{"cer", "errors",
    "length"}`` with ``length`` the total reference character count.
    """
    from rapidfuzz.distance import Levenshtein

    cp = cpwer_meeteval(ref_utts_by_spk, hyp_utts_by_spk, session_id, lang=lang)

    def _concat_norm(utts_by_spk):
        return {
            spk: _normalize_text(" ".join(u.text for u in utts), lang)
            for spk, utts in utts_by_spk.items()
        }

    ref_txt = _concat_norm(ref_utts_by_spk)
    hyp_txt = _concat_norm(hyp_utts_by_spk)

    total_err = 0
    total_len = 0
    for ref_spk, hyp_spk in cp["cp_assignment"]:
        r = ref_txt.get(ref_spk, "")
        h = hyp_txt.get(hyp_spk, "")
        total_err += Levenshtein.distance(r, h)
        total_len += len(r)

    return {
        "cer": float(total_err / max(total_len, 1)),
        "errors": int(total_err),
        "length": int(total_len),
    }


def _cer_under_routing(
    per_hyp: Dict[str, List[Utterance]],
    hyp_by_spk: Dict[str, List[Utterance]],
    lang: str,
    leftover: "List[Utterance] | tuple" = (),
) -> Dict[str, object]:
    """Character errors for a *fixed* reference→hypothesis routing.

    Shared back end for :func:`orc_cer_meeteval` and the multi-stream branch of
    :func:`mimo_cer_meeteval`. ``per_hyp`` maps each hypothesis stream to the
    reference utterances routed to it (already in the order the WER metric
    merged them); ``hyp_by_spk`` is the hypothesis text per stream. We
    concatenate + normalize each side *per stream* and sum char-level edit
    distances — mirroring :func:`cp_cer_meeteval` (reuse the WER assignment,
    then score characters). A hypothesis stream with no reference routed to it
    contributes its whole length as insertions; ``leftover`` reference
    utterances (defensive; normally empty) count as deletions.
    """
    from rapidfuzz.distance import Levenshtein

    hyp_txt = {
        spk: _normalize_text(" ".join(u.text for u in utts), lang)
        for spk, utts in hyp_by_spk.items()
    }
    errors = 0
    length = 0
    scored = set()
    for hyp_spk, utts in per_hyp.items():
        ref_txt = _normalize_text(" ".join(u.text for u in utts), lang)
        errors += Levenshtein.distance(ref_txt, hyp_txt.get(hyp_spk, ""))
        length += len(ref_txt)
        scored.add(hyp_spk)
    for spk, txt in hyp_txt.items():
        if spk not in scored:
            errors += len(txt)          # unmatched hypothesis stream = insertions
    for u in leftover:
        ref_txt = _normalize_text(u.text, lang)
        errors += len(ref_txt)
        length += len(ref_txt)          # unrouted reference = deletions
    return {"cer": float(errors / max(length, 1)),
            "errors": int(errors), "length": int(length)}


def orc_cer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp: "List[Utterance] | Dict[str, List[Utterance]]",
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """ORC-CER — character errors under **ORC-WER's** utterance routing.

    ``hyp`` may be a flat ``List[Utterance]`` (the single-stream mixture) or a
    ``Dict[str, List[Utterance]]`` of per-speaker streams (the pipeline output),
    like :func:`mimo_wer_meeteval`. We run *word-level* ORC-WER to get its
    routing (each reference utterance → the hypothesis stream that recognised it
    best), then score characters under **that fixed routing** — reusing the WER
    assignment exactly as :func:`cp_cer_meeteval` reuses cpWER's, *not*
    re-optimising ORC at the character level (that is :func:`orc_cer_multistream`,
    which grants extra routing freedom and so reads more permissively). Each
    stream's reference is concatenated in segment/time order, matching ORC's own
    merge; the single-stream case therefore reduces to the time-ordered mixture
    CER.

    ORC-CER is attribution-blind: ``cp-CER − ORC-CER`` is the attribution
    penalty in characters (non-negative up to the word-vs-char merge-order
    caveat shared with :func:`cp_cer_meeteval`). Returns ``{"cer", "errors",
    "length"}``.
    """
    from collections import defaultdict

    from meeteval.wer import orcwer

    hyp_by_spk = hyp if isinstance(hyp, dict) else {"mixture": hyp}
    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    hyp_seg = _ensure_nonempty_hyp(
        _seglst_from_dict(hyp_by_spk, session_id, lang), session_id, "orc_cer_meeteval",
    )
    orc = orcwer(ref, hyp_seg)[session_id]

    # `orc.assignment` is one hypothesis-stream label per reference utterance,
    # aligned to the reference SegLST row order — the same speaker-major,
    # non-empty order `_seglst_from_dict` and this comprehension both produce.
    flat = [u for _spk, utts in ref_utts_by_spk.items()
            for u in utts if u.text.strip()]
    per_hyp: Dict[str, List[Utterance]] = defaultdict(list)
    for u, hyp_spk in zip(flat, orc.assignment):
        per_hyp[hyp_spk].append(u)
    for utts in per_hyp.values():   # ORC concatenates each stream in time order
        utts.sort(key=lambda u: (u.start if u.start is not None else 0.0))
    return _cer_under_routing(per_hyp, hyp_by_spk, lang)


def mimo_cer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp: "List[Utterance] | Dict[str, List[Utterance]]",
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """MIMO-CER — character errors under **MIMO-WER's** interleaving.

    ``hyp`` may be a flat ``List[Utterance]`` (the single-stream mixture — the
    original use) or a ``Dict[str, List[Utterance]]`` of per-speaker streams
    (the pipeline output), mirroring :func:`mimo_wer_meeteval`. We run
    *word-level* MIMO-WER to get its optimal interleaving/routing (the merge
    that minimises *word* errors, preserving each speaker's internal order),
    then score characters under **that fixed merge**.

    We deliberately reuse the word-level assignment rather than re-running MIMO
    at the character level (:func:`mimo_cer_multistream`): char-level MIMO
    re-optimises the interleaving to minimise *character* errors, which grants
    extra reordering freedom (more permissive) and is ~150x slower. This is the
    same discipline as :func:`cp_cer_meeteval`. Consequence of that choice: the
    merge is word-optimal, not char-optimal, so MIMO-CER is **not** guaranteed
    ≤ ORC-CER at the character level — a small, honest artifact of reusing the
    word assignment, not a bug.

    Contrast the time-ordered mixture CER (:func:`orc_cer_meeteval` single
    stream): that merges the reference by timestamp and is penalised when the
    speakers interleave in an order the hypothesis doesn't follow; MIMO forgives
    it. Returns ``{"cer", "errors", "length"}``.
    """
    from collections import defaultdict, deque

    from meeteval.wer import mimower

    hyp_by_spk = hyp if isinstance(hyp, dict) else {"mixture": hyp}
    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    hyp_seg = _ensure_nonempty_hyp(
        _seglst_from_dict(hyp_by_spk, session_id, lang), session_id, "mimo_cer_meeteval",
    )
    m = mimower(ref, hyp_seg)[session_id]

    # Route each reference utterance to its hypothesis stream in MIMO's merge
    # order. `m.assignment` lists one (ref_spk, hyp_spk) per reference
    # utterance, in merge order; MIMO keeps each speaker's internal order, so we
    # pop that speaker's utterances (same raw-non-empty filter as the SegLST) as
    # the assignment calls them.
    queues = {
        spk: deque(u for u in utts if u.text.strip())
        for spk, utts in ref_utts_by_spk.items()
    }
    per_hyp: Dict[str, List[Utterance]] = defaultdict(list)
    for ref_spk, hyp_spk in m.assignment:
        q = queues.get(ref_spk)
        if q:
            per_hyp[hyp_spk].append(q.popleft())
    leftover = [u for q in queues.values() for u in q]  # defensive; normally empty
    return _cer_under_routing(per_hyp, hyp_by_spk, lang, leftover)
