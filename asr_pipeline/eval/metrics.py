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
    - `cp_cer_meeteval` / `mimo_cer_meeteval` — character-error-rate analogs
      of cpWER and the MIMO mixture WER (same assignment, chars not words).

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
(`yyy`, `eee`, `mmm`, `hmm`, `mhm`, `yhy`). Lexical backchannels that *are*
words — `no`, `tak`, `aha` — are deliberately kept.
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
# clear hesitation sounds — lexical backchannels (`no`, `tak`, `aha`) are kept.
_FILLER_RE = re.compile(r"(?:y{2,}|e{2,}|m{2,}|hm+|mhm+|yhy)")

# Interchangeable spelling variants mapped to one canonical token, applied to
# BOTH reference and hypothesis so the choice never costs WER. Extend this as
# more equivalences turn up (keep only genuinely free variants — same word,
# different spelling — not different words).
_CANON = {"okej": "ok"}

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
    words in `lang`, split alphanumeric tokens, strip punctuation, collapse
    whitespace.

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
            out.append(tok)
    return " ".join(out)


def _seglst_from_dict(
    utts_by_spk: Dict[str, List[Utterance]], session_id: str, lang: str = "pl"
):
    """SegLST rows from per-speaker utterances, with normalization applied.

    Shared by every metric below — one row per non-empty utterance.
    `lang` selects the number speller (see ``_normalize_text``). meeteval is
    imported lazily so the module stays importable without it.
    """
    from meeteval.io.seglst import SegLST

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
            "words": _normalize_text(u.text, lang),
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

    `skip_tcp` exists for untimed references (e.g. EdAcc, which ships no
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
    """ORC-WER: best assignment of reference utterances to a single hypothesis.

    Use this for the *single-stream* baseline — running Whisper on the raw
    mixture as one transcript ("mixture mode"). ORC-WER selects the optimal
    permutation of reference utterances against that one hypothesis stream
    so the score isn't penalised by speaker-label arbitrariness.

    Inputs:
      - ``ref_utts_by_spk``: same shape as for ``cpwer_meeteval`` — the
        per-speaker GT.
      - ``hyp_utterances``: flat list of ``Utterance`` from the mixture
        transcript (parsed via ``parse_gt_txt``).

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


def mimo_cer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utterances: List[Utterance],
    session_id: str,
    lang: str = "pl",
) -> Dict[str, object]:
    """Character error rate for the single-stream mixture under MIMO's merge.

    The mixture-baseline analog of :func:`cp_cer_meeteval`. We run MIMO-WER
    to get the optimal interleaving of the reference speaker streams into the
    single hypothesis (the merge that minimises *word* errors, preserving
    each speaker's internal order), reorder the reference into that merge,
    then take char-level edit distance against the mixture hypothesis.

    Contrast with the older time-ordered mixture CER (reference merged by
    timestamp ≈ ORC order): that one is penalised when the two speakers
    interleave in an order Whisper doesn't follow inside overlaps; this one
    isn't — it matches the MIMO-WER floor's forgiveness. Report both.

    Same caveat as :func:`cp_cer_meeteval`: the merge order is *word*-optimal,
    not char-optimal (we reuse the WER metric's assignment rather than
    re-optimising at the character level). Returns ``{"cer", "errors",
    "length"}``.
    """
    from collections import deque

    from rapidfuzz.distance import Levenshtein
    from meeteval.wer import mimower

    ref = _seglst_from_dict(ref_utts_by_spk, session_id, lang)
    hyp = _seglst_from_list(hyp_utterances, session_id, lang=lang)
    m = mimower(ref, hyp)[session_id]

    # Rebuild the reference in MIMO's merge order. `m.assignment` lists one
    # (ref_spk, hyp_spk) per reference utterance, in merge order; MIMO keeps
    # each speaker's internal order, so we pop each speaker's utterances (same
    # raw-non-empty filter as the SegLST above) as the assignment calls them.
    queues = {
        spk: deque(u for u in utts if u.text.strip())
        for spk, utts in ref_utts_by_spk.items()
    }
    ordered: List[Utterance] = []
    for ref_spk, _hyp_spk in m.assignment:
        q = queues.get(ref_spk)
        if q:
            ordered.append(q.popleft())
    # Defensive: append anything the assignment didn't cover (shouldn't happen).
    for q in queues.values():
        ordered.extend(q)

    # Normalize the *joined* text once on each side (identical to the
    # time-ordered mixture CER), so the only difference is the merge order.
    ref_all = _normalize_text(" ".join(u.text for u in ordered), lang)
    hyp_all = _normalize_text(" ".join(u.text for u in hyp_utterances), lang)
    err = Levenshtein.distance(ref_all, hyp_all)
    length = max(len(ref_all), 1)
    return {"cer": float(err / length), "errors": int(err), "length": int(length)}
