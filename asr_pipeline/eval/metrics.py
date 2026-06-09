"""Metrics for the three-layer evaluation.

- **Layer 1 (DER)** — `compute_der`, thin wrapper around
  `pyannote.metrics.DiarizationErrorRate`. The metric does its own
  optimal speaker assignment via the Hungarian solver; we don't
  pre-permute.
- **Layer 2 (separation)** — no helper here; the notebook uses
  `torchmetrics.functional.audio.*` directly for SI-SDR / PESQ-WB /
  STOI (intrusive) and `torchaudio.pipelines.SQUIM_OBJECTIVE` for the
  non-intrusive estimates. Both libraries are already used elsewhere
  in the project (e.g. evaluate.py), so we stay
  consistent with them rather than rolling our own.
- **Layer 3 (cpWER / tcpWER)** — `cpwer_meeteval`, wraps the MeetEval
  package's cpWER + tcpWER. Permutation, alignment, and time-constrained
  scoring all live inside MeetEval — the CHiME-7/8 evaluation toolkit.

We strip punctuation and lowercase before scoring (preserving Polish
diacritics), since both Whisper and the pipeline emit casing /
punctuation that doesn't reflect actual ASR errors. MeetEval's built-in
normalizers either over-strip (`lower,rm([^a-z0-9 ])` removes Polish
letters) or under-strip (`lower,rm(.?!,)` misses `;:—…`), so we
pre-normalize the SegLST `words` field ourselves.

We also fold digit tokens to their spoken Polish words (`2024` →
`dwa tysiące dwadzieścia cztery`) so the GT (written as words, the way
they're spoken) and Whisper (which sometimes emits digits) land in the
same surface form instead of scoring as substitutions. The conversion is
cardinal-only — an ordinal written `5.` reads as `pięć`, not `piąty`, so
the rare cardinal/ordinal mismatch survives; everything else collapses.

Finally, following the CHiME normalizer, we drop non-verbal material that
neither side should be scored on: bracketed non-speech markup (`[śmiech]`,
`<muzyka>`) and a conservative list of non-lexical filler vocalizations
(`yyy`, `eee`, `mmm`, `hmm`, `mhm`, `yhy`). Lexical backchannels that *are*
words — `no`, `tak`, `aha` — are deliberately kept.

Note: an equivalent pure-numpy DER implementation exists in git history
(commit before this one) — restore it if pyannote.metrics ever misbehaves
inside a long-running Jupyter kernel. Numbers agree to within ~0.005 pp
on real data, so the switch is mechanical.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from asr_pipeline.eval.transcript_parser import Utterance

try:
    from num2words import num2words as _num2words
except ImportError:  # eval-only dep; scorer still runs, digits just stay digits
    _num2words = None


# ---------------------------------------------------------------------------
# Layer 1 — DER (diarization error rate)
# ---------------------------------------------------------------------------


def compute_der(
    ref_segments: Dict[str, List[Tuple[float, float]]],
    hyp_segments: Dict[str, List[Tuple[float, float]]],
    total_duration_s: float,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> Dict[str, float]:
    """DER + miss / false-alarm / confusion breakdown.

    `ref_segments`, `hyp_segments` map each speaker label to a list of
    (start, end) tuples (seconds). DER is reported as a fraction of the
    reference speech duration; multiply by 100 for a percent.

    `collar` (seconds) optionally forgives boundary mismatches within
    ±collar/2 of each reference segment edge.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def _to_annotation(per_spk):
        ann = Annotation()
        for spk, segs in per_spk.items():
            for s, e in segs:
                if e > s:
                    ann[Segment(s, e)] = spk
        return ann

    ref = _to_annotation(ref_segments)
    hyp = _to_annotation(hyp_segments)
    uem = Segment(0.0, total_duration_s)

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detailed = metric.compute_components(ref, hyp, uem=uem)
    total = max(detailed.get("total", 1.0), 1e-9)
    miss = detailed.get("missed detection", 0.0)
    fa = detailed.get("false alarm", 0.0)
    conf = detailed.get("confusion", 0.0)
    return {
        "der": (miss + fa + conf) / total,
        "miss": miss / total,
        "false_alarm": fa / total,
        "confusion": conf / total,
        "total_ref_s": float(total),
        "collar": collar,
        "skip_overlap": skip_overlap,
    }


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


@lru_cache(maxsize=4096)
def _digits_to_words_pl(token: str) -> str:
    """`'2024'` → `'dwa tysiące dwadzieścia cztery'` (cardinal, Polish).

    Returns the token unchanged when num2words is unavailable or the
    integer is out of its range. Cached because the same small integers
    recur across thousands of utterances.
    """
    if _num2words is None:
        return token
    try:
        return _num2words(int(token), lang="pl")
    except (ValueError, OverflowError, NotImplementedError):
        return token


def _normalize_text(s: str) -> str:
    """Lowercase, drop non-speech markup + fillers, fold digits to Polish
    words, strip punctuation, collapse whitespace.

    Preserves Polish diacritics (phonemic — `ł` vs `l` is a real
    substitution and should count as a WER error). Digit tokens become
    their spoken cardinal form so they match GT written as words;
    bracketed non-speech and non-lexical fillers are removed from both
    sides so they never count as errors.
    """
    s = _BRACKET_RE.sub(" ", s.lower())
    tokens = _PUNCT_RE.sub(" ", s).split()
    out = []
    for tok in tokens:
        if _FILLER_RE.fullmatch(tok):
            continue
        tok = _CANON.get(tok, tok)
        out.append(_digits_to_words_pl(tok) if tok.isdigit() else tok)
    return " ".join(out)


def _seglst_from_dict(utts_by_spk: Dict[str, List[Utterance]], session_id: str):
    """SegLST rows from per-speaker utterances, with normalization applied.

    Shared by every metric below — one row per non-empty utterance.
    meeteval is imported lazily so the module stays importable without it.
    """
    from meeteval.io.seglst import SegLST

    return SegLST([
        {
            "session_id": session_id,
            "speaker": spk,
            "start_time": float(u.start),
            "end_time": float(u.end),
            "words": _normalize_text(u.text),
        }
        for spk, utts in utts_by_spk.items()
        for u in utts
        if u.text.strip()
    ])


def _seglst_from_list(
    utterances: List[Utterance], session_id: str, speaker: str = "mixture"
):
    """SegLST rows from a flat utterance list under one pseudo-speaker."""
    return _seglst_from_dict({speaker: utterances}, session_id)


def cpwer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    tcp_collar_s: float = 5.0,
) -> Dict[str, object]:
    """cpWER + tcpWER via MeetEval, with Polish-aware text normalization.

    Inputs are dicts mapping speaker label → list of
    `Utterance(start, end, text)` (as produced by
    `parse_transcript_file` and `parse_gt_txt`).

    Returns a dict with::

        {
            "cpwer": float,                  # 0..1 fraction
            "cp_assignment": tuple,          # (ref_spk, hyp_spk) pairs
            "tcpwer": float,
            "tcp_assignment": tuple,
            "cp_errors": int, "cp_length": int,
            "tcp_errors": int, "tcp_length": int,
        }

    `tcp_collar_s` is the per-word time tolerance for tcpWER (CHiME-7
    default is 5.0). MeetEval places each word at its segment midpoint
    by default — fine for our use since GT segments come from Whisper's
    own segmentation.
    """
    from meeteval.wer import cpwer, tcpwer

    ref = _seglst_from_dict(ref_utts_by_spk, session_id)
    hyp = _seglst_from_dict(hyp_utts_by_spk, session_id)

    cp = cpwer(ref, hyp)[session_id]
    tcp = tcpwer(ref, hyp, collar=tcp_collar_s)[session_id]

    return {
        "cpwer": float(cp.error_rate),
        "cp_assignment": tuple(cp.assignment),
        "cp_errors": int(cp.errors),
        "cp_length": int(cp.length),
        "tcpwer": float(tcp.error_rate),
        "tcp_assignment": tuple(tcp.assignment),
        "tcp_errors": int(tcp.errors),
        "tcp_length": int(tcp.length),
        "tcp_collar_s": float(tcp_collar_s),
    }


def orc_wer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utterances: List[Utterance],
    session_id: str,
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
    # meeteval renamed cpwer/tcpwer style but kept the WER variants joined.
    from meeteval.wer import orcwer

    ref = _seglst_from_dict(ref_utts_by_spk, session_id)
    # One pseudo-speaker for the mixture hypothesis.
    hyp = _seglst_from_list(hyp_utterances, session_id)
    orc = orcwer(ref, hyp)[session_id]
    return {
        "orc_wer": float(orc.error_rate),
        "errors": int(orc.errors),
        "length": int(orc.length),
    }


def mimo_wer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utterances: List[Utterance],
    session_id: str,
) -> Dict[str, object]:
    """MIMO-WER for the single-stream mixture baseline.

    Same inputs as :func:`orc_wer_meeteval`, but uses MeetEval's MIMO-WER
    instead of ORC-WER. The difference matters for a single hypothesis
    stream (the un-separated mixture transcript):

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

    ref = _seglst_from_dict(ref_utts_by_spk, session_id)
    hyp = _seglst_from_list(hyp_utterances, session_id)
    m = mimower(ref, hyp)[session_id]
    return {
        "mimo_wer": float(m.error_rate),
        "errors": int(m.errors),
        "length": int(m.length),
    }


def orc_wer_multistream(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
) -> Dict[str, object]:
    """ORC-WER on a *multi-stream* hypothesis — attribution-blind WER.

    Optimally assigns each reference utterance to one of the hypothesis
    streams (the per-speaker pipeline outputs), ignoring reference speaker
    grouping. ORC-WER <= cpWER always; the gap ``cpWER - ORC-WER`` is the
    speaker-attribution penalty — how much error comes from routing words
    to the wrong speaker rather than mis-recognising them.

    Same shape as ``cpwer_meeteval`` inputs; same Polish-aware normalization.
    """
    from meeteval.wer import orcwer

    orc = orcwer(
        _seglst_from_dict(ref_utts_by_spk, session_id),
        _seglst_from_dict(hyp_utts_by_spk, session_id),
    )[session_id]
    return {
        "orc_wer": float(orc.error_rate),
        "errors": int(orc.errors),
        "length": int(orc.length),
    }


def cp_cer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
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

    cp = cpwer_meeteval(ref_utts_by_spk, hyp_utts_by_spk, session_id)

    def _concat_norm(utts_by_spk):
        return {
            spk: _normalize_text(" ".join(u.text for u in utts))
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

    ref = _seglst_from_dict(ref_utts_by_spk, session_id)
    hyp = _seglst_from_list(hyp_utterances, session_id)
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
    ref_all = _normalize_text(" ".join(u.text for u in ordered))
    hyp_all = _normalize_text(" ".join(u.text for u in hyp_utterances))
    err = Levenshtein.distance(ref_all, hyp_all)
    length = max(len(ref_all), 1)
    return {"cer": float(err / length), "errors": int(err), "length": int(length)}
