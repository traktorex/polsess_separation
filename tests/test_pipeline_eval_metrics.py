"""Tests for the Layer-3 scoring logic in asr_pipeline/eval/metrics.py.

Covers the Polish-aware text normalization (the part most likely to silently
change WER numbers), the cpWER/ORC/MIMO wrappers, and the CER variants.
Heavy deps (meeteval, rapidfuzz, num2words) are lazy inside metrics.py, so each
test group guards with importorskip — on this machine they're all installed and
everything runs.
"""

import pytest

from asr_pipeline.eval.metrics import (
    _digits_to_words,
    _normalize_text,
    cp_cer_meeteval,
    cpwer_meeteval,
    mimo_cer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
    orc_wer_multistream,
)
from asr_pipeline.eval.transcript_parser import Utterance as U


# ---------------------------------------------------------------------------
# _normalize_text — the normalization both sides pass through before scoring
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips_punct_keeps_diacritics():
    assert _normalize_text("Świetnie, Łódź!") == "świetnie łódź"


def test_normalize_strips_bracket_markup():
    assert _normalize_text("tak [śmiech] dobrze <muzyka> już") == "tak dobrze już"


def test_normalize_drops_fillers_keeps_lexical_backchannels():
    # yyy/eee/hmm/mhm/yhy are non-lexical fillers; `no`, `tak`, `aha` are words.
    assert _normalize_text("yyy tak eee mhm hmm yhy no aha") == "tak no aha"


def test_normalize_canonical_variants():
    assert _normalize_text("Okej, dobrze") == "ok dobrze"


def test_normalize_digits_to_polish_words():
    pytest.importorskip("num2words")
    assert _digits_to_words("2") == "dwa"
    assert _normalize_text("mam 2 koty") == "mam dwa koty"


def test_digits_helper_leaves_non_integers():
    assert _digits_to_words("abc") == "abc"


def test_digits_long_run_left_unchanged():
    # A single unbroken 70-digit token (Whisper hallucinating on silence/music)
    # passes str.isdigit() into num2words, whose Polish magnitude table raises
    # KeyError — which must be swallowed, returning the token unchanged, not
    # propagate out and crash every Layer-3 scorer for the recording.
    assert _digits_to_words("1" * 70) == "1" * 70


def test_digits_leading_zeros_collapse_via_int():
    pytest.importorskip("num2words")
    assert _digits_to_words("007") == "siedem"


# E13 — language-aware number spelling
# ---------------------------------------------------------------------------


def test_normalize_default_lang_is_polish_byte_identical():
    # Pin: the default (no lang arg) must reproduce the pre-E13 Polish path
    # byte-for-byte — the thesis L2/L3 numbers depend on it.
    pytest.importorskip("num2words")
    assert _normalize_text("mam 2024 koty") == "mam dwa tysiące dwadzieścia cztery koty"
    assert _digits_to_words("3") == "trzy"
    assert _digits_to_words("3", "pl") == "trzy"


def test_normalize_english_spells_digits_in_english():
    # E13 regression: scoring an English hypothesis must spell digits in
    # English, not Polish — `3` → `three`, never `trzy`.
    pytest.importorskip("num2words")
    assert _digits_to_words("3", "en") == "three"
    assert _normalize_text("i have 3 cats", "en") == "i have three cats"
    assert "trzy" not in _normalize_text("3", "en")


def test_normalize_splits_hyphenated():
    # Hyphen is punctuation (not \w), so it becomes a token boundary.
    assert _normalize_text("biało-czerwony") == "biało czerwony"


def test_normalize_alphanumeric_split_en_matches_spelled_out():
    # Mixed letter/digit tokens split into letter/digit runs with the digit
    # runs spelled out, so a compact recognizer form matches a written-out
    # reference. EdAcc's participant codes (`C3P2`) vs ref `C THREE P TWO`.
    pytest.importorskip("num2words")
    assert _normalize_text("C3P2", "en") == "c three p two"
    assert _normalize_text("C3P2", "en") == _normalize_text("C THREE P TWO", "en")
    assert _normalize_text("A24", "en") == "a twenty-four"
    assert _normalize_text("10x", "en") == "ten x"


def test_normalize_alphanumeric_split_pl_uses_polish_words():
    # The rule is language-uniform — Polish digit words for a Polish recording.
    pytest.importorskip("num2words")
    assert _normalize_text("A24", "pl") == "a dwadzieścia cztery"
    assert _normalize_text("10x", "pl") == "dziesięć x"


def test_normalize_alphanumeric_does_not_touch_pure_alpha_or_digit():
    # Pure-alpha tokens stay verbatim; pure-digit tokens still go through the
    # existing digit speller (not the alnum split) — byte-identical to before.
    pytest.importorskip("num2words")
    assert _normalize_text("hello world", "en") == "hello world"
    assert _normalize_text("mam 2024 koty") == "mam dwa tysiące dwadzieścia cztery koty"


def test_normalize_alphanumeric_is_symmetric_so_match_is_free():
    # Applied identically to both sides → a token glued one way and spelled the
    # other still matches (cpWER 0), the whole point of the rule.
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "my code is C3P2")]}
    hyp = {"A": [U(0.0, 1.0, "my code is C three P two")]}
    r = cpwer_meeteval(ref, hyp, session_id="s", lang="en")
    assert r["cpwer"] == pytest.approx(0.0)


def test_normalize_empty_input():
    assert _normalize_text("") == ""


def test_filler_regex_only_matches_whole_non_lexical_tokens():
    # `mm`/`hm` are dropped as fillers; `mama`/`my`/`em` are real tokens kept
    # (fullmatch, not substring — `mama` is not a run of `m`, `em` is one `e`).
    assert _normalize_text("mm hm mama my em") == "mama my em"


def test_normalize_collapses_whitespace():
    assert _normalize_text("  ala   ma\tkota  ") == "ala ma kota"


# ---------------------------------------------------------------------------
# cpWER
# ---------------------------------------------------------------------------


def _two_speaker_ref():
    return {
        "A": [U(0.0, 1.0, "ala ma kota")],
        "B": [U(1.0, 2.0, "kot ma alę")],
    }


def test_cpwer_perfect_match_is_zero():
    pytest.importorskip("meeteval")
    ref = _two_speaker_ref()
    out = cpwer_meeteval(ref, ref, session_id="t")
    assert out["cpwer"] == 0.0
    assert out["cp_errors"] == 0
    assert out["cp_length"] == 6


def test_cpwer_invariant_to_speaker_permutation():
    pytest.importorskip("meeteval")
    ref = _two_speaker_ref()
    hyp = {"A": ref["B"], "B": ref["A"]}      # labels swapped
    out = cpwer_meeteval(ref, hyp, session_id="t")
    assert out["cpwer"] == 0.0                # cp assignment absorbs the swap


def test_cpwer_counts_substitution():
    pytest.importorskip("meeteval")
    ref = _two_speaker_ref()
    hyp = {
        "A": [U(0.0, 1.0, "ala ma psa")],     # kota -> psa
        "B": ref["B"],
    }
    out = cpwer_meeteval(ref, hyp, session_id="t")
    assert out["cp_errors"] == 1
    assert out["cpwer"] == pytest.approx(1 / 6)


def test_cpwer_normalization_applied_to_both_sides():
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")]}
    hyp = {"A": [U(0.0, 1.0, "Ala, ma kota!!! [śmiech]")]}
    out = cpwer_meeteval(ref, hyp, session_id="t")
    assert out["cpwer"] == 0.0


def test_cpwer_skip_tcp_returns_none_tcpwer_not_zero():
    # Untimed-reference path: skip_tcp=True must compute cpWER normally but
    # return tcpwer=None (never a fabricated number) with tcp_skipped=True.
    pytest.importorskip("meeteval")
    ref = _two_speaker_ref()
    out = cpwer_meeteval(ref, ref, session_id="t", skip_tcp=True)
    assert out["cpwer"] == 0.0           # cpWER still computed (time-agnostic)
    assert out["tcpwer"] is None         # NOT 0.0 — never computed
    assert out["tcp_skipped"] is True
    assert out["tcp_assignment"] is None and out["tcp_errors"] is None


def test_cpwer_default_computes_tcp():
    # Default path (timed ref) still computes tcpWER — skip is opt-in only.
    pytest.importorskip("meeteval")
    ref = _two_speaker_ref()
    out = cpwer_meeteval(ref, ref, session_id="t")
    assert out["tcpwer"] == pytest.approx(0.0)
    assert out["tcp_skipped"] is False
    assert out["tcp_errors"] == 0


def test_cpwer_all_filler_session_scores_zero_not_crash():
    # An exchange of only non-lexical fillers ("yyy" / "mhm") normalizes to
    # empty on both sides, so meeteval reports length 0 and error_rate None.
    # That must score as a perfect match (0.0), not raise float(None) and
    # abort the whole evaluation batch on one backchannel-only recording.
    pytest.importorskip("meeteval")
    out = cpwer_meeteval(
        {"A": [U(0.0, 1.0, "yyy")]}, {"A": [U(0.0, 1.0, "mhm")]}, session_id="t"
    )
    assert out["cpwer"] == 0.0
    assert out["tcpwer"] == 0.0
    assert out["cp_errors"] == 0
    assert out["cp_length"] == 0


# ---------------------------------------------------------------------------
# ORC vs MIMO on the single-stream mixture baseline
# ---------------------------------------------------------------------------


def test_orc_and_mimo_zero_on_perfect_single_speaker():
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")]}
    hyp = [U(0.0, 1.0, "ala ma kota")]
    assert orc_wer_meeteval(ref, hyp, session_id="t")["orc_wer"] == 0.0
    assert mimo_wer_meeteval(ref, hyp, session_id="t")["mimo_wer"] == 0.0


def test_mimo_forgives_interleaving_orc_does_not():
    """The defining difference: MIMO optimises the interleaving of the
    per-speaker reference streams into the single hypothesis; ORC merges
    the reference in time order. When Whisper emits the speakers in an
    order that disagrees with the timeline, MIMO stays at 0 and ORC pays.
    """
    pytest.importorskip("meeteval")
    ref = {
        "A": [U(0.0, 1.0, "jeden"), U(2.0, 3.0, "dwa")],
        "B": [U(1.0, 2.0, "trzy")],
    }
    # Time-ordered ref merge = "jeden trzy dwa"; hypothesis says
    # "jeden dwa trzy" (speaker A finished before B was transcribed).
    hyp = [U(0.0, 3.0, "jeden dwa trzy")]
    orc = orc_wer_meeteval(ref, hyp, session_id="t")["orc_wer"]
    mimo = mimo_wer_meeteval(ref, hyp, session_id="t")["mimo_wer"]
    assert mimo == 0.0
    assert orc > mimo                       # MIMO <= ORC, strictly here


def test_mimo_wer_dict_hypothesis_is_speaker_agnostic():
    """`mimo_wer_meeteval` accepts a per-speaker *dict* hypothesis (the
    pipeline's two-stream output), not only the single-stream mixture list.
    That branch was untested. MIMO charges no attribution, so swapping the two
    hypothesis streams' content must still score 0 — every reference word is
    recoverable by re-interleaving regardless of which hyp stream it sits in.
    """
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "jeden dwa")], "B": [U(1.0, 2.0, "trzy cztery")]}
    # Hypothesis streams carry the right words but attribution-swapped (A<->B).
    hyp = {"A": [U(1.0, 2.0, "trzy cztery")], "B": [U(0.0, 1.0, "jeden dwa")]}
    assert mimo_wer_meeteval(ref, hyp, session_id="t")["mimo_wer"] == 0.0


def test_orc_multistream_below_cpwer_is_the_attribution_penalty():
    """ORC-WER on the multi-stream hypothesis is attribution-blind, so it
    assigns each reference utterance to whichever output stream fits best.
    When the words are all present but routed to the wrong speaker streams,
    cpWER pays the attribution penalty while multi-stream ORC does not — the
    gap cpWER - ORC is exactly that penalty (ORC <= cpWER always).
    """
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "jeden"), U(2.0, 3.0, "dwa")], "B": [U(1.0, 2.0, "trzy")]}
    # "dwa" and "trzy" are swapped between the two hypothesis streams.
    hyp = {"A": [U(0.0, 1.0, "jeden"), U(2.0, 3.0, "trzy")], "B": [U(1.0, 2.0, "dwa")]}
    orc = orc_wer_multistream(ref, hyp, session_id="t")["orc_wer"]
    cp = cpwer_meeteval(ref, hyp, session_id="t")["cpwer"]
    assert orc == 0.0                       # every word recoverable by re-routing
    assert cp > orc                         # cpWER charges the misattribution


# ---------------------------------------------------------------------------
# CER variants
# ---------------------------------------------------------------------------


def test_cp_cer_perfect_is_zero():
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = _two_speaker_ref()
    out = cp_cer_meeteval(ref, ref, session_id="t")
    assert out["cer"] == 0.0


def test_cp_cer_counts_char_edits():
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "kot")]}
    hyp = {"A": [U(0.0, 1.0, "kos")]}       # one char substitution
    out = cp_cer_meeteval(ref, hyp, session_id="t")
    assert out["errors"] == 1
    assert out["length"] == 3
    assert out["cer"] == pytest.approx(1 / 3)


def test_mimo_cer_perfect_is_zero():
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")]}
    hyp = [U(0.0, 1.0, "ala ma kota")]
    assert mimo_cer_meeteval(ref, hyp, session_id="t")["cer"] == 0.0


def test_mimo_cer_multi_speaker_interleave_merge_is_zero():
    """The multi-speaker branch of mimo_cer: with >1 reference speaker, MIMO
    decides the merge order and the helper rebuilds the reference by popping
    each speaker's utterances in assignment order. Only single-speaker refs were
    tested before. Two speakers whose words concatenate (in the speech timeline)
    to the single hypothesis must score 0 — the merge reconstructs the hyp
    exactly. (Contrast the time-ordered mixture CER, which can't reorder.)
    """
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "jeden"), U(2.0, 3.0, "dwa")], "B": [U(3.0, 4.0, "trzy")]}
    hyp = [U(0.0, 4.0, "jeden dwa trzy")]
    assert mimo_cer_meeteval(ref, hyp, session_id="t")["cer"] == 0.0


def test_mimo_cer_multi_speaker_counts_char_edit():
    """The complement: a genuine character edit in the multi-speaker merge is
    not forgiven — the rebuilt reference differs from the hypothesis by exactly
    that edit. ('trzy' vs 'trxy' is one substitution out of the joined length.)
    """
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "jeden"), U(2.0, 3.0, "dwa")], "B": [U(3.0, 4.0, "trzy")]}
    hyp = [U(0.0, 4.0, "jeden dwa trxy")]       # one char wrong in "trzy"
    out = mimo_cer_meeteval(ref, hyp, session_id="t")
    assert out["errors"] == 1
    assert out["cer"] > 0.0


def test_cp_cer_unmatched_reference_speaker_counts_as_deletions():
    # Two reference speakers, one hypothesis speaker: cpWER pairs B with the
    # empty hyp stream (assignment (B, None)), so all of B's characters score
    # as deletions. Pins the None-branch of cp_cer's hyp_txt.get(spk, "").
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "kot")], "B": [U(1.0, 2.0, "pies")]}
    hyp = {"A": [U(0.0, 1.0, "kot")]}
    out = cp_cer_meeteval(ref, hyp, session_id="t")
    assert out["errors"] == 4               # "pies" fully deleted
    assert out["length"] == 7               # "kot" (3) + "pies" (4)
    assert out["cer"] == pytest.approx(4 / 7)


# ---------------------------------------------------------------------------
# Empty hypothesis — a real pipeline outcome (WhisperX hears nothing in a
# sparse stream; observed live on LibriCSS OV40_session8_seg2 pipeline_nosep).
# meeteval would abort the whole batch ("Missing recordings in hypothesis");
# the _ensure_nonempty_hyp guard scores it as 100 % deletions instead.
# ---------------------------------------------------------------------------


def test_cpwer_empty_hypothesis_scores_all_deletions():
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")], "B": [U(1.0, 2.0, "pies je")]}
    hyp = {"A": [], "B": []}
    out = cpwer_meeteval(ref, hyp, session_id="t")
    assert out["cpwer"] == 1.0
    assert out["cp_length"] == 5            # all 5 ref words deleted
    assert out["cp_errors"] == 5
    assert out["tcpwer"] == 1.0             # tcp leg survives the guard too


def test_orc_and_mimo_empty_mixture_hypothesis_scores_all_deletions():
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")]}
    assert orc_wer_meeteval(ref, [], session_id="t")["orc_wer"] == 1.0
    assert mimo_wer_meeteval(ref, [], session_id="t")["mimo_wer"] == 1.0


def test_orc_multistream_empty_hypothesis_scores_all_deletions():
    pytest.importorskip("meeteval")
    ref = {"A": [U(0.0, 1.0, "ala ma kota")]}
    hyp = {"A": [], "B": []}
    assert orc_wer_multistream(ref, hyp, session_id="t")["orc_wer"] == 1.0


def test_cp_cer_empty_hypothesis_scores_all_deletions():
    pytest.importorskip("meeteval")
    pytest.importorskip("rapidfuzz")
    ref = {"A": [U(0.0, 1.0, "kot")]}
    out = cp_cer_meeteval(ref, {"A": []}, session_id="t")
    assert out["cer"] == 1.0
