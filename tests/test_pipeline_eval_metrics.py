"""Tests for the Layer-1/Layer-3 scoring logic in asr_pipeline/eval/metrics.py.

Covers the Polish-aware text normalization (the part most likely to silently
change WER numbers), the cpWER/ORC/MIMO wrappers, the CER variants, and DER.
Heavy deps (meeteval, pyannote.metrics, rapidfuzz, num2words) are lazy inside
metrics.py, so each test group guards with importorskip — on this machine
they're all installed and everything runs.
"""

import pytest

from asr_pipeline.eval.metrics import (
    _digits_to_words_pl,
    _normalize_text,
    compute_der,
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
    assert _digits_to_words_pl("2") == "dwa"
    assert _normalize_text("mam 2 koty") == "mam dwa koty"


def test_digits_helper_leaves_non_integers():
    assert _digits_to_words_pl("abc") == "abc"


def test_digits_long_run_left_unchanged():
    # A single unbroken 70-digit token (Whisper hallucinating on silence/music)
    # passes str.isdigit() into num2words, whose Polish magnitude table raises
    # KeyError — which must be swallowed, returning the token unchanged, not
    # propagate out and crash every Layer-3 scorer for the recording.
    assert _digits_to_words_pl("1" * 70) == "1" * 70


def test_digits_leading_zeros_collapse_via_int():
    pytest.importorskip("num2words")
    assert _digits_to_words_pl("007") == "siedem"


def test_normalize_splits_hyphenated():
    # Hyphen is punctuation (not \w), so it becomes a token boundary.
    assert _normalize_text("biało-czerwony") == "biało czerwony"


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
# DER
# ---------------------------------------------------------------------------


def test_der_perfect_is_zero():
    pytest.importorskip("pyannote.metrics")
    ref = {"A": [(0.0, 5.0)], "B": [(5.0, 10.0)]}
    out = compute_der(ref, ref, total_duration_s=10.0)
    assert out["der"] == pytest.approx(0.0, abs=1e-9)


def test_der_miss():
    pytest.importorskip("pyannote.metrics")
    ref = {"A": [(0.0, 10.0)]}
    hyp = {"A": [(0.0, 5.0)]}               # second half missed
    out = compute_der(ref, hyp, total_duration_s=10.0)
    assert out["miss"] == pytest.approx(0.5, abs=1e-6)
    assert out["false_alarm"] == pytest.approx(0.0, abs=1e-9)
    assert out["der"] == pytest.approx(0.5, abs=1e-6)
    assert out["total_ref_s"] == pytest.approx(10.0)


def test_der_false_alarm():
    pytest.importorskip("pyannote.metrics")
    ref = {"A": [(0.0, 5.0)]}
    hyp = {"A": [(0.0, 10.0)]}              # 5 s of phantom speech
    out = compute_der(ref, hyp, total_duration_s=10.0)
    # Normalised by reference speech (5 s) -> FA fraction 1.0.
    assert out["false_alarm"] == pytest.approx(1.0, abs=1e-6)
    assert out["miss"] == pytest.approx(0.0, abs=1e-9)


def test_der_confusion():
    pytest.importorskip("pyannote.metrics")
    ref = {"A": [(0.0, 10.0)]}
    hyp = {"X": [(0.0, 5.0)], "Y": [(5.0, 10.0)]}
    out = compute_der(ref, hyp, total_duration_s=10.0)
    # Optimal mapping pairs one hyp speaker with A; the other 5 s is confusion.
    assert out["confusion"] == pytest.approx(0.5, abs=1e-6)
    assert out["der"] == pytest.approx(0.5, abs=1e-6)


def test_der_denominator_sums_per_speaker_speech_over_overlap():
    # The DER denominator is the sum of per-speaker reference speech, NOT the
    # union: 10 s of A overlapping 5 s of B totals 15 s of reference, so an
    # overlap region counts once for each speaker present. Pins the meaning of
    # the reported total_ref_s (and hence every DER fraction in the thesis).
    pytest.importorskip("pyannote.metrics")
    ref = {"A": [(0.0, 10.0)], "B": [(5.0, 10.0)]}
    out = compute_der(ref, ref, total_duration_s=10.0)
    assert out["total_ref_s"] == pytest.approx(15.0)
    assert out["der"] == pytest.approx(0.0, abs=1e-9)
