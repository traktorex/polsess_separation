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
