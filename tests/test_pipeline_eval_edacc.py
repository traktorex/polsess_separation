"""Tests for EdAcc eval support — untimed GT + Stella-passage excision.

Three behaviours, all on the EdAcc input universe (untimed English GT,
elicitation-passage readings the official eval time-gated out and we can't):

- ``parse_gt_txt`` reads the untimed format (``# untimed`` header) without
  breaking the timestamped path; mixed-format lines are handled visibly.
- ``eval.edacc.excise_stella_passage`` removes the passage reading from a
  hypothesis stream via fuzzy block matching, up to 2× per stream, and leaves
  conversational text (incl. a *mention* of the passage) untouched. The
  positive and negative cases use the real recognized wording from the
  EAEC-C02 pipeline run.
- ``compute_layer3`` skips tcpWER for untimed references (visibly:
  ``ref_untimed`` + ``tcp_skipped``), still computes cpWER, and threads the
  ``hyp_filter`` to both per-speaker and mixture hypotheses.

CPU-only; meeteval/rapidfuzz are lazy-imported and present on this machine.
"""

import pytest

from asr_pipeline.eval.edacc import (
    _PASSAGE_TOKENS,
    excise_stella_passage,
)
from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.recordings import Recording
from asr_pipeline.eval.transcript_parser import (
    Utterance as U,
    format_untimed_gt,
    is_untimed,
    parse_gt_txt,
)


# Real recognized wording from ~/datasets/eval/clarin/EAEC-C02/pipeline — the
# passage as WhisperX actually transcribed it (note "slaps" for "slabs", the
# recognition slip the fuzzy matcher must absorb).
_STELLA_HYP_REAL = [
    "Please call Stella.",
    "Ask her to bring these things with her from the store.",
    "Six spoons of fresh snow peas, five thick slaps of blue cheese, and maybe "
    "a snack for her brother, Bob.",
    "We also need a small plastic snake and a big toy frog for the kids.",
    "She can scoop these things into three red bags and we will go meet her "
    "Wednesday at the train station.",
]
# Real conversational *mention* of the passage from the EAEC-C02 reference —
# the negative control. Must survive (it is not the passage, just talk about it).
_NEG_CONTROL = [
    "I wonder who came up with this.",
    "Doesn't make any sense.",
    "It is a very weird one.",
]
# Filler conversation to pad streams past the minimum window length.
_FILLER = [f"this is conversational sentence number {i} about edinburgh weather"
           for i in range(12)]


# ---------------------------------------------------------------------------
# Untimed GT format — parser
# ---------------------------------------------------------------------------


def test_parse_gt_txt_reads_untimed(tmp_path):
    p = tmp_path / "speaker_A.txt"
    p.write_text("# untimed\nfirst utterance here\nsecond utterance here\n",
                 encoding="utf-8")
    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["first utterance here", "second utterance here"]
    assert all(u.start is None and u.end is None for u in utts)
    assert is_untimed(utts)


def test_parse_gt_txt_untimed_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("# untimed\n# a comment\n\nreal line\n", encoding="utf-8")
    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["real line"]


def test_timestamped_path_unchanged_not_untimed(tmp_path):
    # The timestamped format must still parse with real times and not be
    # flagged untimed — the existing CLARIN/LibriCSS path is untouched.
    p = tmp_path / "a.txt"
    p.write_text("[  1.00 →   2.00]  pierwszy\n[  3.00 →   4.00]  drugi\n",
                 encoding="utf-8")
    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["pierwszy", "drugi"]
    assert utts[0].start == pytest.approx(1.0)
    assert not is_untimed(utts)


def test_untimed_mixed_format_line_kept_with_warning(tmp_path, capsys):
    # A bracketed (timed-looking) line inside an untimed file is a format mix;
    # SCOPE §4.1 — it is kept (text preserved), not silently dropped, and warned.
    p = tmp_path / "a.txt"
    p.write_text("# untimed\nplain line\n[  1.00 → 2.00] timed leak\n",
                 encoding="utf-8")
    utts = parse_gt_txt(p)
    # Both lines survive; the bracketed one keeps its raw text.
    assert "plain line" in [u.text for u in utts]
    assert any("timed leak" in u.text for u in utts)
    # dlog warns; capture from stdout (dlog mirrors to stdout in this project).
    out = capsys.readouterr().out
    assert "untimed file" in out or "format mix" in out


def test_format_untimed_gt_roundtrips(tmp_path):
    body = format_untimed_gt(["one", "  ", "two", ""])
    assert body.startswith("# untimed\n")
    p = tmp_path / "a.txt"
    p.write_text(body, encoding="utf-8")
    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["one", "two"]   # blanks dropped


# ---------------------------------------------------------------------------
# Stella-passage excision matcher
# ---------------------------------------------------------------------------


def test_excise_removes_real_passage_reading():
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, t) for t in (_FILLER[:3] + _STELLA_HYP_REAL + _FILLER[3:])]
    filtered, report = excise_stella_passage(utts, lang="en", session_id="t")
    assert report.n_excised == 1
    assert report.n_tokens_after < report.n_tokens_before
    # The passage utterances are gone; the filler conversation stays.
    remaining = " ".join(u.text for u in filtered).lower()
    assert "stella" not in remaining
    assert "conversational sentence" in remaining


def test_excise_match_score_is_tight():
    # The real reading matches the canonical passage closely (one "slaps" slip);
    # the window WER must be well under threshold.
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, t) for t in (_FILLER[:2] + _STELLA_HYP_REAL + _FILLER[2:])]
    _filtered, report = excise_stella_passage(utts, lang="en")
    assert report.n_excised == 1
    assert report.spans[0].score < 0.1          # ~0.014 in practice
    assert report.spans[0].n_tokens == len(_PASSAGE_TOKENS)


def test_negative_control_passage_mention_survives():
    # Real conversational text that *talks about* the passage must NOT be
    # excised, even surrounded by enough conversation to form windows.
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, t) for t in (_FILLER + _NEG_CONTROL + _FILLER)]
    filtered, report = excise_stella_passage(utts, lang="en")
    assert report.n_excised == 0
    survived = " ".join(u.text for u in filtered)
    assert "I wonder who came up with this." in survived
    assert "It is a very weird one." in survived


def test_excise_two_readings_in_one_stream():
    # The mixture-transcript case: BOTH speakers' readings land in one stream.
    # The matcher must remove both (cap = 2), and stop there.
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, t) for t in (
        _FILLER[:2] + _STELLA_HYP_REAL + _FILLER[2:6]
        + _STELLA_HYP_REAL + _FILLER[6:])]
    filtered, report = excise_stella_passage(utts, lang="en", session_id="mix")
    assert report.n_excised == 2
    assert "stella" not in " ".join(u.text for u in filtered).lower()


def test_excise_caps_at_two():
    # Three readings present, but the cap is 2 — the third survives (no
    # runaway loop eating real speech). Documents the hard cap.
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, t) for t in (
        _STELLA_HYP_REAL + _FILLER[:3]
        + _STELLA_HYP_REAL + _FILLER[3:6]
        + _STELLA_HYP_REAL)]
    _filtered, report = excise_stella_passage(utts, lang="en")
    assert report.n_excised == 2


def test_excise_noop_on_short_stream():
    pytest.importorskip("rapidfuzz")
    utts = [U(None, None, "hello there"), U(None, None, "how are you")]
    filtered, report = excise_stella_passage(utts, lang="en")
    assert report.n_excised == 0
    assert filtered == utts


# ---------------------------------------------------------------------------
# Layer 3 — untimed ref → tcpWER skipped; hyp_filter threaded
# ---------------------------------------------------------------------------


def _edacc_rec(tmp_path, *, pipeline_dir=None, rec_id="EAEC-X"):
    ref = tmp_path / "reference"
    ref.mkdir(exist_ok=True)
    (ref / "speaker_A.txt").write_text(
        format_untimed_gt(["my number is C three P one", "hello edinburgh"]),
        encoding="utf-8")
    (ref / "speaker_B.txt").write_text(
        format_untimed_gt(["my number is C three P two", "weather is windy"]),
        encoding="utf-8")
    return Recording(
        id=rec_id, dataset="edacc",
        mixture_path=tmp_path / f"{rec_id}.wav",
        reference_audio=None,
        reference_transcripts={"A": ref / "speaker_A.txt",
                               "B": ref / "speaker_B.txt"},
        reference_eaf=None,
        pipeline_dir=pipeline_dir,
        pipeline_nosep_dir=None, pipeline_noenh_dir=None,
        pipeline_minimal_dir=None,
    )


def _write_untimed_hyp(pdir, a_lines, b_lines):
    pdir.mkdir(exist_ok=True)
    (pdir / "transcript_A.txt").write_text(format_untimed_gt(a_lines), encoding="utf-8")
    (pdir / "transcript_B.txt").write_text(format_untimed_gt(b_lines), encoding="utf-8")
    import json
    (pdir / "metadata.json").write_text(
        json.dumps({"config": {"transcription": {"language": "en"}}}))


def test_layer3_untimed_ref_skips_tcpwer_visibly(tmp_path):
    pytest.importorskip("meeteval")
    pdir = tmp_path / "pipeline"
    _write_untimed_hyp(
        pdir,
        ["my number is C3P1", "hello edinburgh"],   # C3P1 must match "C three P one"
        ["my number is C3P2", "weather is windy"],
    )
    rec = _edacc_rec(tmp_path, pipeline_dir=pdir)
    l3 = compute_layer3(rec)
    assert l3 is not None
    assert l3["ref_untimed"] is True
    full = l3["modes"]["full"]
    assert full["tcp_skipped"] is True
    assert full["tcpwer"] is None                # never fabricated from fake times
    # cpWER still computed; the alphanumeric split makes C3P1/C3P2 match the GT.
    assert full["cpwer"] == pytest.approx(0.0)


def test_layer3_timed_ref_still_computes_tcpwer(tmp_path):
    # Control: a timed reference (CLARIN-style) keeps tcpWER — the skip is
    # strictly an untimed-only behaviour.
    pytest.importorskip("meeteval")
    from asr_pipeline.transcript_format import write_eaf
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")], "B": [(3.0, 4.0, "pies je")]},
              tmp_path / "r.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    pdir.joinpath("transcript_A.txt").write_text("[  1.00 → 2.00]  ala ma kota\n")
    pdir.joinpath("transcript_B.txt").write_text("[  3.00 → 4.00]  pies je\n")
    rec = Recording(
        id="r", dataset="clarin", mixture_path=tmp_path / "r.wav",
        reference_audio=None, reference_transcripts={}, reference_eaf=eaf,
        pipeline_dir=pdir, pipeline_nosep_dir=None, pipeline_noenh_dir=None,
        pipeline_minimal_dir=None,
    )
    l3 = compute_layer3(rec)
    assert l3["ref_untimed"] is False
    assert l3["modes"]["full"]["tcp_skipped"] is False
    assert l3["modes"]["full"]["tcpwer"] is not None


def test_layer3_hyp_filter_lowers_cpwer(tmp_path):
    # End-to-end: the passage appears in the hyp but the (untimed) ref dropped
    # it. Without the filter it's scored as insertions; with the filter the
    # passage is excised and cpWER drops.
    pytest.importorskip("meeteval")
    pdir = tmp_path / "pipeline"
    a_hyp = ["my number is C3P1"] + _STELLA_HYP_REAL + ["hello edinburgh"]
    b_hyp = ["my number is C3P2", "weather is windy"]
    _write_untimed_hyp(pdir, a_hyp, b_hyp)
    rec = _edacc_rec(tmp_path, pipeline_dir=pdir)

    no_filter = compute_layer3(rec)["modes"]["full"]["cpwer"]
    with_filter = compute_layer3(
        rec,
        hyp_filter=lambda u: excise_stella_passage(u, lang="en"),
    )["modes"]["full"]["cpwer"]
    assert with_filter < no_filter          # passage no longer charged as inserts


def test_layer3_hyp_filter_applies_to_mixture(tmp_path):
    # The mixture (ORC/MIMO) floor must also have the passage excised — it
    # appears in the single-stream mixture transcript too.
    pytest.importorskip("meeteval")
    pdir = tmp_path / "pipeline"
    _write_untimed_hyp(pdir, ["hello edinburgh"], ["weather is windy"])
    # Mixture transcript carries the passage twice (both readings).
    mix = ["my number is C3P1", "my number is C3P2"] + _STELLA_HYP_REAL \
        + _STELLA_HYP_REAL + ["hello edinburgh", "weather is windy"]
    pdir.joinpath("transcript_mixture.txt").write_text(
        format_untimed_gt(mix), encoding="utf-8")
    rec = _edacc_rec(tmp_path, pipeline_dir=pdir)

    no_filter = compute_layer3(rec)["mixture_orc"]["orc_wer"]
    with_filter = compute_layer3(
        rec, hyp_filter=lambda u: excise_stella_passage(u, lang="en"),
    )["mixture_orc"]["orc_wer"]
    assert with_filter < no_filter


# ---------------------------------------------------------------------------
# Prep-script reference parsing — IGNORE handling (no silent drops)
# ---------------------------------------------------------------------------


def _load_prep():
    """Import the prep script as a module (it lives under scripts/, not a package)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "prepare_edacc_eval",
        Path(__file__).resolve().parent.parent / "scripts" / "prepare_edacc_eval.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prep_drops_whole_utterance_ignore_with_count(tmp_path):
    prep = _load_prep()
    p = tmp_path / "EAEC-Z.txt"
    p.write_text(
        "# conversation: EAEC-Z\n"
        "1\tA\tHELLO MY NUMBER IS C THREE P ONE\n"
        "2\tB\tHELLO MY NUMBER IS C THREE P TWO\n"
        "3\tA\tIGNORE_TIME_SEGMENT_IN_SCORING\n"
        "4\tB\tIGNORE_TIME_SEGMENT_IN_SCORING\n"
        "5\tA\tREAL CONVERSATION HERE\n",
        encoding="utf-8",
    )
    by_spk, n_ignore, n_inline = prep.parse_edacc_transcript(p)
    assert n_ignore == 2
    assert n_inline == 0
    assert by_spk["A"] == ["HELLO MY NUMBER IS C THREE P ONE", "REAL CONVERSATION HERE"]
    assert by_spk["B"] == ["HELLO MY NUMBER IS C THREE P TWO"]
    # The marker never survives into the output.
    assert all("IGNORE_TIME_SEGMENT_IN_SCORING" not in t
               for ts in by_spk.values() for t in ts)


def test_prep_handles_inline_ignore_loudly(tmp_path, capsys):
    # The deliverable's defensive case: if IGNORE appears inline (not as the
    # whole field), report it loudly and strip the marker, keeping the text.
    prep = _load_prep()
    p = tmp_path / "EAEC-Y.txt"
    p.write_text(
        "1\tA\tREAL WORDS IGNORE_TIME_SEGMENT_IN_SCORING MORE WORDS\n",
        encoding="utf-8",
    )
    by_spk, n_ignore, n_inline = prep.parse_edacc_transcript(p)
    assert n_inline == 1
    assert by_spk["A"] == ["REAL WORDS MORE WORDS"]   # marker stripped, text kept
    assert "inline" in capsys.readouterr().out.lower()


def test_prep_word_containing_ignored_is_not_touched(tmp_path):
    # Real EdAcc case (EAEC-C35_P3): the word "IGNORED" must not be confused
    # with the IGNORE_TIME_SEGMENT_IN_SCORING marker.
    prep = _load_prep()
    p = tmp_path / "EAEC-W.txt"
    p.write_text("1\tA\tHE IGNORED THE SOLDIER\n", encoding="utf-8")
    by_spk, n_ignore, n_inline = prep.parse_edacc_transcript(p)
    assert n_ignore == 0 and n_inline == 0
    assert by_spk["A"] == ["HE IGNORED THE SOLDIER"]


def test_prep_intersection_reports_mismatches(tmp_path):
    prep = _load_prep()
    audio = tmp_path / "audios"
    trans = tmp_path / "transcripts"
    audio.mkdir(); trans.mkdir()
    for name in ("EAEC-A", "EAEC-B", "EAEC-C_P0", "EAEC-C_P1"):
        (audio / f"{name}.wav").touch()
    for name in ("EAEC-A", "EAEC-C_P0", "EAEC-C_P1", "EAEC-ORPHAN"):
        (trans / f"{name}.txt").touch()
    common, audio_only, transcript_only = prep._intersection(audio, trans)
    assert common == ["EAEC-A", "EAEC-C_P0", "EAEC-C_P1"]
    assert audio_only == ["EAEC-B"]                  # audio, no transcript
    assert transcript_only == ["EAEC-ORPHAN"]        # transcript, no audio
