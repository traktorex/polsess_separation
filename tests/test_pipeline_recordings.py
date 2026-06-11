"""Unit tests for eval-tree discovery (`recordings.py`) + transcript parsers.

Builds a synthetic eval tree under tmp_path. Discovery only checks file
existence (it never decodes audio), so empty `.wav` files suffice.
"""

import pytest

from asr_pipeline.eval.recordings import (
    load_recording,
    load_reference_utterances,
    parse_rttm,
    walk_eval_tree,
)
from asr_pipeline.eval.transcript_parser import parse_gt_txt, parse_transcript_file
from asr_pipeline.transcript_format import write_eaf


@pytest.fixture
def eval_tree(tmp_path):
    """<root>/clarin/{rec1,rec2,rec3} — new layout, old layout, invalid."""
    root = tmp_path / "eval"

    # rec1 — new layout: <id>.wav, root EAF, reference/, pipeline/.
    rec1 = root / "clarin" / "rec1"
    (rec1 / "reference").mkdir(parents=True)
    (rec1 / "pipeline").mkdir()
    (rec1 / "rec1.wav").touch()
    write_eaf(
        {"A": [(0.5, 1.5, "z eaf")]},
        media_path=rec1 / "rec1.wav",
        eaf_path=rec1 / "annotation.eaf",
    )
    (rec1 / "reference" / "speaker_A.txt").write_text(
        "[  0.50 →   1.50]  z pliku txt\n", encoding="utf-8"
    )
    (rec1 / "reference" / "speaker_B.txt").write_text(
        "[  2.00 →   3.00]  druga osoba\n", encoding="utf-8"
    )
    (rec1 / "reference" / "diarization.rttm").write_text(
        "SPKR-INFO file1 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
        "SPEAKER file1 1 0.50 1.00 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER file1 1 2.00 0.50 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8",
    )

    # rec2 — old layout: only mixture.wav.
    rec2 = root / "clarin" / "rec2"
    rec2.mkdir(parents=True)
    (rec2 / "mixture.wav").touch()

    # rec3 — no audio at all -> not a recording.
    (root / "clarin" / "rec3").mkdir(parents=True)

    return root


# ---------------------------------------------------------------------------
# load_recording / walk_eval_tree
# ---------------------------------------------------------------------------


def test_load_recording_new_layout(eval_tree):
    rec = load_recording(eval_tree / "clarin" / "rec1")
    assert rec is not None
    assert rec.id == "rec1" and rec.dataset == "clarin"
    assert rec.mixture_path.name == "rec1.wav"
    assert rec.reference_eaf is not None
    assert set(rec.reference_transcripts) == {"A", "B"}
    assert rec.reference_audio is None          # no reference wavs created
    assert rec.reference_diarization is not None
    assert rec.pipeline_dir is not None
    assert rec.pipeline_nosep_dir is None and rec.pipeline_noenh_dir is None


def test_load_recording_old_mixture_layout(eval_tree):
    rec = load_recording(eval_tree / "clarin" / "rec2")
    assert rec is not None
    assert rec.mixture_path.name == "mixture.wav"
    assert rec.reference_eaf is None
    assert rec.reference_transcripts == {}


def test_load_recording_without_audio_is_none(eval_tree):
    assert load_recording(eval_tree / "clarin" / "rec3") is None


def test_walk_eval_tree_skips_invalid(eval_tree):
    recs = list(walk_eval_tree(eval_tree))
    assert [r.id for r in recs] == ["rec1", "rec2"]


def test_walk_eval_tree_dataset_filter(eval_tree):
    assert len(list(walk_eval_tree(eval_tree, dataset="clarin"))) == 2
    assert list(walk_eval_tree(eval_tree, dataset="libricss")) == []


def test_walk_eval_tree_missing_root(tmp_path):
    assert list(walk_eval_tree(tmp_path / "nope")) == []


def test_reference_eaf_preferred_over_txt(eval_tree):
    rec = load_recording(eval_tree / "clarin" / "rec1")
    utts = load_reference_utterances(rec)
    # Both EAF and reference txt exist with different text — EAF must win.
    assert utts["A"][0].text == "z eaf"


def test_reference_txt_fallback(eval_tree):
    # Remove the EAF -> the per-speaker txt files take over.
    (eval_tree / "clarin" / "rec1" / "annotation.eaf").unlink()
    rec = load_recording(eval_tree / "clarin" / "rec1")
    utts = load_reference_utterances(rec)
    assert utts["A"][0].text == "z pliku txt"
    assert utts["B"][0].text == "druga osoba"


# ---------------------------------------------------------------------------
# parse_rttm
# ---------------------------------------------------------------------------


def test_parse_rttm(eval_tree):
    rttm = eval_tree / "clarin" / "rec1" / "reference" / "diarization.rttm"
    turns = parse_rttm(rttm)
    assert turns == {"A": [(0.5, 1.5)], "B": [(2.0, 2.5)]}   # start + dur -> end


# ---------------------------------------------------------------------------
# parse_gt_txt / parse_transcript_file
# ---------------------------------------------------------------------------


def test_parse_gt_txt_both_timestamp_formats(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text(
        "# komentarz\n"
        "\n"
        "[00:01:01.20 → 00:01:03.45] pierwszy segment\n"
        "[  4.00 →   5.50]  drugi segment\n"
        "[  6.00 →   6.50]  \n",          # empty text -> skipped
        encoding="utf-8",
    )
    utts = parse_gt_txt(p)
    assert len(utts) == 2
    assert utts[0].start == pytest.approx(61.20)   # HH:MM:SS folded to seconds
    assert utts[0].end == pytest.approx(63.45)
    assert utts[0].text == "pierwszy segment"
    assert utts[1].start == pytest.approx(4.0)


def test_parse_transcript_file_headers_and_arrows(tmp_path):
    p = tmp_path / "transcript.txt"
    p.write_text(
        "[  0.10 →   0.50]  przed nagłówkiem (ignorowane)\n"
        "=== Speaker A (SPEAKER_00) ===\n"
        "[  1.00 →   2.00]  pierwsza kwestia\n"
        "[  2.50 →   3.00]  druga\n"
        "\n"
        "=== Speaker B (SPEAKER_01) ===\n"
        "[  4.00 ->  5.00]  inna strzałka\n",
        encoding="utf-8",
    )
    out = parse_transcript_file(p)
    assert set(out) == {"A", "B"}
    assert [u.text for u in out["A"]] == ["pierwsza kwestia", "druga"]
    assert out["B"][0].start == pytest.approx(4.0)   # `->` arrow accepted
