"""Unit tests for the eval transcript parsers (`transcript_parser.py`).

Focus on the two thesis-number-bearing behaviours the review flagged:

- `parse_gt_txt` must *warn visibly* (SCOPE §4.1, no-silent-substitution) when
  it skips a `[`-bracketed line it cannot parse — the negative-timestamp case
  an older writer produced — while still skipping plain comments / blanks
  silently.
- The writer→reader roundtrip (`format_transcript` → `parse_gt_txt`) is the
  contract the clamp in `stages/transcription.py:_finite_or_zero` protects: a
  clamped-to-zero start must survive the roundtrip rather than vanish.

`parse_eaf` time-slot hygiene (float values, missing-ref skip) is covered too.
"""

import pytest

from asr_pipeline.eval.transcript_parser import (
    parse_eaf,
    parse_gt_txt,
)
from asr_pipeline.stages.transcription import _normalise_result
from asr_pipeline.transcript_format import format_transcript, write_eaf


# ---------------------------------------------------------------------------
# parse_gt_txt — visible warning on unparseable bracketed lines
# ---------------------------------------------------------------------------


def test_parse_gt_txt_warns_on_unparseable_bracketed_line(tmp_path, capsys):
    # `[ -0.30 → ...]` matches neither grammar (the seconds regex rejects the
    # leading `-`): the utterance is dropped, but it must NOT drop silently.
    p = tmp_path / "gt.txt"
    p.write_text(
        "[ -0.30 →   1.50]  ginie po cichu\n"
        "[  2.00 →   3.00]  zostaje\n",
        encoding="utf-8",
    )
    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["zostaje"]
    out = capsys.readouterr().out
    assert "skipped unparseable bracketed line" in out
    assert "-0.30" in out


def test_parse_gt_txt_silent_on_comments_and_blanks(tmp_path, capsys):
    # Plain comments and blank lines are not bracketed — they must stay silent
    # (the load-bearing constraint the warn must not trip).
    p = tmp_path / "gt.txt"
    p.write_text(
        "# komentarz\n"
        "\n"
        "[  4.00 →   5.50]  segment\n",
        encoding="utf-8",
    )
    utts = parse_gt_txt(p)
    assert len(utts) == 1
    out = capsys.readouterr().out
    assert "skipped unparseable bracketed line" not in out


def test_parse_gt_txt_both_formats_no_false_warning(tmp_path, capsys):
    # Both accepted timestamp grammars must parse without warning.
    p = tmp_path / "gt.txt"
    p.write_text(
        "[00:01:01.20 → 00:01:03.45] pierwszy\n"
        "[  4.00 →   5.50]  drugi\n",
        encoding="utf-8",
    )
    utts = parse_gt_txt(p)
    assert len(utts) == 2
    assert "skipped unparseable bracketed line" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# writer → reader roundtrip (the clamp contract)
# ---------------------------------------------------------------------------


def test_clamp_roundtrip_negative_start_survives(tmp_path):
    """`_normalise_result` clamps a negative start to 0.0; the rendered line is
    then re-readable by `parse_gt_txt` — the whole utterance survives instead of
    vanishing through the non-negative-only grammar."""
    result = _normalise_result(
        {"segments": [
            {"start": -0.30, "end": 1.50, "text": "pierwszy"},
            {"start": 2.00, "end": 3.00, "text": "drugi"},
        ]},
        "pl",
    )
    txt = format_transcript(result)
    p = tmp_path / "transcript_A.txt"
    p.write_text(txt + "\n", encoding="utf-8")

    utts = parse_gt_txt(p)
    assert [u.text for u in utts] == ["pierwszy", "drugi"]
    assert utts[0].start == pytest.approx(0.0)   # clamped, not dropped
    assert utts[0].end == pytest.approx(1.50)


# ---------------------------------------------------------------------------
# parse_eaf — time-slot hygiene
# ---------------------------------------------------------------------------


def test_parse_eaf_reads_written_eaf(tmp_path):
    media = tmp_path / "rec.wav"
    media.touch()
    eaf = tmp_path / "annotation.eaf"
    write_eaf(
        {"A": [(0.5, 1.5, "z eaf A")], "B": [(2.0, 3.0, "z eaf B")]},
        media_path=media,
        eaf_path=eaf,
    )
    out = parse_eaf(eaf)
    assert out["A"][0].text == "z eaf A"
    assert out["A"][0].start == pytest.approx(0.5)
    assert out["B"][0].end == pytest.approx(3.0)


def test_parse_eaf_skips_annotation_with_missing_time_ref(tmp_path):
    # A hand-edited EAF whose annotation points at a non-existent time slot:
    # `slots.get` returns None and the utterance is skipped (boundary guard),
    # not crashed on.
    eaf = tmp_path / "annotation.eaf"
    eaf.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ANNOTATION_DOCUMENT FORMAT="3.0" VERSION="3.0">\n'
        "  <TIME_ORDER>\n"
        '    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="500"/>\n'
        '    <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>\n'
        "  </TIME_ORDER>\n"
        '  <TIER TIER_ID="Speaker_A">\n'
        "    <ANNOTATION>\n"
        '      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" '
        'TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">\n'
        "        <ANNOTATION_VALUE>dobry</ANNOTATION_VALUE>\n"
        "      </ALIGNABLE_ANNOTATION>\n"
        "    </ANNOTATION>\n"
        "    <ANNOTATION>\n"
        '      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" '
        'TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts_missing">\n'
        "        <ANNOTATION_VALUE>zly</ANNOTATION_VALUE>\n"
        "      </ALIGNABLE_ANNOTATION>\n"
        "    </ANNOTATION>\n"
        "  </TIER>\n"
        "</ANNOTATION_DOCUMENT>\n",
        encoding="utf-8",
    )
    out = parse_eaf(eaf)
    assert [u.text for u in out["A"]] == ["dobry"]   # the dangling-ref one dropped
