"""Tests for the pure parts of ``scripts/build_nzr_aids.py``.

GPU-free: covers the robione.txt parser, the EAF -> nzr-occurrence scanner
(XML-unescape handling, bare-``<nzr>`` utterance, context windowing), and the
bundle-directory name sanitisation. The model phases are not exercised here.
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not a package; add it to the path like the script itself does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_nzr_aids as bna  # noqa: E402


# --------------------------------------------------------------------------- #
# robione.txt parsing
# --------------------------------------------------------------------------- #


def test_parse_robione_basic_tab_and_space():
    text = "065a9896__seg00\tdone\n150d1ccc__seg00 done\n"
    assert bna.parse_robione(text) == ["065a9896__seg00", "150d1ccc__seg00"]


def test_parse_robione_ignores_junk_lines():
    text = (
        "595aa511__seg00 done\r\n"          # CRLF, valid
        "\r\n"                               # blank
        "f7ed6ed8__seg00 - remove, whole recording is in clarin_gotowe\r\n"  # id but not done
        "KEEP IN MIND: search all eafs for utterances\r\n"  # free text
        "add normalization:\r\n"
        "okej <-> ok\r\n"
        "8993ff2e__seg00 done\r\n"           # valid
    )
    assert bna.parse_robione(text) == ["595aa511__seg00", "8993ff2e__seg00"]


def test_parse_robione_dedupes_preserving_order():
    text = "a1__seg00 done\nb2__seg01 done\na1__seg00 done\n"
    assert bna.parse_robione(text) == ["a1__seg00", "b2__seg01"]


def test_parse_robione_done_case_insensitive_and_extra_tokens():
    text = "x9__seg02   DONE  (checked twice)\n"
    assert bna.parse_robione(text) == ["x9__seg02"]


def test_parse_robione_id_alone_is_not_done():
    # A bare id with no `done` token must not count.
    assert bna.parse_robione("595aa511__seg00\n") == []


# --------------------------------------------------------------------------- #
# EAF -> nzr occurrence scanner
# --------------------------------------------------------------------------- #


_EAF_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2026-06-12T00:00:00+00:00"
    FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER>
{slots}
    </TIME_ORDER>
{tiers}
</ANNOTATION_DOCUMENT>
"""


def _write_eaf(tmp_path: Path, annotations: list[tuple[str, float, float, str]]) -> Path:
    """Build a minimal EAF.

    ``annotations`` = list of ``(tier, start_s, end_s, raw_value)``. The raw
    value is written verbatim into ``ANNOTATION_VALUE`` (caller supplies the
    XML-escaped form, e.g. ``&lt;nzr&gt;``), matching how ELAN stores it.
    """
    slots: list[str] = []
    tiers: dict[str, list[str]] = {}
    sid = 1
    aid = 1
    for tier, start, end, value in annotations:
        s1, s2 = f"ts{sid}", f"ts{sid + 1}"
        slots.append(f'        <TIME_SLOT TIME_SLOT_ID="{s1}" TIME_VALUE="{int(start * 1000)}"/>')
        slots.append(f'        <TIME_SLOT TIME_SLOT_ID="{s2}" TIME_VALUE="{int(end * 1000)}"/>')
        sid += 2
        tiers.setdefault(tier, []).append(
            f'            <ANNOTATION>\n'
            f'                <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{aid}" '
            f'TIME_SLOT_REF1="{s1}" TIME_SLOT_REF2="{s2}">\n'
            f'                    <ANNOTATION_VALUE>{value}</ANNOTATION_VALUE>\n'
            f'                </ALIGNABLE_ANNOTATION>\n'
            f'            </ANNOTATION>'
        )
        aid += 1
    tier_blocks = []
    for tier, anns in tiers.items():
        tier_blocks.append(
            f'    <TIER LINGUISTIC_TYPE_REF="default" TIER_ID="Speaker_{tier}">\n'
            + "\n".join(anns)
            + "\n    </TIER>"
        )
    eaf = _EAF_TEMPLATE.format(slots="\n".join(slots), tiers="\n".join(tier_blocks))
    path = tmp_path / "annotation.eaf"
    path.write_text(eaf, encoding="utf-8")
    return path


def test_scan_unescapes_and_finds_inline_nzr(tmp_path):
    eaf = _write_eaf(tmp_path, [
        ("A", 0.0, 1.0, "Zwykła wypowiedź."),
        ("B", 1.5, 3.0, "To jest &lt;nzr&gt; coś."),
    ])
    occs = bna.scan_eaf_for_nzr(eaf, "frag__seg00")
    assert len(occs) == 1
    occ = occs[0]
    assert occ.tier == "B"
    # The parser unescaped &lt;nzr&gt; -> <nzr> in the stored text.
    assert "<nzr>" in occ.text
    assert occ.text == "To jest <nzr> coś."


def test_scan_handles_bare_nzr_utterance(tmp_path):
    eaf = _write_eaf(tmp_path, [
        ("A", 0.0, 1.0, "Pełne zdanie."),
        ("A", 2.0, 2.4, "&lt;nzr&gt;"),
    ])
    occs = bna.scan_eaf_for_nzr(eaf, "frag__seg00")
    assert len(occs) == 1
    assert occs[0].text == "<nzr>"


def test_scan_indexes_and_orders_occurrences_chronologically(tmp_path):
    # Two tiers interleaved in time; both carry an nzr. Indices run 0,1 in
    # global chronological order regardless of tier.
    eaf = _write_eaf(tmp_path, [
        ("A", 0.0, 1.0, "start A"),
        ("B", 1.2, 2.0, "&lt;nzr&gt; B early"),   # earlier nzr -> index 0
        ("A", 2.5, 3.5, "A later &lt;nzr&gt;"),   # later nzr   -> index 1
    ])
    occs = bna.scan_eaf_for_nzr(eaf, "frag__seg00")
    assert [o.index for o in occs] == [0, 1]
    assert occs[0].tier == "B" and occs[0].start == 1.2
    assert occs[1].tier == "A" and occs[1].start == 2.5


def test_scan_context_spans_both_tiers(tmp_path):
    eaf = _write_eaf(tmp_path, [
        ("A", 0.0, 1.0, "first"),
        ("B", 1.1, 2.0, "second"),
        ("A", 2.1, 3.0, "third &lt;nzr&gt;"),   # the nzr utterance
        ("B", 3.1, 4.0, "fourth"),
        ("A", 4.1, 5.0, "fifth"),
    ])
    occs = bna.scan_eaf_for_nzr(eaf, "frag__seg00")
    assert len(occs) == 1
    occ = occs[0]
    # 2 preceding (both tiers), 2 following (both tiers).
    assert [u.text for _, u in occ.preceding] == ["first", "second"]
    assert [u.text for _, u in occ.following] == ["fourth", "fifth"]
    assert [t for t, _ in occ.preceding] == ["A", "B"]


def test_scan_no_nzr_returns_empty(tmp_path):
    eaf = _write_eaf(tmp_path, [("A", 0.0, 1.0, "Nic tu nie ma.")])
    assert bna.scan_eaf_for_nzr(eaf, "frag__seg00") == []


def test_scan_preceding_prompt_uses_nearest_preceding(tmp_path):
    eaf = _write_eaf(tmp_path, [
        ("A", 0.0, 1.0, "far"),
        ("B", 1.1, 2.0, "near"),
        ("A", 2.1, 3.0, "&lt;nzr&gt;"),
    ])
    occ = bna.scan_eaf_for_nzr(eaf, "frag__seg00")[0]
    assert bna.preceding_prompt(occ) == "near"


def test_preceding_prompt_empty_when_no_context(tmp_path):
    eaf = _write_eaf(tmp_path, [("A", 0.0, 1.0, "&lt;nzr&gt;")])
    occ = bna.scan_eaf_for_nzr(eaf, "frag__seg00")[0]
    assert bna.preceding_prompt(occ) == ""


# --------------------------------------------------------------------------- #
# Bundle directory name sanitisation
# --------------------------------------------------------------------------- #


def test_bundle_dir_name_is_drvfs_safe():
    occ = bna.NzrOccurrence(
        frag_id="f", tier="B", start=6.08, end=7.56, text="x", index=0
    )
    name = bna.bundle_dir_name(occ)
    assert name == "00_B_6_08s"
    for bad in (":", "<", ">", ".", "/", "\\", "|", "*", "?"):
        assert bad not in name


def test_bundle_dir_name_sanitises_exotic_tier_and_pads_index():
    occ = bna.NzrOccurrence(
        frag_id="f", tier="SPEAKER 1:x", start=123.4, end=124.0, text="x", index=7
    )
    name = bna.bundle_dir_name(occ)
    assert name == "07_SPEAKER_1_x_123_40s"
    for bad in (":", "<", ">", ".", " ", "/"):
        assert bad not in name


def test_bundle_dir_names_unique_per_index_on_same_start():
    # Two occurrences at the same tier+time stay distinct via the index prefix.
    base = dict(frag_id="f", tier="A", start=1.0, end=2.0, text="x")
    n0 = bna.bundle_dir_name(bna.NzrOccurrence(index=0, **base))
    n1 = bna.bundle_dir_name(bna.NzrOccurrence(index=1, **base))
    assert n0 != n1


# --------------------------------------------------------------------------- #
# Index helpers
# --------------------------------------------------------------------------- #


def test_recover_best_guess_from_guesses_body():
    body = (
        "ASR candidate readings — a GUESS MENU, not ground truth.\n"
        "Each line: <variant> @ temp <t> -> decoded text.\n"
        "Priming prompt (preceding utterance): 'x'\n"
        "\n"
        "mix    @ 0.0  ->  To jest pewno przez pół godziny.\n"
        "mix    @ 0.6  ->  To jest pewne.\n"
        "sep_A  @ 0.0  ->  Inna wersja.\n"
    )
    assert bna.recover_best_guess(body) == "To jest pewno przez pół godziny."


def test_recover_best_guess_empty_when_absent():
    assert bna.recover_best_guess("errors:\n  - transcription failed\n") == ""


def test_md_cell_keeps_nzr_visible_and_escapes_pipes():
    out = bna._md_cell("a | b <nzr> c")
    assert "\\|" in out
    # Angle brackets escaped so Markdown renderers don't eat the tag.
    assert "\\<nzr\\>" in out
    assert "<nzr>" not in out.replace("\\<nzr\\>", "")


# --------------------------------------------------------------------------- #
# slice_with_pad bounds
# --------------------------------------------------------------------------- #


def test_slice_with_pad_clamps_to_bounds():
    import numpy as np
    audio = np.arange(16_000, dtype=np.float32)  # 1 s @ 16 kHz
    # utterance near the start; pad would run negative -> clamp to 0
    sl = bna.slice_with_pad(audio, start_s=0.1, end_s=0.2, sr=16_000, pad_s=2.5)
    assert sl[0] == 0.0  # starts at sample 0
    assert len(sl) == len(audio)  # clamped to full array (pad exceeds both ends)


def test_slice_with_pad_never_empty():
    import numpy as np
    audio = np.arange(100, dtype=np.float32)
    # interval entirely past the array -> still returns >=1 sample, not empty
    sl = bna.slice_with_pad(audio, start_s=10.0, end_s=11.0, sr=16_000, pad_s=0.0)
    assert len(sl) >= 1


# ---------------------------------------------------------------------------
# collapse_repeats — hallucination-loop guard on the guess menu
# ---------------------------------------------------------------------------


def test_collapse_repeats_collapses_long_runs_with_count():
    assert (
        bna.collapse_repeats("no tak tak tak tak tak tak koniec")
        == "no tak tak tak [×6] koniec"
    )


def test_collapse_repeats_leaves_short_runs_alone():
    text = "tak tak tak no dobrze dobrze"
    assert bna.collapse_repeats(text) == text


def test_collapse_repeats_handles_trailing_run_and_empty():
    assert bna.collapse_repeats("a b b b b") == "a b b b [×4]"
    assert bna.collapse_repeats("") == ""


def test_recover_best_guess_ignores_unprimed_control_line():
    """The un-primed control line must not satisfy the primed-mix regex —
    recovery has to return the primed reading even when the un-primed line
    comes first in the file."""
    body = (
        "mix    @ 0.0 (no prompt)  ->  echo-free reading\n"
        "mix    @ 0.0  ->  primed reading\n"
    )
    assert bna.recover_best_guess(body) == "primed reading"
