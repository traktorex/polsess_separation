"""Unit tests for transcript_format (writer side) + parser round-trips.

The writer/reader pairs are the load-bearing contracts:
- `format_transcript` output must stay parseable by `parse_gt_txt`.
- `write_eaf` output must stay parseable by `parse_eaf` (the GT workflow).
"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from asr_pipeline.eval.transcript_parser import parse_eaf, parse_gt_txt
from asr_pipeline.transcript_format import (
    _whisper_segments_to_tuples,
    format_transcript,
    to_jsonable,
    write_eaf,
    write_eaf_from_whisper_results,
)


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------


def test_format_transcript_roundtrips_through_parse_gt_txt(tmp_path):
    result = {
        "segments": [
            {"start": 1.0, "end": 2.5, "text": " Ala ma kota. "},
            {"start": 3.0, "end": 4.25, "text": "Drugi segment"},
        ],
    }
    path = tmp_path / "transcript_A.txt"
    path.write_text(format_transcript(result) + "\n", encoding="utf-8")

    utts = parse_gt_txt(path)
    assert len(utts) == 2
    assert utts[0].start == pytest.approx(1.0)
    assert utts[0].end == pytest.approx(2.5)
    assert utts[0].text == "Ala ma kota."
    assert utts[1].text == "Drugi segment"


def test_format_transcript_falls_back_to_text_without_segments():
    assert format_transcript({"segments": [], "text": "  cześć  "}) == "cześć"


def test_format_transcript_non_dict_is_empty():
    assert format_transcript(None) == ""


def test_format_transcript_tolerates_nonfinite_timestamps():
    # NaN / None per-segment boundaries (WhisperX's interpolate_nans can emit an
    # all-NaN segment) must not raise or leak literal `nan` into the .txt.
    result = {
        "segments": [
            {"start": None, "end": 1.0, "text": "a"},
            {"start": float("nan"), "end": float("nan"), "text": "b"},
        ],
    }
    out = format_transcript(result)
    assert "nan" not in out.lower()
    assert "[  0.00" in out          # coerced to 0.0


def test_format_transcript_collapses_embedded_newline():
    # A segment carrying an embedded newline must stay on one physical line, or
    # parse_gt_txt would split it into a malformed second utterance.
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "ala\nma kota"}]}
    out = format_transcript(result)
    assert out.count("\n") == 0
    assert "ala ma kota" in out


# ---------------------------------------------------------------------------
# to_jsonable
# ---------------------------------------------------------------------------


def test_to_jsonable_handles_numpy_and_torch():
    payload = {
        "f": np.float64(1.5),
        "i": np.int32(3),
        "b": np.bool_(True),
        "arr": np.array([1, 2]),
        "t": torch.tensor([1.0, 2.0]),
        "nested": [{"x": np.float32(0.25)}],
        "s": "ok",
        "none": None,
    }
    out = to_jsonable(payload)
    assert out["f"] == 1.5 and isinstance(out["f"], float)
    assert out["i"] == 3 and isinstance(out["i"], int)
    assert out["b"] is True and isinstance(out["b"], bool)
    assert out["arr"] == [1, 2]
    assert out["t"] == [1.0, 2.0]
    assert out["nested"][0]["x"] == pytest.approx(0.25)
    # The whole structure must be json.dump-safe.
    json.dumps(out)


def test_to_jsonable_passthrough_natives():
    for v in ("tekst", 7, 1.5, True, None):
        assert to_jsonable(v) == v


def test_to_jsonable_unhandled_type_logs_and_passes_through(capsys):
    # A non-JSON-native scalar (here: a plain object) hits the contingency
    # branch — it is passed through unchanged AND surfaced via dlog so a
    # recurring unhandled type gets noticed instead of failing silently in a
    # later json.dump.
    class _Weird:
        pass

    obj = _Weird()
    out = to_jsonable({"x": obj})
    assert out["x"] is obj                     # passed through, not converted
    captured = capsys.readouterr().out
    assert "unhandled type" in captured
    assert "_Weird" in captured


# ---------------------------------------------------------------------------
# format_transcript empty / None fallbacks
# ---------------------------------------------------------------------------


def test_format_transcript_empty_dict_is_empty():
    # No segments, no text → empty string (not "None", not a raise).
    assert format_transcript({}) == ""


def test_format_transcript_text_none_is_empty():
    # The top-level text fallback must tolerate an explicit None value.
    assert format_transcript({"segments": [], "text": None}) == ""


# ---------------------------------------------------------------------------
# write_eaf <-> parse_eaf round-trip (the GT correction format)
# ---------------------------------------------------------------------------


def test_write_eaf_parse_eaf_roundtrip(tmp_path):
    utts = {
        "A": [(0.5, 1.25, "ala ma kota"), (2.0, 3.0, "drugi")],
        "B": [(1.0, 1.9, "tak")],
    }
    eaf_path = tmp_path / "annotation.eaf"
    write_eaf(utts, media_path=tmp_path / "rec.wav", eaf_path=eaf_path)

    tiers = parse_eaf(eaf_path)
    assert set(tiers) == {"A", "B"}        # Speaker_ prefix stripped
    assert len(tiers["A"]) == 2
    a0 = tiers["A"][0]
    assert a0.start == pytest.approx(0.5, abs=1e-3)   # ms rounding tolerance
    assert a0.end == pytest.approx(1.25, abs=1e-3)
    assert a0.text == "ala ma kota"
    assert tiers["B"][0].text == "tak"


def test_write_eaf_sorts_on_parse(tmp_path):
    # parse_eaf sorts by start even if written out of order.
    utts = {"A": [(5.0, 6.0, "później"), (1.0, 2.0, "wcześniej")]}
    eaf_path = tmp_path / "a.eaf"
    write_eaf(utts, media_path=tmp_path / "rec.wav", eaf_path=eaf_path)
    tiers = parse_eaf(eaf_path)
    assert [u.text for u in tiers["A"]] == ["wcześniej", "później"]


def test_write_eaf_from_whisper_results(tmp_path):
    results = {
        "A": {"segments": [{"start": 0.0, "end": 1.0, "text": " cześć "}]},
        "B": {"segments": [{"start": 1.0, "end": 2.0, "text": "   "}]},
    }
    eaf_path = tmp_path / "out.eaf"
    n = write_eaf_from_whisper_results(
        results, media_path=tmp_path / "rec.wav", eaf_path=eaf_path
    )
    assert n == 1                          # B's empty segment dropped
    tiers = parse_eaf(eaf_path)
    assert set(tiers) == {"A"}             # empty tier never written


# ---------------------------------------------------------------------------
# write_eaf MEDIA_DESCRIPTOR / relative-path logic (S6)
# ---------------------------------------------------------------------------


def _media_descriptor(eaf_path):
    root = ET.fromstring(eaf_path.read_bytes())
    md = root.find("./HEADER/MEDIA_DESCRIPTOR")
    assert md is not None
    return md


def test_write_eaf_media_descriptor_same_dir(tmp_path):
    # Audio next to the EAF → RELATIVE_MEDIA_URL is "./rec.wav"; MEDIA_URL is an
    # absolute file:// URL (ELAN's whole point is finding the audio).
    eaf_path = tmp_path / "annotation.eaf"
    media = tmp_path / "rec.wav"
    write_eaf({"A": [(0.0, 1.0, "x")]}, media_path=media, eaf_path=eaf_path)

    md = _media_descriptor(eaf_path)
    assert md.get("RELATIVE_MEDIA_URL") == "./rec.wav"
    assert md.get("MEDIA_URL").startswith("file://")
    assert md.get("MEDIA_URL").endswith("rec.wav")
    assert md.get("MIME_TYPE") == "audio/x-wav"


def test_write_eaf_media_descriptor_parent_dir(tmp_path):
    # Audio one directory up from the EAF → the relative URL must climb with
    # "../" so ELAN resolves it after a move that preserves layout.
    sub = tmp_path / "transcripts"
    sub.mkdir()
    eaf_path = sub / "annotation.eaf"
    media = tmp_path / "rec.wav"
    write_eaf({"A": [(0.0, 1.0, "x")]}, media_path=media, eaf_path=eaf_path)

    md = _media_descriptor(eaf_path)
    assert md.get("RELATIVE_MEDIA_URL") == "../rec.wav"


def test_write_eaf_dedups_shared_time_boundaries(tmp_path):
    # Both speakers share the boundary 1.0 s. The TIME_ORDER block must carry it
    # as a single TIME_SLOT (the dedup the docstring promises), not one per use.
    utts = {
        "A": [(0.0, 1.0, "a")],
        "B": [(1.0, 2.0, "b")],
    }
    eaf_path = tmp_path / "shared.eaf"
    write_eaf(utts, media_path=tmp_path / "rec.wav", eaf_path=eaf_path)

    root = ET.fromstring(eaf_path.read_bytes())
    slots = root.findall("./TIME_ORDER/TIME_SLOT")
    values = [s.get("TIME_VALUE") for s in slots]
    # 3 distinct boundaries (0, 1000, 2000 ms), not 4 — the shared 1000 ms
    # appears exactly once.
    assert values.count("1000") == 1
    assert sorted(int(v) for v in values) == [0, 1000, 2000]
    # ...and every TIME_VALUE is unique (the dedup invariant, asserted directly).
    assert len(values) == len(set(values))


def test_write_eaf_from_whisper_results_all_empty_writes_nothing(tmp_path):
    eaf_path = tmp_path / "empty.eaf"
    n = write_eaf_from_whisper_results(
        {"A": {"segments": []}}, media_path=tmp_path / "rec.wav", eaf_path=eaf_path
    )
    assert n == 0
    assert not eaf_path.exists()


# ---------------------------------------------------------------------------
# Non-finite timestamp tolerance at the leaf functions
# ---------------------------------------------------------------------------


def test_whisper_segments_to_tuples_tolerates_nonfinite_timestamps():
    result = {
        "segments": [
            {"start": None, "end": 1.0, "text": "a"},
            {"start": float("nan"), "end": float("nan"), "text": "b"},
        ],
    }
    out = _whisper_segments_to_tuples(result)
    assert [t for _, _, t in out] == ["a", "b"]
    # No NaN / None survives — every boundary is a finite float.
    for start, end, _ in out:
        assert isinstance(start, float) and isinstance(end, float)
        assert start == start and end == end          # not NaN
    assert out[0][0] == 0.0 and out[1][0] == 0.0 and out[1][1] == 0.0


def test_write_eaf_from_whisper_results_tolerates_nonfinite_timestamps(tmp_path):
    # A NaN boundary used to raise `ValueError: cannot convert float NaN to
    # integer` in write_eaf, losing the whole recording's output.
    results = {
        "A": {"segments": [
            {"start": None, "end": 1.0, "text": "cześć"},
            {"start": float("nan"), "end": float("nan"), "text": "świat"},
        ]},
    }
    eaf_path = tmp_path / "nonfinite.eaf"
    n = write_eaf_from_whisper_results(
        results, media_path=tmp_path / "rec.wav", eaf_path=eaf_path
    )
    assert n == 2
    tiers = parse_eaf(eaf_path)                        # round-trips, no raise
    assert {u.text for u in tiers["A"]} == {"cześć", "świat"}


def test_write_eaf_nudges_zero_width_annotation(tmp_path):
    # start == end must still produce REF1 < REF2 (ELAN rejects coincident
    # boundaries); the writer nudges end to span ≥1 ms.
    utts = {"A": [(1.0, 1.0, "punkt")]}
    eaf_path = tmp_path / "zerowidth.eaf"
    write_eaf(utts, media_path=tmp_path / "rec.wav", eaf_path=eaf_path)
    tiers = parse_eaf(eaf_path)
    assert tiers["A"][0].end > tiers["A"][0].start
