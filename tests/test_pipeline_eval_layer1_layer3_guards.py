"""Guard/None-path tests for the eval orchestration layers L1 and L3.

The happy path (DER≈0, cpWER≈0 on matching output) is pinned by
test_pipeline_io_roundtrip.py. These tests cover the branches it doesn't:
the None returns when a reference is absent or empty, and the two defects
fixed in the 2026-06-10 review — the empty-reference ~5e9 DER (L1) and the
silent drop of a one-speaker pipeline collapse (L3).

All CPU, no models: a Recording is constructed directly over a tmp_path tree.
"""

import json

import pytest

from asr_pipeline.eval.layer1 import compute_layer1
from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.recordings import Recording
from asr_pipeline.transcript_format import write_eaf


def _rec(tmp_path, *, reference_eaf=None, reference_transcripts=None,
         reference_diarization=None, pipeline_dir=None, rec_id="rec1") -> Recording:
    return Recording(
        id=rec_id,
        dataset="clarin",
        mixture_path=tmp_path / f"{rec_id}.wav",
        reference_audio=None,
        reference_transcripts=reference_transcripts or {},
        reference_diarization=reference_diarization,
        reference_eaf=reference_eaf,
        pipeline_dir=pipeline_dir,
        pipeline_nosep_dir=None,
        pipeline_noenh_dir=None,
    )


def _write_diar_json(pdir, turns, total_s):
    """turns: list of (speaker, start_s, end_s)."""
    (pdir / "diarization.json").write_text(json.dumps({
        "turns": [{"speaker": s, "start": st, "end": e} for s, st, e in turns],
        "total_duration_s": total_s,
    }))


def _write_gt_txt(path, utts):
    """utts: list of (start_s, end_s, text), in the parse_gt_txt line format."""
    path.write_text(
        "".join(f"[{s:6.2f} → {e:6.2f}]  {t}\n" for s, e, t in utts),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Layer 1 — None guards (incl. the empty-reference regression)
# ---------------------------------------------------------------------------


def test_layer1_none_on_empty_eaf_reference(tmp_path):
    # Regression: an all-empty-tier EAF used to slip the `is None` guard and
    # produce DER ~5e9 (compute_der's 1e-9 floor). MUST have a populated
    # diarization.json, else the test would pass via the pdir/hyp guard for the
    # wrong reason — pre-fix this reached compute_der and returned a giant DER.
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [], "B": []}, tmp_path / "rec1.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_diar_json(pdir, [("SPEAKER_00", 0.0, 2.0), ("SPEAKER_01", 5.0, 7.0)], 10.0)

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=pdir)
    assert rec.reference_eaf is not None and rec.pipeline_dir is not None
    assert compute_layer1(rec) is None


def test_layer1_none_on_empty_rttm_reference(tmp_path):
    # SPKR-INFO-only RTTM parses to {} → must read as absent, not zero-turn.
    rttm = tmp_path / "diarization.rttm"
    rttm.write_text("SPKR-INFO rec1 1 <NA> <NA> <NA> unknown SPEAKER_00 <NA> <NA>\n")
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_diar_json(pdir, [("SPEAKER_00", 0.0, 2.0)], 10.0)

    rec = _rec(tmp_path, reference_diarization=rttm, pipeline_dir=pdir)
    assert compute_layer1(rec) is None


def test_layer1_none_when_no_pipeline_diarization(tmp_path):
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")]}, tmp_path / "rec1.wav", eaf)
    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=None)
    assert compute_layer1(rec) is None


def test_layer1_der_positive_on_diarization_mismatch(tmp_path):
    # A real (non-empty) reference + a hypothesis that doesn't match → der > 0.
    # Proves DER isn't trivially zero and that a populated reference is scored.
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 3.0, "ala")], "B": [(5.0, 7.0, "pies")]},
              tmp_path / "rec1.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_diar_json(pdir, [("SPEAKER_00", 0.0, 0.2)], 10.0)  # almost no overlap

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=pdir)
    l1 = compute_layer1(rec)
    assert l1 is not None
    assert l1["reference_source"] == "eaf"
    assert l1["der_stage1"]["der"] > 0.0


# ---------------------------------------------------------------------------
# Layer 3 — None guards + the one-speaker-collapse fix
# ---------------------------------------------------------------------------


def test_layer3_none_when_no_gt(tmp_path):
    rec = _rec(tmp_path)   # no EAF, no reference transcripts
    assert compute_layer3(rec) is None


def test_layer3_none_when_gt_missing_speaker_b(tmp_path):
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")], "B": []},
              tmp_path / "rec1.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_gt_txt(pdir / "transcript_A.txt", [(1.0, 2.0, "ala ma kota")])
    _write_gt_txt(pdir / "transcript_B.txt", [(5.0, 6.0, "pies je obiad")])

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=pdir)
    assert compute_layer3(rec) is None   # reference has no B → unscorable


def test_layer3_scores_one_speaker_collapse(tmp_path):
    # A one-speaker pipeline collapse writes only transcript_A.txt. Pre-fix the
    # both-required gate dropped the mode silently; now it must be SCORED, with
    # speaker B's words charged as deletions.
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")],
               "B": [(5.0, 6.0, "pies je obiad")]},
              tmp_path / "rec1.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_gt_txt(pdir / "transcript_A.txt", [(1.0, 2.0, "ala ma kota")])
    # deliberately NO transcript_B.txt

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=pdir)
    l3 = compute_layer3(rec)
    assert l3 is not None
    full = l3["modes"]["full"]
    assert full is not None                 # not silently dropped
    assert full["cpwer"] > 0.0              # B charged as deletions
    assert l3["modes"]["no_sep"] is None    # that dir was never created


def test_layer3_modes_dict_always_has_three_keys(tmp_path):
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")],
               "B": [(5.0, 6.0, "pies je obiad")]},
              tmp_path / "rec1.wav", eaf)
    pdir = tmp_path / "pipeline"
    pdir.mkdir()
    _write_gt_txt(pdir / "transcript_A.txt", [(1.0, 2.0, "ala ma kota")])
    _write_gt_txt(pdir / "transcript_B.txt", [(5.0, 6.0, "pies je obiad")])

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_dir=pdir)
    l3 = compute_layer3(rec)
    assert set(l3["modes"]) == {"full", "no_sep", "no_enh"}
    assert l3["modes"]["full"]["cpwer"] == pytest.approx(0.0)   # exact match
    assert l3["modes"]["no_sep"] is None and l3["modes"]["no_enh"] is None
    assert l3["mixture_orc"] is None and l3["mixture_mimo"] is None  # no mixture txt
