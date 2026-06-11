"""Guard/None-path tests for the L3 eval orchestration layer.

The happy path (cpWER≈0 on matching output) is pinned by
test_pipeline_io_roundtrip.py. These tests cover the branches it doesn't:
the None returns when a reference is absent or empty, and the defect fixed
in the 2026-06-10 review — the silent drop of a one-speaker pipeline
collapse (L3).

All CPU, no models: a Recording is constructed directly over a tmp_path tree.
"""

import pytest

from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.recordings import Recording
from asr_pipeline.transcript_format import write_eaf


def _rec(tmp_path, *, reference_eaf=None, reference_transcripts=None,
         pipeline_dir=None, pipeline_minimal_dir=None, rec_id="rec1") -> Recording:
    return Recording(
        id=rec_id,
        dataset="clarin",
        mixture_path=tmp_path / f"{rec_id}.wav",
        reference_audio=None,
        reference_transcripts=reference_transcripts or {},
        reference_eaf=reference_eaf,
        pipeline_dir=pipeline_dir,
        pipeline_nosep_dir=None,
        pipeline_noenh_dir=None,
        pipeline_minimal_dir=pipeline_minimal_dir,
    )


def _write_gt_txt(path, utts):
    """utts: list of (start_s, end_s, text), in the parse_gt_txt line format."""
    path.write_text(
        "".join(f"[{s:6.2f} → {e:6.2f}]  {t}\n" for s, e, t in utts),
        encoding="utf-8",
    )


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


def test_layer3_modes_dict_always_has_four_keys(tmp_path):
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
    assert set(l3["modes"]) == {"full", "no_sep", "no_enh", "minimal"}
    assert l3["modes"]["full"]["cpwer"] == pytest.approx(0.0)   # exact match
    assert l3["modes"]["no_sep"] is None and l3["modes"]["no_enh"] is None
    assert l3["modes"]["minimal"] is None   # that dir was never created
    assert l3["mixture_orc"] is None and l3["mixture_mimo"] is None  # no mixture txt


def test_layer3_scores_minimal_mode(tmp_path):
    # The (no-sep, no-enh) ablation arm — SCOPE §6 / open question 6: a
    # populated pipeline_minimal/ dir must be scored as mode "minimal".
    eaf = tmp_path / "annotation.eaf"
    write_eaf({"A": [(1.0, 2.0, "ala ma kota")],
               "B": [(5.0, 6.0, "pies je obiad")]},
              tmp_path / "rec1.wav", eaf)
    mdir = tmp_path / "pipeline_minimal"
    mdir.mkdir()
    _write_gt_txt(mdir / "transcript_A.txt", [(1.0, 2.0, "ala ma kota")])
    _write_gt_txt(mdir / "transcript_B.txt", [(5.0, 6.0, "pies je kolacje")])

    rec = _rec(tmp_path, reference_eaf=eaf, pipeline_minimal_dir=mdir)
    l3 = compute_layer3(rec)
    assert l3 is not None
    minimal = l3["modes"]["minimal"]
    assert minimal is not None
    assert minimal["cpwer"] == pytest.approx(1 / 6)   # 1 sub in 6 ref words
    assert l3["modes"]["full"] is None


def test_load_recording_discovers_pipeline_minimal(tmp_path):
    # Discovery side: load_recording must populate pipeline_minimal_dir from
    # the on-disk pipeline_minimal/ subdir, same as the other three modes.
    from asr_pipeline.eval.recordings import load_recording

    rec_dir = tmp_path / "clarin" / "rec1"
    rec_dir.mkdir(parents=True)
    (rec_dir / "rec1.wav").touch()          # existence is all discovery checks
    (rec_dir / "pipeline_minimal").mkdir()

    rec = load_recording(rec_dir)
    assert rec is not None
    assert rec.pipeline_minimal_dir == rec_dir / "pipeline_minimal"
    assert rec.pipeline_dir is None
