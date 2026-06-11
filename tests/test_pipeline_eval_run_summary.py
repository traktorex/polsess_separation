"""Tests for the eval orchestrator (run.py) and the summary DataFrames.

Two untested surfaces the roundtrip test doesn't reach:
- evaluate_many's SQUIM lifecycle — load once across recordings, and unload
  even when a layer raises (the GPU-leak-on-failure class, cf. commit 9ec1353);
- summary.py's None-row handling and percentage conversions.

All CPU, no models: the layer computations are monkeypatched at the
asr_pipeline.eval.run binding (where evaluate_recording resolves them).
"""

import pandas as pd
import pytest

from asr_pipeline.eval import run
from asr_pipeline.eval.run import ScoreCard, evaluate_many
from asr_pipeline.eval.recordings import Recording
from asr_pipeline.eval.summary import (
    summarize_layer2_intrusive,
    summarize_layer2_squim,
    summarize_layer3,
)


def _min_rec(rec_id="rec1") -> Recording:
    return Recording(
        id=rec_id, dataset="clarin", mixture_path=None,
        reference_audio=None, reference_transcripts={},
        reference_eaf=None,
        pipeline_dir=None, pipeline_nosep_dir=None, pipeline_noenh_dir=None,
    )


# ---------------------------------------------------------------------------
# evaluate_many — SQUIM lifecycle
# ---------------------------------------------------------------------------


def test_evaluate_many_loads_squim_once_and_threads_it(monkeypatch):
    sentinel = object()
    loads: list = []
    unloads: list = []
    seen_models: list = []
    monkeypatch.setattr(run, "load_squim_model",
                        lambda: (loads.append(1), (sentinel, "cpu"))[1])
    monkeypatch.setattr(run, "unload_squim_model", lambda m: unloads.append(m))
    monkeypatch.setattr(run, "compute_layer3",
                        lambda rec, tcp_collar_s, hyp_filter=None: None)
    monkeypatch.setattr(
        run, "compute_layer2",
        lambda rec, sr, squim_model, squim_device: seen_models.append(squim_model),
    )

    cards = evaluate_many([_min_rec("r1"), _min_rec("r2")])

    assert len(cards) == 2
    assert len(loads) == 1                       # loaded exactly once
    assert unloads == [sentinel]                 # unloaded exactly once
    assert seen_models == [sentinel, sentinel]   # shared model threaded to each L2


def test_evaluate_many_unloads_squim_on_error(monkeypatch):
    sentinel = object()
    unloads: list = []
    monkeypatch.setattr(run, "load_squim_model", lambda: (sentinel, "cpu"))
    monkeypatch.setattr(run, "unload_squim_model", lambda m: unloads.append(m))
    monkeypatch.setattr(run, "compute_layer2",
                        lambda rec, sr, squim_model, squim_device: None)

    def _boom(rec, tcp_collar_s, hyp_filter=None):
        raise RuntimeError("L3 boom")

    monkeypatch.setattr(run, "compute_layer3", _boom)

    with pytest.raises(RuntimeError, match="L3 boom"):
        evaluate_many([_min_rec("r1")])
    assert unloads == [sentinel]                 # finally unloaded despite error


# ---------------------------------------------------------------------------
# summary.py — None rows + percentage conversions
# ---------------------------------------------------------------------------


def test_summarize_layer3_drops_none_layer():
    card = ScoreCard(_min_rec(), None, None)
    assert len(summarize_layer3([card])) == 0


def test_summarize_layer3_percentages_and_none_modes():
    l3 = {
        "ref_lengths": {"A": 3, "B": 2},
        "modes": {
            "full":   {"cpwer": 0.20, "tcpwer": 0.25},
            "no_sep": None,
            "no_enh": {"cpwer": 0.40, "tcpwer": 0.50},
        },
        "mixture_orc": {"orc_wer": 0.60},
        "mixture_mimo": None,
        "tcp_collar_s": 5.0,
    }
    r = summarize_layer3([ScoreCard(_min_rec(), None, l3)]).iloc[0]
    assert r["ref_n_utts"] == 5
    assert r["full_cpwer"] == pytest.approx(20.0)
    assert r["full_tcpwer"] == pytest.approx(25.0)
    assert r["no_enh_cpwer"] == pytest.approx(40.0)
    assert pd.isna(r["no_sep_cpwer"])            # None mode → None cell
    assert r["mixture_orc"] == pytest.approx(60.0)
    assert pd.isna(r["mixture_mimo"])            # None mixture → None cell


def test_summarize_layer2_skips_when_no_layer2():
    card = ScoreCard(_min_rec(), None, None)
    assert len(summarize_layer2_intrusive([card])) == 0
    assert len(summarize_layer2_squim([card])) == 0
