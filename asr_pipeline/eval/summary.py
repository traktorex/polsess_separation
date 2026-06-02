"""Tabular summaries of `ScoreCard` lists.

Notebook-facing helpers. Each returns a ``pandas.DataFrame`` ready for
``display(df)``. Aggregation across recordings stays caller-side; these
helpers just unpack the per-recording dicts into rows.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from asr_pipeline.eval.run import ScoreCard


# ---------------------------------------------------------------------------
# Layer 1 — DER
# ---------------------------------------------------------------------------


def summarize_layer1(scores: Iterable[ScoreCard]) -> pd.DataFrame:
    """One row per recording. Columns: DER, miss, false_alarm, confusion (all %)."""
    rows = []
    for s in scores:
        row = {"dataset": s.dataset, "id": s.id}
        if s.layer1 is None:
            row.update({"der_pct": None, "miss_pct": None, "fa_pct": None, "conf_pct": None})
        else:
            d = s.layer1["der_stage1"]
            row.update({
                "der_pct":  100.0 * d["der"],
                "miss_pct": 100.0 * d["miss"],
                "fa_pct":   100.0 * d["false_alarm"],
                "conf_pct": 100.0 * d["confusion"],
                "total_ref_s": d["total_ref_s"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 2 — audio quality
# ---------------------------------------------------------------------------


def summarize_layer2_intrusive(scores: Iterable[ScoreCard]) -> pd.DataFrame:
    """One row per (recording, speaker). Intrusive scores only — recordings
    without oracle audio are excluded from this table."""
    rows = []
    for s in scores:
        if s.layer2 is None or s.layer2.get("intrusive") is None:
            continue
        for label, m in s.layer2["intrusive"].items():
            rows.append({
                "dataset": s.dataset, "id": s.id, "speaker": label,
                "si_sdr_db": m["si_sdr"],
                "si_sdr_baseline_db": m["si_sdr_baseline"],
                "si_sdri_db": m["si_sdri"],
                "pesq": m["pesq"],
                "pesq_baseline": m["pesq_baseline"],
                "pesqi": m["pesqi"],
                "stoi": m["stoi"],
                "stoi_baseline": m["stoi_baseline"],
                "stoii": m["stoii"],
                "pesq_n_scored": m["pesq_n_scored"],
            })
    return pd.DataFrame(rows)


def summarize_layer2_squim(scores: Iterable[ScoreCard]) -> pd.DataFrame:
    """One row per (recording, stream). Non-intrusive SQUIM — always
    available when the pipeline ran."""
    rows = []
    for s in scores:
        if s.layer2 is None or s.layer2.get("squim") is None:
            continue
        for stream_name, q in s.layer2["squim"].items():
            rows.append({
                "dataset": s.dataset, "id": s.id, "stream": stream_name,
                "squim_stoi": q["squim_stoi"],
                "squim_pesq": q["squim_pesq"],
                "squim_si_sdr_db": q["squim_si_sdr"],
                "n_chunks": q["n_chunks"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 3 — WER ablation
# ---------------------------------------------------------------------------


def summarize_layer3(scores: Iterable[ScoreCard]) -> pd.DataFrame:
    """One row per recording, columns per ablation mode + mixture baseline.

    Columns (all % WER):
      - ``mixture_orc``     ORC-WER of single-stream Whisper on the raw mixture
      - ``mixture_mimo``    MIMO-WER of the same (optimised reference interleaving;
                            MIMO <= ORC, more robust to faulty GT timestamps)
      - ``no_enh_cpwer``    cpWER of pipeline run with enhancement disabled
      - ``no_sep_cpwer``    cpWER of pipeline run with separation disabled
      - ``full_cpwer``      cpWER of the full pipeline
      - ``full_tcpwer``     tcpWER (time-constrained variant) of the full pipeline
      - ``ref_n_utts``      total reference utterances (A + B)

    The mixture → no_enh → no_sep → full progression is the ablation story.
    """
    rows = []
    for s in scores:
        if s.layer3 is None:
            continue
        l3 = s.layer3
        ref_n = sum(l3["ref_lengths"].values())
        modes = l3["modes"]
        def _pct(mode_d, key):
            return 100.0 * mode_d[key] if mode_d is not None else None
        row = {
            "dataset": s.dataset, "id": s.id,
            "ref_n_utts": ref_n,
            "mixture_orc": (
                100.0 * l3["mixture_orc"]["orc_wer"]
                if l3.get("mixture_orc") is not None else None
            ),
            "mixture_mimo": (
                100.0 * l3["mixture_mimo"]["mimo_wer"]
                if l3.get("mixture_mimo") is not None else None
            ),
            "no_enh_cpwer": _pct(modes.get("no_enh"), "cpwer"),
            "no_sep_cpwer": _pct(modes.get("no_sep"), "cpwer"),
            "full_cpwer":   _pct(modes.get("full"),   "cpwer"),
            "full_tcpwer":  _pct(modes.get("full"),   "tcpwer"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def inventory(scores: Iterable[ScoreCard]) -> pd.DataFrame:
    """What's available per recording — quick triage before running L1/L2/L3."""
    rows = []
    for s in scores:
        rec = s.recording
        rows.append({
            "dataset": s.dataset, "id": s.id,
            "ref_eaf":   rec.reference_eaf is not None,
            "ref_audio": rec.reference_audio is not None,
            "ref_diar":  rec.reference_diarization is not None,
            "ref_txt_A": "A" in rec.reference_transcripts,
            "ref_txt_B": "B" in rec.reference_transcripts,
            "pipe_full": rec.pipeline_dir is not None,
            "pipe_nosep": rec.pipeline_nosep_dir is not None,
            "pipe_noenh": rec.pipeline_noenh_dir is not None,
        })
    return pd.DataFrame(rows)
