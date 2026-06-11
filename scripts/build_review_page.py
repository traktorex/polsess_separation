#!/usr/bin/env python3
"""Build a self-contained HTML review page for the CLARIN pilot sweep.

Produces a single static site a supervisor can open with zero setup: a
config dropdown (best pipeline vs. ablations), a mixture-vs-pipeline toggle,
an audio player per stream, a diarization timeline with the overlap regions
highlighted, karaoke-highlighted transcripts, and per-recording + aggregate
WER / CER / ORC-WER tables.

All per-recording data is *inlined* into ``index.html`` as a JSON blob, so the
page works identically from ``file://`` (double-click) or over HTTP (hosted).
Only the audio sits beside it as files (transcoded to mono 16 kHz MP3 to keep
the bundle small enough to host).

Inputs are read from the eval tree built during the sweep::

    <eval_root>/<id>/
      <id>.wav                       # mixture
      annotation.eaf                 # hand-corrected GT (source of truth)
      sweep/<config>/
        stream_A.wav, stream_B.wav   # separated pipeline output
        transcript_A.txt, _B.txt     # per-speaker transcripts
        transcript_mixture.txt       # single-stream Whisper-on-mixture floor
        diarization.json, routing.json, metadata.json

Security: ``metadata.json`` carries the raw HF token. We read only the backend
names from it and never copy the file, so the token never reaches the bundle.

Usage::

    python scripts/build_review_page.py                 # 5 pilot fragments, default configs
    python scripts/build_review_page.py --out ~/clarin_review
    python scripts/build_review_page.py --configs frcrn_vad_strict enh_frcrn baseline nosep
    python scripts/build_review_page.py --audio-format wav   # pristine audio (big)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Make `asr_pipeline` importable when run as `python scripts/build_review_page.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_pipeline.eval.metrics import (  # noqa: E402
    _normalize_text,
    cp_cer_meeteval,
    cpwer_meeteval,
    mimo_cer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
    orc_wer_multistream,
)
from asr_pipeline.eval.transcript_parser import (  # noqa: E402
    Utterance,
    parse_eaf,
    parse_gt_txt,
)

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_EVAL_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
DEFAULT_OUT = Path("~/clarin_review").expanduser()

PILOT = [
    "065a9896__seg00",
    "150d1ccc__seg00",
    "ccfbb9db__seg00",
    "2bf3474d__seg00",
    "d1e63652__seg00",
]

# Recordings whose GT / sweep / mixture don't live under DEFAULT_EVAL_ROOT with
# the standard `<id>/annotation.eaf` layout. Each spec gives explicit paths;
# `label_map` renames GT tiers onto the pipeline's A/B (442dd69e's hand GT uses
# L/R channel tiers — L is speaker A, R is speaker B, per spk_to_label).
EXTRA_RECORDINGS = [
    {
        "id": "442dd69e",
        "rec_dir": "~/datasets/eval/clarin_gotowy/442dd69e",
        "gt_eaf": "~/datasets/clarin_gotowy/gotowy/true_transcripts/442dd69e.eaf",
        "mixture": "~/datasets/eval/clarin_gotowy/442dd69e/442dd69e.wav",
        "label_map": {"L": "A", "R": "B"},
    },
]

# Curated set that tells the story without 24 near-duplicate rows. The first
# entry is the default shown on load.
DEFAULT_CONFIGS = [
    "frcrn_vad_strict",
    "enh_frcrn",
    "enh_mossformer",
    "baseline",
    "enh_none",
    "nosep",
    "nosep_noenh",
    "asr_largev3",
]

CONFIG_LABELS = {
    "frcrn_vad_strict": "FRCRN + separation + VAD-strict  ★ best",
    "enh_frcrn": "FRCRN + separation (default VAD)",
    "enh_mossformer": "MossFormerGAN + separation",
    "baseline": "MP-SENet + separation  (shipped default)",
    "enh_none": "no enhancement + separation",
    "nosep": "FRCRN enhancement, NO separation  (ablation)",
    "nosep_noenh": "no enhancement, NO separation  (ablation)",
    "asr_largev3": "FRCRN + separation, Whisper large-v3",
}

# Description of what each layer of the eval measures — shown verbatim in the
# page so the reader needs no separate briefing.
METRIC_HELP = {
    "cpWER": "concatenated-permutation WER: best speaker matching, then word "
    "errors. The headline number — recognition AND who-said-what.",
    "tcpWER": "time-constrained cpWER (±5 s collar): a word only counts as "
    "correct if it lands near the right time.",
    "ORC-WER": "attribution-blind WER: each reference utterance is matched to "
    "whichever stream recognised it best. cpWER − ORC-WER = the cost of "
    "routing words to the wrong speaker.",
    "CER": "character error rate under the same speaker matching as cpWER.",
    "floor": "ORC-WER of one Whisper transcript on the raw, unseparated "
    "mixture — a no-pipeline baseline (fixes the reference merge by time).",
    "MIMO": "MIMO-WER on the raw-mixture transcript: like ORC but optimises "
    "how the two reference speakers interleave into the single stream. The "
    "tighter, more principled no-pipeline floor (MIMO ≤ ORC) and more robust "
    "to imperfect GT timestamps.",
    "MIMO-CER": "Character error rate of the raw-mixture transcript with the "
    "reference merged in MIMO's order (vs the time/ORC-ordered CER) — the CER "
    "analog of the MIMO floor.",
}


# --------------------------------------------------------------------------- #
# Reading the eval tree
# --------------------------------------------------------------------------- #


def _utts_to_rows(utts: list[Utterance]) -> list[list]:
    """Utterance list → compact ``[[start, end, text], ...]`` for JSON."""
    return [[round(u.start, 2), round(u.end, 2), u.text] for u in utts]


def _nonempty(by_spk: dict[str, list[Utterance]]) -> dict[str, list[Utterance]]:
    return {k: v for k, v in by_spk.items() if v}


def _peaks(path: Path, nbuckets: int = 400) -> list[int]:
    """Downsampled peak envelope of a mono wav → ints 0..100 (per-stream max).

    One value per time bucket = max |sample| in that bucket, scaled to the
    stream's own peak. Cheap to embed, enough to draw a review waveform.
    """
    import numpy as np
    import soundfile as sf

    x, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0:
        return [0] * nbuckets
    ax = np.abs(x)
    peak = float(ax.max()) or 1.0
    edges = (np.arange(nbuckets + 1) * n / nbuckets).astype(int)
    return [
        int(round(float(ax[edges[i]:edges[i + 1]].max()) / peak * 100))
        if edges[i + 1] > edges[i] else 0
        for i in range(nbuckets)
    ]


def _read_config(cdir: Path, gt: dict[str, list[Utterance]], fid: str) -> dict | None:
    """Read one sweep-config directory → its JSON payload + scores, or None."""
    sa, sb = cdir / "stream_A.wav", cdir / "stream_B.wav"
    ta, tb = cdir / "transcript_A.txt", cdir / "transcript_B.txt"
    if not (sa.exists() and sb.exists() and ta.exists() and tb.exists()):
        return None

    hyp = _nonempty({"A": parse_gt_txt(ta), "B": parse_gt_txt(tb)})

    cp = cpwer_meeteval(gt, hyp, fid)
    orc = orc_wer_multistream(gt, hyp, fid)
    cer = cp_cer_meeteval(gt, hyp, fid)

    # Map pyannote SPEAKER_xx turns to the A/B labels via metadata.
    diar_rows: list[list] = []
    overlaps: list[list] = []
    backends = ""
    meta_path = cdir / "metadata.json"
    spk_to_label: dict[str, str] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spk_to_label = meta.get("spk_to_label", {})
        cfg = meta.get("config", {})
        enh = cfg.get("enhancement", {})
        sep = cfg.get("separation", {})
        psp = cfg.get("post_separation_processing", {})
        asr = cfg.get("transcription", {})
        backends = " · ".join([
            f"enh={enh.get('backend') if enh.get('enabled') else 'off'}",
            f"sep={'on' if sep.get('enabled') else 'off'}",
            f"bwe={psp.get('backend', '?')}",
            f"asr={asr.get('backend', '?')} {asr.get('model_name', '')}".strip(),
        ])

    diar_path = cdir / "diarization.json"
    if diar_path.exists():
        for t in json.loads(diar_path.read_text(encoding="utf-8")).get("turns", []):
            lbl = spk_to_label.get(t["speaker"], t["speaker"])
            diar_rows.append([lbl, round(t["start"], 2), round(t["end"], 2)])

    routing_path = cdir / "routing.json"
    if routing_path.exists():
        for r in json.loads(routing_path.read_text(encoding="utf-8")).get(
            "overlap_regions", []
        ):
            overlaps.append([round(r["start"], 2), round(r["end"], 2)])

    return {
        "transcript": {k: _utts_to_rows(hyp.get(k, [])) for k in ("A", "B")},
        "diar": diar_rows,
        "overlaps": overlaps,
        "backends": backends,
        "waveform": {"A": _peaks(sa), "B": _peaks(sb)},
        "metrics": {
            "cpwer": round(cp["cpwer"] * 100, 1),
            "tcpwer": round(cp["tcpwer"] * 100, 1),
            "orcwer": round(orc["orc_wer"] * 100, 1),
            "cer": round(cer["cer"] * 100, 1),
            "cp_err": cp["cp_errors"], "cp_len": cp["cp_length"],
            "tcp_err": cp["tcp_errors"], "tcp_len": cp["tcp_length"],
            "orc_err": orc["errors"], "orc_len": orc["length"],
            "cer_err": cer["errors"], "cer_len": cer["length"],
        },
    }


def _mixture_metrics(
    gt: dict[str, list[Utterance]], mix_utts: list[Utterance], fid: str
) -> dict:
    """Single-stream floor: ORC-WER + a single-stream CER on the mixture."""
    from rapidfuzz.distance import Levenshtein

    orc = orc_wer_meeteval(gt, mix_utts, fid)
    mimo = mimo_wer_meeteval(gt, mix_utts, fid)
    # Single-stream CER: char-level edit distance can't reorder, so the
    # reference must be in the same temporal order as the single hypothesis
    # stream — interleave both speakers' GT by start time (mix_utts already
    # come in time order from Whisper).
    ref_utts = sorted((u for spk in gt for u in gt[spk]), key=lambda u: u.start)
    ref_all = _normalize_text(" ".join(u.text for u in ref_utts))
    hyp_all = _normalize_text(" ".join(u.text for u in mix_utts))
    cer_err = Levenshtein.distance(ref_all, hyp_all)
    cer_len = max(len(ref_all), 1)
    # MIMO-flavoured CER: same char distance, but reference merged by MIMO's
    # optimal interleaving instead of by timestamp — the CER analog of the
    # MIMO-WER floor (not penalised by overlap interleaving order).
    cer_mimo = mimo_cer_meeteval(gt, mix_utts, fid)
    return {
        "orcwer": round(orc["orc_wer"] * 100, 1),
        "orc_err": orc["errors"], "orc_len": orc["length"],
        "mimower": round(mimo["mimo_wer"] * 100, 1),
        "mimo_err": mimo["errors"], "mimo_len": mimo["length"],
        # `cer` = time-ordered (ORC-flavoured) reference merge; `cer_mimo` = MIMO merge.
        "cer": round(cer_err / cer_len * 100, 1),
        "cer_err": cer_err, "cer_len": cer_len,
        "cer_mimo": round(cer_mimo["cer"] * 100, 1),
        "cer_mimo_err": cer_mimo["errors"], "cer_mimo_len": cer_mimo["length"],
    }


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def _encode_audio(src: Path, dst: Path, fmt: str, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    if fmt == "wav":
        shutil.copyfile(src, dst)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst)],
        check=True,
    )


def _encode_both(a_src: Path, b_src: Path, dst: Path, fmt: str, force: bool) -> None:
    """Stereo file with stream A on the LEFT channel, B on the RIGHT.

    Pre-rendering the stereo mix lets the page pan A/B with a plain native
    ``<audio>`` element — Web Audio's MediaElementSource is muted for
    ``file://`` origins, so routing through it would silence local playback.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(a_src), "-i", str(b_src),
           "-filter_complex",
           "[0:a][1:a]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]",
           "-map", "[a]", "-ar", "16000"]
    if fmt != "wav":
        cmd += ["-b:a", "96k"]   # stereo → a touch more than the 64k mono
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build(
    eval_root: Path,
    out: Path,
    fragments: list[str],
    configs: list[str],
    audio_fmt: str,
    force: bool,
    extras: list[dict] | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ext = "wav" if audio_fmt == "wav" else "mp3"

    # Uniform recording specs: pilot fragments under eval_root + explicit-path
    # extras (different eval root / external GT EAF / tier relabel).
    specs: list[dict] = [
        {"id": fid, "rec_dir": eval_root / fid,
         "gt_eaf": eval_root / fid / "annotation.eaf",
         "mixture": eval_root / fid / f"{fid}.wav", "label_map": None}
        for fid in fragments
    ]
    for ex in (EXTRA_RECORDINGS if extras is None else extras):
        specs.append({
            "id": ex["id"],
            "rec_dir": Path(ex["rec_dir"]).expanduser(),
            "gt_eaf": Path(ex["gt_eaf"]).expanduser(),
            "mixture": Path(ex["mixture"]).expanduser(),
            "label_map": ex.get("label_map"),
        })

    frag_payloads = []
    present_configs: list[str] = []  # union of configs actually found

    for spec in specs:
        fid, rec_dir = spec["id"], spec["rec_dir"]
        mixture_wav, eaf, label_map = spec["mixture"], spec["gt_eaf"], spec["label_map"]
        if not (mixture_wav.exists() and eaf.exists()):
            print(f"  ! skip {fid}: missing mixture or GT EAF")
            continue

        gt = _nonempty(parse_eaf(eaf))
        if label_map:
            gt = {label_map.get(k, k): v for k, v in gt.items()}
        if not gt:
            print(f"  ! skip {fid}: empty GT")
            continue

        # Mixture transcript + floor come from the default (large-v2) config, so
        # the no-pipeline baseline is fixed across the dropdown. Fall back to the
        # first config actually present if the default wasn't run for this rec.
        default_cfg_dir = next(
            (rec_dir / "sweep" / c for c in configs
             if (rec_dir / "sweep" / c / "metadata.json").exists()),
            rec_dir / "sweep" / configs[0],
        )
        mix_txt = default_cfg_dir / "transcript_mixture.txt"
        mix_utts = parse_gt_txt(mix_txt) if mix_txt.exists() else []

        # Duration from the default config's metadata; fall back to GT extent.
        duration = 0.0
        meta_path = default_cfg_dir / "metadata.json"
        if meta_path.exists():
            duration = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "total_duration_s", 0.0
            )
        if not duration:
            duration = max(
                (u.end for spk in gt for u in gt[spk]), default=0.0
            )

        _encode_audio(
            mixture_wav, out / "audio" / fid / f"mixture.{ext}", audio_fmt, force
        )

        cfg_payloads: dict[str, dict] = {}
        for c in configs:
            cdir = rec_dir / "sweep" / c
            payload = _read_config(cdir, gt, fid)
            if payload is None:
                continue
            for lbl, stream in (("A", "stream_A.wav"), ("B", "stream_B.wav")):
                _encode_audio(
                    cdir / stream,
                    out / "audio" / fid / f"{c}__{lbl}.{ext}",
                    audio_fmt,
                    force,
                )
            _encode_both(
                cdir / "stream_A.wav", cdir / "stream_B.wav",
                out / "audio" / fid / f"{c}__both.{ext}", audio_fmt, force,
            )
            payload["audio"] = {
                "A": f"audio/{fid}/{c}__A.{ext}",
                "B": f"audio/{fid}/{c}__B.{ext}",
                "both": f"audio/{fid}/{c}__both.{ext}",
            }
            cfg_payloads[c] = payload
            if c not in present_configs:
                present_configs.append(c)

        frag_payloads.append({
            "id": fid,
            "gt_source": str(eaf),
            "duration": round(duration, 2),
            "mixture_audio": f"audio/{fid}/mixture.{ext}",
            "gt": {k: _utts_to_rows(gt.get(k, [])) for k in ("A", "B")},
            "mixture_transcript": _utts_to_rows(mix_utts),
            "mixture_metrics": _mixture_metrics(gt, mix_utts, fid),
            "configs": cfg_payloads,
        })
        print(f"  ✓ {fid}: {len(cfg_payloads)} configs, {duration:.0f}s")

    # config_order/labels/backends scoped to what we actually have.
    ordered = [c for c in configs if c in present_configs]
    backends = {}
    for f in frag_payloads:
        for c, p in f["configs"].items():
            backends.setdefault(c, p.get("backends", ""))

    data = {
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "default_config": configs[0] if configs[0] in present_configs else (ordered[0] if ordered else ""),
        "config_order": ordered,
        "config_labels": {c: CONFIG_LABELS.get(c, c) for c in ordered},
        "config_backends": backends,
        "metric_help": METRIC_HELP,
        "fragments": frag_payloads,
    }

    (out / "index.html").write_text(
        _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False)),
        encoding="utf-8",
    )
    return data


# --------------------------------------------------------------------------- #
# HTML template (single file; JSON injected at __DATA_JSON__)
# --------------------------------------------------------------------------- #

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLARIN pilot — separation-pipeline review</title>
<style>
  :root{
    --A:#1f6feb; --B:#e36209; --ovl:rgba(220,38,38,.16); --ovl-line:rgba(220,38,38,.5);
    --bg:#fff; --panel:#f7f8fa; --line:#e1e4e8; --ink:#1b1f24; --mut:#6b7077;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
  header{padding:16px 22px;border-bottom:1px solid var(--line)}
  header h1{margin:0 0 4px;font-size:18px}
  header p{margin:2px 0;color:var(--mut);max-width:80ch}
  main{display:flex;align-items:flex-start;gap:0}
  aside{width:300px;flex:0 0 300px;border-right:1px solid var(--line);padding:14px;height:calc(100vh - 86px);overflow:auto;position:sticky;top:0}
  section#panel{flex:1;padding:18px 22px;min-width:0}
  h2{font-size:15px;margin:18px 0 8px}
  h3{font-size:13px;margin:14px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
  ul#fraglist{list-style:none;margin:8px 0 0;padding:0}
  ul#fraglist li{padding:8px 10px;border-radius:7px;cursor:pointer;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
  ul#fraglist li:hover{background:#eef1f4}
  ul#fraglist li.sel{background:#1f6feb;color:#fff}
  table.agg{border-collapse:collapse;width:100%;font-size:11px}
  table.agg th,table.agg td{padding:4px 4px;text-align:right;border-bottom:1px solid var(--line)}
  table.agg td:first-child{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px}
  table.agg th:first-child,table.agg td:first-child{text-align:left}
  table.agg tr.best td{font-weight:700;background:#eaf2ff}
  table.agg tr.floor td{color:var(--mut);font-style:italic}
  .controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:6px 0 14px}
  select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:#fff;max-width:420px}
  .toggle button{font:inherit;padding:7px 14px;border:1px solid var(--line);background:#fff;cursor:pointer}
  .toggle button:first-child{border-radius:7px 0 0 7px}
  .toggle button:last-child{border-radius:0 7px 7px 0;border-left:0}
  .toggle button.on{background:#1f6feb;color:#fff;border-color:#1f6feb}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 14px}
  .chip{padding:6px 11px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-size:12.5px;cursor:help}
  .chip b{font-size:15px}
  .chip.floor{opacity:.7}
  .backends{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--mut);margin:-8px 0 12px}
  /* timeline */
  .tl-wrap{border:1px solid var(--line);border-radius:9px;overflow:hidden;margin-bottom:8px}
  .tl-row{display:flex;align-items:stretch}
  .tl-gutter{width:74px;flex:0 0 74px;background:var(--panel);border-right:1px solid var(--line)}
  .tl-gutter div{height:26px;display:flex;align-items:center;padding-left:8px;font-size:11px;color:var(--mut);border-bottom:1px solid var(--line)}
  .tl-svgwrap{flex:1;min-width:0}
  svg.tl{display:block;width:100%;height:104px;cursor:pointer}
  svg.ruler{display:block;width:100%;height:18px}
  .lane-A{fill:var(--A)} .lane-B{fill:var(--B)}
  .ovl{fill:var(--ovl)} .ovl-edge{stroke:var(--ovl-line);stroke-width:.5}
  .playhead{stroke:#111;stroke-width:1.5}
  .legend{display:flex;gap:16px;font-size:11.5px;color:var(--mut);margin:2px 0 16px;flex-wrap:wrap}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
  /* players + transcripts */
  .players{display:grid;gap:14px;margin:6px 0 16px}
  .players.two{grid-template-columns:1fr 1fr}
  .pl h4{margin:0 0 6px;font-size:12.5px}
  .pl audio{width:100%}
  .cols{display:grid;gap:16px}
  .cols.two{grid-template-columns:1fr 1fr}
  .tcol{border:1px solid var(--line);border-radius:9px;overflow:hidden}
  .tcol .hd{padding:7px 11px;font-weight:600;font-size:12.5px;color:#fff}
  .tcol.A .hd{background:var(--A)} .tcol.B .hd{background:var(--B)}
  .tcol.M .hd{background:#444}
  .tcol .body{max-height:360px;overflow:auto;padding:4px;position:relative}
  .seg{padding:4px 8px;border-radius:6px;cursor:pointer;display:flex;gap:9px;font-size:13px}
  .seg:hover{background:#eef1f4}
  .seg.cur{background:#fff3cd;outline:1px solid #ffe08a}
  .seg .t{color:var(--mut);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;flex:0 0 76px;white-space:nowrap}
  /* stream selector + transport */
  .streamsel{display:inline-flex;margin:2px 0 12px}
  .streamsel button{font:inherit;padding:7px 14px;border:1px solid var(--line);background:#fff;cursor:pointer}
  .streamsel button:first-child{border-radius:7px 0 0 7px}
  .streamsel button:last-child{border-radius:0 7px 7px 0}
  .streamsel button:not(:first-child){border-left:0}
  .streamsel button.on{background:#1f6feb;color:#fff;border-color:#1f6feb}
  .transport{display:flex;align-items:center;gap:12px;margin:0 0 16px;max-width:760px}
  .playbtn{font-size:15px;width:46px;height:36px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}
  .seek{flex:1}
  .tlbl{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:var(--mut);white-space:nowrap}
  .gt h3{margin-top:22px}
  .note{font-size:12px;color:var(--mut);margin:4px 0 0}
  /* collapsible config table + waveforms */
  details.aggwrap > summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--ink);list-style:none;padding:2px 0}
  details.aggwrap > summary::-webkit-details-marker{display:none}
  details.aggwrap > summary::before{content:"\25B8 ";color:var(--mut)}
  details.aggwrap[open] > summary::before{content:"\25BE "}
  details.wfwrap{margin:0 0 16px}
  details.wfwrap > summary{cursor:pointer;font-size:12px;color:var(--mut);padding:2px 0;list-style:none}
  details.wfwrap > summary::-webkit-details-marker{display:none}
  details.wfwrap > summary::before{content:"\25B8 "}
  details.wfwrap[open] > summary::before{content:"\25BE "}
  details.wfwrap[open] > summary{margin-bottom:6px}
  svg.wf{display:block;width:100%;height:92px;cursor:pointer}
  svg.wf .wa{stroke:var(--A)} svg.wf .wb{stroke:var(--B)}
  .wf-gut div{height:46px}
</style>
</head>
<body>
<header>
  <h1>CLARIN — separation pipeline, manual review</h1>
  <p class="note" id="genline"></p>
</header>
<main>
  <aside>
    <details class="aggwrap">
      <summary>Config comparison (5 fragments, micro-avg)</summary>
      <table class="agg" id="aggtable"></table>
    </details>
    <h3 style="margin-top:22px">Recordings</h3>
    <ul id="fraglist"></ul>
  </aside>
  <section id="panel"></section>
</main>

<script>const DATA = __DATA_JSON__;</script>
<script>
"use strict";
const $ = (s, r=document) => r.querySelector(s);
const el = (t, props={}, kids=[]) => {
  const n = document.createElement(t);
  for (const [k,v] of Object.entries(props)){
    if (k==="class") n.className=v;
    else if (k==="html") n.innerHTML=v;
    else if (k==="text") n.textContent=v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  (Array.isArray(kids)?kids:[kids]).forEach(c=>c&&n.appendChild(typeof c==="string"?document.createTextNode(c):c));
  return n;
};
const fmt = s => { s=Math.max(0,s); const m=Math.floor(s/60), x=(s%60); return m+":"+(x<10?"0":"")+x.toFixed(1); };

let state = { fid: DATA.fragments[0]?.id, config: DATA.default_config, mode: "pipeline", streamMode: "both" };
let RAF = null, audios = [], activeAudio = null;

$("#genline").textContent = "Generated " + DATA.generated + " · scores vs. corrected GT · mixture floor = Whisper large-v2 on raw mixture.";

/* ---- aggregate (micro-averaged across fragments) ---- */
function aggregate(){
  const rows = [];
  for (const c of DATA.config_order){
    let cpE=0,cpL=0,orcE=0,orcL=0,cerE=0,cerL=0,n=0;
    for (const f of DATA.fragments){
      const m = f.configs[c]?.metrics; if(!m) continue;
      cpE+=m.cp_err; cpL+=m.cp_len; orcE+=m.orc_err; orcL+=m.orc_len; cerE+=m.cer_err; cerL+=m.cer_len; n++;
    }
    if(!n) continue;
    rows.push({c, label:DATA.config_labels[c], cp:100*cpE/cpL, orc:100*orcE/orcL, cer:100*cerE/cerL});
  }
  rows.sort((a,b)=>a.cp-b.cp);
  // mixture floor (ORC + MIMO single-stream)
  let mO=0,mOl=0,mMi=0,mMil=0,mC=0,mCl=0,mMiC=0,mMiCl=0;
  for (const f of DATA.fragments){ const m=f.mixture_metrics; mO+=m.orc_err;mOl+=m.orc_len;mMi+=m.mimo_err;mMil+=m.mimo_len;mC+=m.cer_err;mCl+=m.cer_len;mMiC+=m.cer_mimo_err;mMiCl+=m.cer_mimo_len; }
  const t = $("#aggtable");
  t.innerHTML="";
  const hdr = el("tr",{},[el("th",{text:"config"}),el("th",{text:"cpWER"}),el("th",{text:"ORC"}),el("th",{text:"MIMO"}),el("th",{text:"CER"})]);
  hdr.querySelectorAll("th").forEach(th=>{ const k={cpWER:"cpWER",ORC:"ORC-WER",MIMO:"MIMO",CER:"CER"}[th.textContent]; if(k&&DATA.metric_help[k]) th.title=DATA.metric_help[k]; });
  t.appendChild(hdr);
  rows.forEach((r,i)=>{
    const tr = el("tr",{class:i===0?"best":""},[
      el("td",{text:r.c}),
      el("td",{text:r.cp.toFixed(1)}),
      el("td",{text:r.orc.toFixed(1)}),
      el("td",{text:"—"}),                       // MIMO is single-stream only
      el("td",{text:r.cer.toFixed(1)}),
    ]);
    tr.title = r.label;
    tr.addEventListener("click",()=>{ state.config=r.c; render(); });
    tr.style.cursor="pointer";
    t.appendChild(tr);
  });
  const floorCer = el("td",{text:(100*mMiC/mMiCl).toFixed(1)});
  floorCer.title = "MIMO-merge CER · time-ordered CER: "+(100*mC/mCl).toFixed(1);
  t.appendChild(el("tr",{class:"floor"},[
    el("td",{text:"mixture (floor)"}),
    el("td",{text:"—"}),
    el("td",{text:(100*mO/mOl).toFixed(1)}),
    el("td",{text:(100*mMi/mMil).toFixed(1)}),
    floorCer,
  ]));
}

/* ---- fragment list ---- */
function fraglist(){
  const ul = $("#fraglist"); ul.innerHTML="";
  DATA.fragments.forEach(f=>{
    const li = el("li",{class:f.id===state.fid?"sel":"",text:f.id});
    li.addEventListener("click",()=>{ state.fid=f.id; render(); });
    ul.appendChild(li);
  });
}

/* ---- timeline ---- */
const NS="http://www.w3.org/2000/svg";
function svgEl(t,a){ const n=document.createElementNS(NS,t); for(const k in a) n.setAttribute(k,a[k]); return n; }
function buildTimeline(frag, cfg){
  const D = frag.duration || 1;
  const laneH=26, W=1000;
  const lanes = [
    {key:"gtA", cls:"lane-A", segs:frag.gt.A},
    {key:"gtB", cls:"lane-B", segs:frag.gt.B},
    {key:"pA",  cls:"lane-A", segs:(cfg?cfg.diar.filter(d=>d[0]==="A").map(d=>[d[1],d[2]]):[])},
    {key:"pB",  cls:"lane-B", segs:(cfg?cfg.diar.filter(d=>d[0]==="B").map(d=>[d[1],d[2]]):[])},
  ];
  const H = lanes.length*laneH;
  const svg = svgEl("svg",{class:"tl", viewBox:`0 0 ${W} ${H}`, preserveAspectRatio:"none"});
  // overlap bands (behind)
  (cfg?cfg.overlaps:[]).forEach(([s,e])=>{
    svg.appendChild(svgEl("rect",{class:"ovl", x:s/D*W, y:0, width:Math.max((e-s)/D*W,0.6), height:H}));
  });
  // lane separators + segments
  lanes.forEach((ln,i)=>{
    const y=i*laneH;
    if(i>0) svg.appendChild(svgEl("line",{x1:0,y1:y,x2:W,y2:y,stroke:"#e1e4e8","stroke-width":.5}));
    ln.segs.forEach(([s,e])=>{
      svg.appendChild(svgEl("rect",{class:ln.cls, x:s/D*W, y:y+5, width:Math.max((e-s)/D*W,0.8), height:laneH-10, rx:2, opacity:.85}));
    });
  });
  const ph = svgEl("line",{class:"playhead", x1:0,y1:0,x2:0,y2:H}); svg.appendChild(ph);
  svg.addEventListener("click",ev=>{
    const r=svg.getBoundingClientRect();
    const t=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width))*D;
    if(activeAudio){ activeAudio.currentTime=t; }
  });
  // ruler
  const ruler = svgEl("svg",{class:"ruler", viewBox:`0 0 ${W} 18`, preserveAspectRatio:"none"});
  const step = D>240?60:(D>120?30:(D>40?10:5));
  for(let t=0;t<=D;t+=step){
    const x=t/D*W;
    ruler.appendChild(svgEl("line",{x1:x,y1:0,x2:x,y2:5,stroke:"#aaa","stroke-width":.6}));
    const tx=svgEl("text",{x:Math.min(x+2,W-26),y:14,"font-size":10,fill:"#888"}); tx.textContent=fmt(t);
    ruler.appendChild(tx);
  }
  // gutter labels
  const gutter = el("div",{class:"tl-gutter"});
  ["GT · A","GT · B","Pipe · A","Pipe · B"].forEach(l=>gutter.appendChild(el("div",{text:l})));
  const wrap = el("div",{class:"tl-wrap"},[
    el("div",{class:"tl-row"},[gutter, el("div",{class:"tl-svgwrap"},[svg])]),
  ]);
  return {wrap, ruler, ph, D};
}

/* ---- transcript column ---- */
function tcol(label, cls, rows, onSeek){
  const body = el("div",{class:"body"});
  const segs = rows.map(([s,e,txt])=>{
    const seg = el("div",{class:"seg"},[el("span",{class:"t",text:fmt(s)}), el("span",{text:txt})]);
    seg.addEventListener("click",()=>onSeek(s));
    body.appendChild(seg);
    return seg;
  });
  const col = el("div",{class:"tcol "+cls},[el("div",{class:"hd",text:label}), body]);
  return {col, body, segs, rows, lastIdx:-1};
}

/* Karaoke highlight. Only touches the DOM when the current line *changes*,
   so it never fights the user's manual scroll between line changes. */
function highlight(track, ct){
  let curIdx=-1;
  for(let i=0;i<track.rows.length;i++){ const r=track.rows[i]; if(ct>=r[0]&&ct<=r[1]){curIdx=i;break;} }
  if(curIdx===track.lastIdx) return;
  track.segs.forEach((s,i)=>s.classList.toggle("cur",i===curIdx));
  if(curIdx>=0){
    const s=track.segs[curIdx], b=track.body, top=s.offsetTop, bot=top+s.offsetHeight;
    if(top<b.scrollTop || bot>b.scrollTop+b.clientHeight)
      b.scrollTop=top-b.clientHeight/2+s.offsetHeight/2;
  }
  track.lastIdx=curIdx;
}

/* ---- chips ---- */
function chip(label, val, helpKey, cls=""){
  const c = el("div",{class:"chip "+cls},[document.createTextNode(label+" "), el("b",{text:val})]);
  if(DATA.metric_help[helpKey]) c.title = label+": "+DATA.metric_help[helpKey];
  return c;
}

/* ---- main render ---- */
function render(){
  if(RAF) cancelAnimationFrame(RAF);
  audios=[]; activeAudio=null;
  fraglist();
  const frag = DATA.fragments.find(f=>f.id===state.fid);
  const panel = $("#panel"); panel.innerHTML="";
  if(!frag){ panel.textContent="no data"; return; }
  // fall back if the selected config wasn't run for this recording
  if(!frag.configs[state.config]){
    const avail = DATA.config_order.find(c=>frag.configs[c]);
    if(avail) state.config = avail;
  }
  const cfg = frag.configs[state.config];

  // controls — configs not run for this recording are shown disabled
  const sel = el("select",{});
  DATA.config_order.forEach(c=>{
    const has = !!frag.configs[c];
    const o=el("option",{value:c,text:DATA.config_labels[c]+(has?"":"  — not run")});
    if(!has) o.disabled=true;
    if(c===state.config) o.selected=true;
    sel.appendChild(o);
  });
  sel.addEventListener("change",()=>{ state.config=sel.value; render(); });
  const tog = el("div",{class:"toggle"},[
    el("button",{class:state.mode==="mixture"?"on":"",text:"Mixture (no pipeline)",onclick:()=>{state.mode="mixture";render();}}),
    el("button",{class:state.mode==="pipeline"?"on":"",text:"Pipeline",onclick:()=>{state.mode="pipeline";render();}}),
  ]);
  const gtShort = frag.gt_source.split("/").slice(-2).join("/");
  panel.appendChild(el("div",{class:"controls"},[
    el("div",{},[
      el("h2",{text:frag.id}),
      el("div",{class:"note",text:frag.duration.toFixed(0)+" s · GT: "+gtShort
        +" ("+(frag.gt.A.length+frag.gt.B.length)+" utts)", title:"scored against "+frag.gt_source}),
    ]),
    el("div",{},[el("div",{class:"note",text:"pipeline config"}), sel]),
    tog,
  ]));
  if(cfg && DATA.config_backends[state.config])
    panel.appendChild(el("div",{class:"backends",text:DATA.config_backends[state.config]}));

  // chips
  const chips = el("div",{class:"chips"});
  if(state.mode==="pipeline" && cfg){
    const m=cfg.metrics;
    chips.appendChild(chip("cpWER", m.cpwer+"%", "cpWER"));
    chips.appendChild(chip("tcpWER", m.tcpwer+"%", "tcpWER"));
    chips.appendChild(chip("ORC-WER", m.orcwer+"%", "ORC-WER"));
    chips.appendChild(chip("attr-gap", (m.cpwer-m.orcwer).toFixed(1)+" pt", "ORC-WER"));
    chips.appendChild(chip("CER", m.cer+"%", "CER"));
    chips.appendChild(chip("floor MIMO", frag.mixture_metrics.mimower+"%", "MIMO", "floor"));
    chips.appendChild(chip("floor ORC", frag.mixture_metrics.orcwer+"%", "floor", "floor"));
  } else {
    const m=frag.mixture_metrics;
    chips.appendChild(chip("MIMO-WER (floor)", m.mimower+"%", "MIMO"));
    chips.appendChild(chip("ORC-WER (floor)", m.orcwer+"%", "floor"));
    chips.appendChild(chip("CER (MIMO)", m.cer_mimo+"%", "MIMO-CER"));
    chips.appendChild(chip("CER (time-ord)", m.cer+"%", "CER"));
  }
  panel.appendChild(chips);

  // timeline
  const tl = buildTimeline(frag, cfg);
  panel.appendChild(tl.wrap);
  panel.appendChild((function(){ const w=el("div",{class:"tl-wrap"},[el("div",{class:"tl-row"},[el("div",{class:"tl-gutter",html:"<div style='border:0'></div>"}), el("div",{class:"tl-svgwrap"},[tl.ruler])])]); return w; })());
  panel.appendChild(el("div",{class:"legend"},[
    el("span",{},[el("span",{class:"sw",style:"background:var(--A)"}),"speaker A"]),
    el("span",{},[el("span",{class:"sw",style:"background:var(--B)"}),"speaker B"]),
    el("span",{},[el("span",{class:"sw",style:"background:var(--ovl)"}),"overlap region (separator ran here)"]),
  ]));

  let trackers = [], transport = null, seekPlay = ()=>{}, wfPlayhead = null;

  // waveforms — collapsible, off by default, directly under the diarization
  if(state.mode==="pipeline" && cfg){
    const W=1000, laneH=46, H=laneH*2;
    const svg=svgEl("svg",{class:"wf",viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:"none"});
    [["A","wa",0],["B","wb",laneH]].forEach(([k,cls,y0])=>{
      const peaks=cfg.waveform[k]||[], N=peaks.length||1, mid=y0+laneH/2, half=laneH/2-3;
      svg.appendChild(svgEl("line",{x1:0,y1:mid,x2:W,y2:mid,stroke:"#ddd","stroke-width":.4}));
      const sw=Math.max(W/N*0.8,0.6);
      peaks.forEach((v,i)=>{ const x=i/N*W, h=v/100*half; svg.appendChild(svgEl("line",{class:cls,x1:x,y1:mid-h,x2:x,y2:mid+h,"stroke-width":sw})); });
    });
    svg.appendChild(svgEl("line",{x1:0,y1:laneH,x2:W,y2:laneH,stroke:"#e1e4e8","stroke-width":.5}));
    wfPlayhead=svgEl("line",{class:"playhead",x1:0,y1:0,x2:0,y2:H}); svg.appendChild(wfPlayhead);
    svg.addEventListener("click",ev=>{ const r=svg.getBoundingClientRect(); seekPlay(Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width))*(frag.duration||1)); });
    const wgut=el("div",{class:"tl-gutter wf-gut"},[el("div",{text:"wave A"}),el("div",{text:"wave B"})]);
    const wrap=el("div",{class:"tl-wrap"},[el("div",{class:"tl-row"},[wgut, el("div",{class:"tl-svgwrap"},[svg])])]);
    panel.appendChild(el("details",{class:"wfwrap"},[el("summary",{text:"stream waveforms (A / B)"}), wrap]));
  }

  if(state.mode==="pipeline" && cfg){
    // Native elements only. Web Audio's MediaElementSource is muted for
    // file:// origins, so the "Both" stereo file (A→left, B→right) is
    // pre-rendered at build time and we just swap which native element is
    // active. Identical behaviour from file:// and over HTTP.
    const elA = el("audio",{preload:"metadata",src:cfg.audio.A});
    const elB = el("audio",{preload:"metadata",src:cfg.audio.B});
    const elBoth = el("audio",{preload:"metadata",src:cfg.audio.both});
    [elA,elB,elBoth].forEach(a=>{ a.style.display="none"; panel.appendChild(a); });
    const pick = ()=> state.streamMode==="A" ? elA : (state.streamMode==="B" ? elB : elBoth);
    audios=[elA,elB,elBoth]; activeAudio=pick();

    // selector — swaps the active element, preserving position + play state
    const selWrap = el("div",{class:"streamsel"});
    [["A","A only"],["both","Both  (A → L · B → R)"],["B","B only"]].forEach(([v,lab])=>{
      const b=el("button",{class:state.streamMode===v?"on":"",text:lab,onclick:()=>{
        if(v===state.streamMode) return;
        const prev=activeAudio, wasPlaying=!prev.paused, t=prev.currentTime;
        prev.pause();
        state.streamMode=v; activeAudio=pick(); activeAudio.currentTime=t;
        if(wasPlaying) activeAudio.play();
        selWrap.querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on");
      }});
      selWrap.appendChild(b);
    });
    panel.appendChild(el("div",{},[el("h3",{text:"playback"}), selWrap]));

    // single transport, bound to whichever element is active
    const playBtn = el("button",{class:"playbtn",html:"&#9654;"});
    const seek = el("input",{type:"range",min:"0",max:String(frag.duration),step:"0.01",value:"0",class:"seek"});
    const tlbl = el("span",{class:"tlbl",text:"0:00 / "+fmt(frag.duration)});
    let scrubbing=false;
    const setIcon = ()=>{ playBtn.innerHTML = activeAudio.paused ? "&#9654;" : "&#10073;&#10073;"; };
    playBtn.addEventListener("click",()=>{ activeAudio.paused ? activeAudio.play() : activeAudio.pause(); });
    audios.forEach(a=>{ a.addEventListener("play",setIcon); a.addEventListener("pause",setIcon); });
    seek.addEventListener("pointerdown",()=>scrubbing=true);
    seek.addEventListener("pointerup",()=>scrubbing=false);
    seek.addEventListener("input",()=>{ activeAudio.currentTime=parseFloat(seek.value); });
    panel.appendChild(el("div",{class:"transport"},[playBtn,seek,tlbl]));
    transport = {seek,tlbl,isScrub:()=>scrubbing};

    seekPlay = t=>{ activeAudio.currentTime=t; activeAudio.play(); };

    const cA = tcol("Speaker A — "+state.config,"A",cfg.transcript.A,seekPlay);
    const cB = tcol("Speaker B — "+state.config,"B",cfg.transcript.B,seekPlay);
    panel.appendChild(el("div",{class:"cols two"},[cA.col,cB.col]));
    trackers=[cA,cB];
  } else {
    const aM = el("audio",{controls:"",preload:"none",src:frag.mixture_audio});
    audios=[aM]; activeAudio=aM;
    panel.appendChild(el("div",{class:"players"},[el("div",{class:"pl"},[el("h4",{html:"&#9654; raw mixture"}),aM])]));
    seekPlay = t=>{ aM.currentTime=t; aM.play(); };
    const cM = tcol("Mixture transcript (single stream)","M",frag.mixture_transcript,seekPlay);
    panel.appendChild(el("div",{class:"cols"},[cM.col]));
    trackers=[cM];
  }

  // GT reference (always) — highlighted off the shared clock, click seeks playback
  const gtA = tcol("Speaker A — GT","A",frag.gt.A,seekPlay);
  const gtB = tcol("Speaker B — GT","B",frag.gt.B,seekPlay);
  panel.appendChild(el("div",{class:"gt"},[
    el("h3",{text:"Reference transcript (corrected GT)"}),
    el("div",{class:"cols two"},[gtA.col,gtB.col]),
  ]));
  trackers.push(gtA, gtB);

  // playhead + transport + karaoke, all off the shared clock (activeAudio)
  const D = tl.D;
  function loop(){
    const ct = activeAudio ? activeAudio.currentTime : 0;
    const x = ct/D*1000; tl.ph.setAttribute("x1",x); tl.ph.setAttribute("x2",x);
    if(wfPlayhead){ wfPlayhead.setAttribute("x1",x); wfPlayhead.setAttribute("x2",x); }
    if(transport && !transport.isScrub()){ transport.seek.value=ct; transport.tlbl.textContent=fmt(ct)+" / "+fmt(D); }
    trackers.forEach(t=>highlight(t,ct));
    RAF=requestAnimationFrame(loop);
  }
  RAF=requestAnimationFrame(loop);
}

aggregate();
render();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--recordings", nargs="+", default=PILOT,
                    help="recording ids (default: 5 pilot fragments)")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                    help="sweep config names; first is the default shown")
    ap.add_argument("--audio-format", choices=["mp3", "wav"], default="mp3")
    ap.add_argument("--force", action="store_true", help="re-encode existing audio")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    out = args.out.expanduser()
    print(f"eval root : {eval_root}")
    print(f"output    : {out}")
    print(f"configs   : {', '.join(args.configs)}")
    print(f"audio     : {args.audio_format}")
    print("building…")
    data = build(eval_root, out, args.recordings, args.configs,
                 args.audio_format, args.force)

    n_frag = len(data["fragments"])
    n_audio = sum(1 + 3 * len(f["configs"]) for f in data["fragments"])
    size_mb = sum(p.stat().st_size for p in (out / "audio").rglob("*") if p.is_file()) / 1e6
    print(f"\ndone: {n_frag} fragments, {len(data['config_order'])} configs, "
          f"{n_audio} audio files (~{size_mb:.0f} MB)")
    print(f"open: {out / 'index.html'}")
    # token-leak guard — scan only the inlined text (index.html) for a real
    # HF-token shape (`hf_` + a long alnum tail). Binary MP3s coincidentally
    # contain the bytes "hf_", so we never grep them.
    import re as _re
    html = (out / "index.html").read_text(encoding="utf-8")
    hits = _re.findall(r"hf_[A-Za-z0-9]{20,}", html)
    if hits:
        print(f"\n!! WARNING: possible token leak in index.html: {hits[:1]}…")
    else:
        print("token check: no HF token in index.html ✓")


if __name__ == "__main__":
    main()
