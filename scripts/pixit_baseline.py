"""B3 PixIT baseline — the joint diarization+separation rival arm (backlog B3).

Runs `pyannote/speech-separation-ami-1.0` (PixIT/ToTaToNet) over clarin
fragments and scores its transcripts with the SAME ASR and scorer as the
shipped pipeline, so the only variable is the architecture (modular
region-routed cascade vs joint). Ch7 §7.8 external-reference slot.

Shape mirrors `scripts/compare_asr.py` (fixed-ASR swap harness) and reuses its
bundle/GT helpers. ASR parity: the WhisperX config is read from the SHIPPED
best YAML (`--config`), not re-declared here.

Stages (each resume-safe; per-recording failure isolation, batch continues):
  separate    $PIXIT_VENV_PY scripts/pixit_worker.py per fragment (subprocess,
              neutral cwd) -> <work>/<rec>/source_*.wav + diarization.json + meta.json
  transcribe  asr_pipeline _WhisperXBackend on each source -> <work>/<rec>/pixit_{A,B}.txt
  score       cpWER/cpCER vs <gt-root>/<rec>/annotation.eaf -> <work>/pixit_scores.csv

Source→A/B mapping: PixIT's two sources are ranked by total diarized speech
duration; cpWER/cpCER permute speakers anyway, so the mapping only names the
files. Fragments where PixIT finds ≠2 speakers are reported loudly and scored
on the top-2 sources (the miscount itself is part of the measured
architecture difference — see docs/fable_plans/b3_pixit_arm.md).

Usage:
  python scripts/pixit_baseline.py --split dev
  python scripts/pixit_baseline.py --recordings 005cba37__seg00 --stages separate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_pipeline.config import load_pipeline_config_from_yaml            # noqa: E402
from asr_pipeline.eval.metrics import cpwer_meeteval, cp_cer_meeteval     # noqa: E402
from scripts.compare_asr import (                                         # noqa: E402
    _fmt, _load_gt, _refchars, _refwords, _segments_to_utts,
)
from scripts.eval_harness import REPO, eval_root, load_split              # noqa: E402

WORKER = REPO / "scripts" / "pixit_worker.py"
DEF_FRAG_ROOT = "~/datasets/clarin_all_2speaker_fragments/fragments"
DEF_CONFIG = str(REPO / "asr_pipeline" / "configs" / "sweep_best_e31_refineplus.yaml")


def _pixit_python() -> str:
    py = os.environ.get("PIXIT_VENV_PY")
    if not py or not Path(py).expanduser().exists():
        sys.exit(
            "pixit_baseline: $PIXIT_VENV_PY unset or missing — the isolated "
            "PixIT venv python (~/pixit_venv/bin/python; recipe in "
            "scripts/pixit_worker.py). No silent fallback."
        )
    return str(Path(py).expanduser())


def _speech_dur_per_label(diar_json: Path) -> dict[str, float]:
    segs = json.loads(diar_json.read_text())["segments"]
    dur: dict[str, float] = {}
    for s in segs:
        dur[s["speaker"]] = dur.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    return dur


def _top2_sources(rec_dir: Path, rec: str) -> list[Path]:
    """The two source wavs to transcribe, ranked by diarized speech duration."""
    meta = json.loads((rec_dir / "meta.json").read_text())
    labels = meta["labels"]
    dur = _speech_dur_per_label(rec_dir / "diarization.json")
    ranked = sorted(labels, key=lambda l: dur.get(l, 0.0), reverse=True)
    if len(labels) != 2:
        print(
            f"[note] {rec}: PixIT emitted {len(labels)} speaker(s) "
            f"({', '.join(labels) or 'none'}; durations "
            f"{ {l: round(dur.get(l, 0.0), 1) for l in labels} }) — scoring "
            "top-2 by speech duration; miscount recorded in pixit_scores.csv."
        )
    return [rec_dir / f"source_{l}.wav" for l in ranked[:2]]


def separate(recs: list[str], args) -> None:
    # Deliberate: one worker SUBPROCESS per fragment reloads the pyannote
    # pipeline each time (~5-10 s × N). Accepted for per-recording failure
    # isolation + venv isolation at dev-split scale; if this ever runs on the
    # 118-fragment test split repeatedly, add a batch mode to the worker.
    py = _pixit_python()
    frag_root = Path(args.frag_root).expanduser()
    failures = []
    for i, rec in enumerate(recs, 1):
        out_dir = Path(args.work_dir) / rec
        if (out_dir / "meta.json").exists() and not args.force:
            print(f"[{i}/{len(recs)}] {rec}: cached")
            continue
        wav = frag_root / f"{rec}.wav"
        if not wav.exists():
            failures.append((rec, f"missing input {wav}"))
            print(f"[{i}/{len(recs)}] {rec}: MISSING {wav}")
            continue
        cmd = [
            py, str(WORKER), "--in", str(wav), "--out-dir", str(out_dir),
            "--model", args.model, "--num-speakers", str(args.num_speakers),
        ]
        if args.worker_params_json:
            cmd += ["--params-json", args.worker_params_json]
        # Neutral cwd: the repo's local datasets/ package shadows HF datasets
        # inside the worker's venv (see pixit_worker.py docstring).
        r = subprocess.run(
            cmd, cwd=tempfile.gettempdir(), capture_output=True, text=True
        )
        if r.returncode != 0:
            failures.append((rec, (r.stderr or r.stdout).strip()[-400:]))
            print(f"[{i}/{len(recs)}] {rec}: WORKER FAILED\n{failures[-1][1]}")
            continue
        print(f"[{i}/{len(recs)}] {rec}: {r.stdout.strip().splitlines()[-1]}")
    if failures:
        fcsv = Path(args.work_dir) / "separate_failures.csv"
        with open(fcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fragment", "error"])
            w.writerows(failures)
        print(f"[separate] {len(failures)} failure(s) -> {fcsv}")


def transcribe(recs: list[str], args) -> None:
    # ASR parity: take the transcription block verbatim from the shipped config.
    tcfg = load_pipeline_config_from_yaml(args.config).transcription
    from asr_pipeline.stages.transcription import _WhisperXBackend

    be = _WhisperXBackend(tcfg)
    be.load(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        for i, rec in enumerate(recs, 1):
            rec_dir = Path(args.work_dir) / rec
            if not (rec_dir / "meta.json").exists():
                print(f"[{i}/{len(recs)}] {rec}: no separation output — skipped")
                continue
            outs = [rec_dir / f"pixit_{lab}.txt" for lab in ("A", "B")]
            if all(o.exists() for o in outs) and not args.force:
                print(f"[{i}/{len(recs)}] {rec}: cached")
                continue
            for src, lab, out in zip(_top2_sources(rec_dir, rec), ("A", "B"), outs):
                audio, sr = sf.read(str(src), dtype="float32")
                assert sr == 16_000, f"{src}: expected 16 kHz, got {sr}"
                res = be.transcribe(np.asarray(audio, dtype=np.float32))
                out.write_text(_fmt(lab, _segments_to_utts(res)), encoding="utf-8")
            print(f"[{i}/{len(recs)}] {rec}: transcribed")
    finally:
        be.unload()


def score(recs: list[str], args) -> None:
    from asr_pipeline.eval.transcript_parser import parse_transcript_file

    rows = []
    acc = [0.0, 0.0, 0.0, 0.0]  # Σwer*W, ΣW, Σcer*C, ΣC
    for rec in recs:
        rec_dir = Path(args.work_dir) / rec
        gt = _load_gt(Path(args.gt_root) / rec / "annotation.eaf")
        if not gt:
            print(f"[score] {rec}: no GT — skipped")
            continue
        hyp = {}
        for lab in ("A", "B"):
            p = rec_dir / f"pixit_{lab}.txt"
            utts = []
            if p.exists():
                for v in parse_transcript_file(p).values():
                    utts.extend(v)
            hyp[lab] = utts
        if not any(hyp.values()):
            print(f"[score] {rec}: no transcripts — skipped")
            continue
        n_spk = len(json.loads((rec_dir / "meta.json").read_text())["labels"])
        rw, rc = _refwords(gt), _refchars(gt)
        w = cpwer_meeteval(gt, hyp, session_id=rec, lang=args.language)["cpwer"] * 100
        c = cp_cer_meeteval(gt, hyp, session_id=rec, lang=args.language)["cer"] * 100
        rows.append({
            "fragment": rec, "cpWER": round(w, 2), "cpCER": round(c, 2),
            "pixit_num_speakers": n_spk, "ref_words": rw, "ref_chars": rc,
        })
        acc[0] += w * rw; acc[1] += rw; acc[2] += c * rc; acc[3] += rc

    out = Path(args.work_dir) / "pixit_scores.csv"
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                             ["fragment", "cpWER", "cpCER",
                              "pixit_num_speakers", "ref_words", "ref_chars"])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"[score] {len(rows)} fragment(s) -> {out}")
    if acc[1] > 0:
        print(
            f"[score] micro-avg (ref-weighted): "
            f"cpWER {acc[0]/acc[1]:.2f}  cpCER {acc[2]/acc[3]:.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--recordings", nargs="*", default=None)
    ap.add_argument("--stages", default="separate,transcribe,score")
    ap.add_argument("--frag-root", default=DEF_FRAG_ROOT)
    ap.add_argument("--work-dir", default=None,
                    help="default: <eval-root>/_b3_pixit")
    ap.add_argument("--gt-root", default=str(eval_root()))
    ap.add_argument("--config", default=DEF_CONFIG,
                    help="pipeline YAML whose transcription block is reused verbatim")
    ap.add_argument("--model", default="pyannote/speech-separation-ami-1.0")
    ap.add_argument("--num-speakers", type=int, default=2)
    ap.add_argument("--worker-params-json", default=None,
                    help="passed through to pixit_worker --params-json "
                         "(dev-only sensitivity probes)")
    ap.add_argument("--language", default="pl")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-test", action="store_true",
                    help="permit scoring frozen-TEST-split fragments (requires "
                         "a fresh prereg — the test budget is spent)")
    args = ap.parse_args()

    if args.work_dir is None:
        args.work_dir = str(eval_root() / "_b3_pixit")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    recs = args.recordings or load_split(args.split)
    print(f"pixit_baseline: {len(recs)} fragment(s), stages={args.stages}, "
          f"work={args.work_dir}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ("separate", "transcribe", "score"):
            sys.exit(f"unknown stage {s!r}")
    # Test-split guard (added 2026-07-18 after the smoke test accidentally
    # scored a test fragment): scoring ANY frozen-test fragment with a new
    # system is a test exposure — one-shot discipline, prereg first.
    # load_split (not a hand-built path) so a moved/renamed split file fails
    # LOUD here instead of silently disarming the guard.
    exposed = sorted(set(recs) & set(load_split("test")))
    if exposed and "score" in stages and not args.allow_test:
        sys.exit(
            f"pixit_baseline: {len(exposed)} requested fragment(s) are in the "
            f"FROZEN TEST split (e.g. {', '.join(exposed[:3])}). Scoring them "
            "is a test exposure (budget spent 2026-07-04). Pre-register first, "
            "then pass --allow-test."
        )
    if "separate" in stages:
        separate(recs, args)
    if "transcribe" in stages:
        transcribe(recs, args)
    if "score" in stages:
        score(recs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
