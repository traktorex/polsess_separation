"""Held-out ASR comparison: WhisperX vs Cohere on FIXED dr_refineplus streams.

A *fixed-audio ASR-only swap* — the per-speaker assembled streams are held
constant and the only variable is the transcriber, so any score difference is
attributable to the ASR (not the separator/diarizer). This is the harness for the
WhisperX-vs-Cohere decision once the held-out TEST ground truth is annotated; it
also produces the per-fragment bundles the per-transcript eyeball pass consumes.

Pipeline of evidence (see `_forensics/ANCHORING_EYEBALL_SYNTHESIS.md`): cross-arch
WER is *reference-seed biased* (~±4 pp) — each hand-corrected GT mildly flatters the
ASR it was seeded from. So read the numbers TWO ways: (1) each ASR against the GT,
(2) if a second, differently-seeded GT is supplied via --gt2-root, the cross-seed
table (each ASR on the OTHER seed's reference removes home-field advantage). The
reference-free eyeball pass over the dumped bundles is the seed-independent tiebreak.

PREREQUISITE — produce the streams + WhisperX hyps first (the heavy pipeline pass):
    python scripts/sweep_pipeline.py --configs dr_refineplus --recordings <ids...>
which writes <eval>/<id>/sweep/dr_refineplus/{stream_A,stream_B}.wav and
transcript_{A,B}.txt. This script then adds only the Cohere pass + scoring.

Usage:
    COHEREX_VENV_PY=~/asr_model_compare/coherex_venv/bin/python \
        python scripts/compare_asr.py --split test
    # explicit list + a 2nd (cross-seed) GT, e.g. a Cohere-seeded reference:
    ... python scripts/compare_asr.py --recordings a__seg00 b__seg00 \
          --gt2-root /mnt/f/clarin_fragments_cohere --gt2-label cohere-seeded
    # re-score only (skip the Cohere transcription, reuse existing bundles):
    ... python scripts/compare_asr.py --split test --score-only

Outputs under <work-dir> (default <eval>/_forensics/asr_compare/): per-fragment
bundle dirs (whisperx_{A,B}.txt / cohere_{A,B}.txt / gt_{A,B}.txt [/ gt2_*]) +
scores.csv + SUMMARY.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asr_pipeline.config import TranscriptionConfig                       # noqa: E402
from asr_pipeline.eval.metrics import cpwer_meeteval, cp_cer_meeteval     # noqa: E402
from asr_pipeline.eval.transcript_parser import (                        # noqa: E402
    Utterance, parse_eaf, parse_gt_txt, parse_transcript_file,
)

ARROW = "→"
DEF_ASR = "CohereLabs/cohere-transcribe-03-2026"
DEF_ALIGN = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"


# --------------------------------------------------------------------------- #
# IO helpers (uniform bundle format shared by both ASRs + the GT dumps)
# --------------------------------------------------------------------------- #
def _fmt(label: str, utts: list[Utterance]) -> str:
    lines = [f"=== Speaker {label} ==="]
    for u in utts:
        s = u.start if u.start is not None else 0.0
        e = u.end if u.end is not None else 0.0
        t = (u.text or "").strip()
        if t:
            lines.append(f"[{s:7.2f} {ARROW} {e:7.2f}]  {t}")
    return "\n".join(lines) + "\n"


def _segments_to_utts(result: dict) -> list[Utterance]:
    return [
        Utterance(seg.get("start", 0.0), seg.get("end", 0.0), (seg.get("text") or "").strip())
        for seg in (result.get("segments") or [])
        if (seg.get("text") or "").strip()
    ]


def _load_bundle_hyp(d: Path, asr: str) -> dict[str, list[Utterance]]:
    """Read whisperx_/cohere_ bundle txt -> {'A':[...], 'B':[...]}."""
    out: dict[str, list[Utterance]] = {}
    for lab in ("A", "B"):
        p = d / f"{asr}_{lab}.txt"
        utts: list[Utterance] = []
        if p.exists():
            for v in parse_transcript_file(p).values():
                utts.extend(v)
        out[lab] = utts
    return out


def _load_gt(eaf: Path) -> dict[str, list[Utterance]]:
    if not eaf.exists():
        return {}
    return {k[-1]: [u for u in v if (u.text or "").strip()] for k, v in parse_eaf(eaf).items()}


def _refwords(gt: dict) -> int:
    return sum(len((u.text or "").split()) for v in gt.values() for u in v)


def _refchars(gt: dict) -> int:
    return sum(len(" ".join((u.text or "").split())) for v in gt.values() for u in v)


# --------------------------------------------------------------------------- #
def resolve_recordings(args) -> list[str]:
    if args.recordings:
        return args.recordings
    split = REPO / "asr_pipeline" / "eval" / f"clarin_{args.split}.txt"
    return split.read_text().split()


def cohere_transcribe(recs, args) -> None:
    """Cohere over each fragment's dr_refineplus streams -> bundle cohere_{A,B}.txt.
    Checkpoints per stream (skips an existing output) so a stall can be resumed."""
    cfg = TranscriptionConfig(
        backend="coherex", model_name=args.asr_model,
        language=args.language, align_model_name=args.align_model,
    )
    from asr_pipeline.stages.transcription import _CohereXBackend
    be = _CohereXBackend(cfg)
    be.load(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        for i, rec in enumerate(recs, 1):
            sweep = Path(args.eval_root) / rec / "sweep" / args.config_name
            bundle = Path(args.work_dir) / rec
            bundle.mkdir(parents=True, exist_ok=True)
            for lab in ("A", "B"):
                out = bundle / f"cohere_{lab}.txt"
                if out.exists() and not args.force:
                    continue
                wav = sweep / f"stream_{lab}.wav"
                if not wav.exists():
                    print(f"[{i}/{len(recs)}] {rec} {lab}: MISSING {wav} — run sweep dr_refineplus first")
                    continue
                audio, _ = sf.read(str(wav))
                res = be.transcribe(np.asarray(audio, dtype=np.float32))
                out.write_text(_fmt(lab, _segments_to_utts(res)), encoding="utf-8")
            print(f"[{i}/{len(recs)}] {rec}: cohere done")
    finally:
        be.unload()


def build_bundle(recs, args) -> None:
    """Copy WhisperX hyps + dump GT(s) into each bundle in the uniform format."""
    for rec in recs:
        sweep = Path(args.eval_root) / rec / "sweep" / args.config_name
        bundle = Path(args.work_dir) / rec
        bundle.mkdir(parents=True, exist_ok=True)
        # WhisperX hyp: re-emit the sweep transcript in the uniform bundle format.
        # Sweep per-speaker files are HEADERLESS (speaker is in the filename), so
        # they parse with parse_gt_txt, not the header-aware parse_transcript_file.
        for lab in ("A", "B"):
            tp = sweep / f"transcript_{lab}.txt"
            utts = parse_gt_txt(tp) if tp.exists() else []
            (bundle / f"whisperx_{lab}.txt").write_text(_fmt(lab, utts), encoding="utf-8")
        # GT dump(s)
        for tag, root in [("gt", args.gt_root)] + ([("gt2", args.gt2_root)] if args.gt2_root else []):
            gt = _load_gt(Path(root) / rec / "annotation.eaf")
            for lab, utts in gt.items():
                (bundle / f"{tag}_{lab}.txt").write_text(_fmt(lab, utts), encoding="utf-8")


def score(recs, args) -> None:
    gts = [("gt", args.gt_root, args.gt_label)]
    if args.gt2_root:
        gts.append(("gt2", args.gt2_root, args.gt2_label))
    rows = []
    # accumulators for micro-avg: {(gt_tag, asr): [Σ wer*W, ΣW, Σ cer*C, ΣC]}
    acc: dict[tuple, list[float]] = {}
    for rec in recs:
        bundle = Path(args.work_dir) / rec
        hyps = {asr: _load_bundle_hyp(bundle, asr) for asr in ("whisperx", "cohere")}
        for tag, root, _lbl in gts:
            gt = _load_gt(Path(root) / rec / "annotation.eaf")
            if not gt:
                continue
            rw, rc = _refwords(gt), _refchars(gt)
            for asr in ("whisperx", "cohere"):
                if not any(hyps[asr].values()):
                    continue
                w = cpwer_meeteval(gt, hyps[asr], session_id=rec, lang=args.language)["cpwer"] * 100
                c = cp_cer_meeteval(gt, hyps[asr], session_id=rec, lang=args.language)["cer"] * 100
                rows.append({"fragment": rec, "gt": tag, "asr": asr,
                             "cpWER": round(w, 2), "cpCER": round(c, 2),
                             "ref_words": rw, "ref_chars": rc})
                a = acc.setdefault((tag, asr), [0.0, 0.0, 0.0, 0.0])
                a[0] += w * rw; a[1] += rw; a[2] += c * rc; a[3] += rc

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["fragment", "gt", "asr", "cpWER", "cpCER", "ref_words", "ref_chars"])
        wr.writeheader(); wr.writerows(rows)

    # SUMMARY.md (micro-avg per GT x ASR + gaps)
    lines = [f"# ASR comparison — {len(recs)} fragments (fixed dr_refineplus streams)\n"]
    gt_labels = {tag: lbl for tag, _r, lbl in gts}
    for tag in [g[0] for g in gts]:
        if (tag, "whisperx") not in acc:
            continue
        wx, co = acc[(tag, "whisperx")], acc[(tag, "cohere")]
        wxw, wxc = wx[0] / wx[1], wx[2] / wx[3]
        cow, coc = co[0] / co[1], co[2] / co[3]
        lines.append(f"## vs {gt_labels[tag]} ({tag})")
        lines.append(f"- WhisperX: cpWER {wxw:.2f}  cpCER {wxc:.2f}")
        lines.append(f"- Cohere:   cpWER {cow:.2f}  cpCER {coc:.2f}")
        lines.append(f"- gap (Cohere - WhisperX): cpWER {cow - wxw:+.2f}  cpCER {coc - wxc:+.2f}  "
                     f"({'Cohere' if cow < wxw else 'WhisperX'} better WER, "
                     f"{'Cohere' if coc < wxc else 'WhisperX'} better CER)\n")
    if args.gt2_root:
        # cross-seed: each ASR on the OTHER-seeded reference (home-field removed).
        lines.append("## cross-seed (each ASR on the other seed's reference — fairest)")
        lines.append("Interpret by which seed each GT came from; rows above are the home/away pair.\n")
    (work / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {work/'scores.csv'} and {work/'SUMMARY.md'} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recordings", nargs="+")
    g.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--eval-root", default=str(Path.home() / "datasets/eval/clarin_fragments"))
    ap.add_argument("--config-name", default="dr_refineplus")
    ap.add_argument("--gt-root", default=None, help="GT root (default: --eval-root)")
    ap.add_argument("--gt-label", default="GT")
    ap.add_argument("--gt2-root", default=None, help="optional 2nd, differently-seeded GT for the cross-seed read")
    ap.add_argument("--gt2-label", default="GT2")
    ap.add_argument("--work-dir", default=None, help="default: <eval-root>/_forensics/asr_compare")
    ap.add_argument("--asr-model", default=DEF_ASR)
    ap.add_argument("--align-model", default=DEF_ALIGN)
    ap.add_argument("--language", default="pl")
    ap.add_argument("--score-only", action="store_true", help="skip Cohere transcription, re-score existing bundles")
    ap.add_argument("--force", action="store_true", help="re-transcribe even if a cohere_*.txt exists")
    args = ap.parse_args()
    if args.gt_root is None:
        args.gt_root = args.eval_root
    if args.work_dir is None:
        args.work_dir = str(Path(args.eval_root) / "_forensics" / "asr_compare")

    recs = resolve_recordings(args)
    print(f"comparing WhisperX vs Cohere on {len(recs)} fragments "
          f"(config={args.config_name}, work_dir={args.work_dir})\n")
    if not args.score_only:
        cohere_transcribe(recs, args)
    build_bundle(recs, args)
    score(recs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
