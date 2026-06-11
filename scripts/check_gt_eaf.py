"""Lint hand-corrected GT EAFs in the CLARIN-fragment eval tree.

Walks ``~/datasets/eval/clarin_fragments/<id>/annotation.eaf`` (the corrected
ground truth) and flags issues that would silently corrupt cpWER/CER scoring
or signal an incomplete correction:

ERRORS (fix these):
  - tier set != {A, B}            (ELAN renamed/added a tier)
  - empty annotation             (delete it)
  - end <= start                 (zero/negative duration)
  - within-speaker time overlap  (one speaker can't say two things at once)
  - tight identical run (>=5, gaps < 0.5 s)  (likely uncleaned hallucination loop)

INFO (usually fine — the scorer handles them — but worth a glance):
  - filler tokens still present (yyy/eee/mmm/hmm/mhm/yhy)  -> stripped at scoring
  - bracket markup [..]/<..>                                -> stripped at scoring
  - very long annotation (> 30 s)                          -> maybe an un-split block
  - "untouched": GT text identical to the noenh seed       -> not yet corrected?

Usage::

    python scripts/check_gt_eaf.py
    python scripts/check_gt_eaf.py --eval-root ~/datasets/eval/clarin_fragments
    python scripts/check_gt_eaf.py --errors-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asr_pipeline.eval.transcript_parser import parse_eaf  # noqa: E402

EVAL_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
_FILLER = re.compile(r"^(?:y{2,}|e{2,}|m{2,}|hm+|mhm+|yhy)$", re.I)
_BRACKET = re.compile(r"<[^>]*>|\[[^\]]*\]")
_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _norm_concat(tiers) -> str:
    return " ".join(u.text.strip() for lab in sorted(tiers) for u in tiers[lab]).lower()


def check_one(tiers) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    info: list[str] = []

    if set(tiers) != {"A", "B"}:
        errors.append(f"tiers = {sorted(tiers)} (oczekiwano A, B)")

    n_filler = n_bracket = 0
    for lab, utts in tiers.items():
        for u in utts:
            if not u.text.strip():
                errors.append(f"{lab}: pusta adnotacja @ {u.start:.1f}s")
            if u.end <= u.start:
                errors.append(f"{lab}: end<=start @ {u.start:.1f}s")
            if u.end - u.start > 30:
                info.append(f"{lab}: długa adnotacja {u.end - u.start:.0f}s @ {u.start:.1f}s")
            toks = _TOKEN.findall(u.text.lower())
            n_filler += sum(bool(_FILLER.match(t)) for t in toks)
            n_bracket += len(_BRACKET.findall(u.text))

        su = sorted(utts, key=lambda u: u.start)
        for a, b in zip(su, su[1:]):
            if b.start < a.end - 0.05:
                errors.append(f"{lab}: nakładające się adnotacje tego samego mówcy "
                              f"@ {a.start:.1f}/{b.start:.1f}s")
        # tight identical run (hallucination-loop signature)
        run = 1
        for i in range(1, len(su)):
            same = su[i].text.strip().lower() == su[i - 1].text.strip().lower() and su[i].text.strip()
            tight = (su[i].start - su[i - 1].end) < 0.5
            if same and tight:
                run += 1
                if run == 5:
                    errors.append(f"{lab}: ciasne powtórzenie x5+ '{su[i].text.strip()}' "
                                  f"@ {su[i].start:.1f}s (halucynacja?)")
            else:
                run = 1

    if n_filler:
        info.append(f"tokeny-wypełniacze: {n_filler} (usuwane przy liczeniu)")
    if n_bracket:
        info.append(f"markup [..]/<..>: {n_bracket} (usuwany przy liczeniu)")
    return errors, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    ap.add_argument("--errors-only", action="store_true")
    args = ap.parse_args()
    root = args.eval_root.expanduser()

    corrected = untouched = clean = 0
    frag_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for d in frag_dirs:
        gt = d / "annotation.eaf"
        if not gt.exists():
            continue
        tiers = parse_eaf(gt)
        seed = d / "pipeline_noenh" / "annotation.eaf"
        is_corrected = not (seed.exists() and _norm_concat(parse_eaf(seed)) == _norm_concat(tiers))
        if not is_corrected:
            untouched += 1
            continue
        corrected += 1
        errors, info = check_one(tiers)
        if not errors and not (info and not args.errors_only):
            clean += 1
            continue
        if errors or (info and not args.errors_only):
            print(f"\n### {d.name}")
            for e in errors:
                print(f"  ✗ {e}")
            if not args.errors_only:
                for i in info:
                    print(f"  · {i}")

    print(f"\n— corrected: {corrected}  |  untouched(=seed): {untouched}  |  "
          f"clean: {clean}  |  with-flags: {corrected - clean}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
