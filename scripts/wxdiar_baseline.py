"""WhisperX + diarization word-assignment baseline ("wxdiar") — offline.

The standard *no-separation* speaker-attributed transcription recipe (what
``whisperx --diarize`` does): transcribe the mixture once, align words, give
every word to the diarization speaker whose turn overlaps it most. Built here
OFFLINE from a completed pipeline run's artifacts so that it is matched to the
pipeline on everything except separation:

* **same ASR pass** — ``transcript_mixture.json`` of the source arm: the
  pipeline's own WhisperX (large-v2, wav2vec2 word alignment, loop-retry live)
  over the raw mixture; ``--transcript-arm mixref_enh_oa050`` swaps in the
  enhanced-mixture transcript (OA-0.5 blend, the pipeline's enhancement stage)
  for the enhancement-matched variant;
* **same diarization** — ``diarization.json`` of the source arm (under
  ``v41_merge``: Sortformer v1 offline + ECAPA2 merge-fold);
* **same scorer** — per-speaker transcripts are written as a NEW arm directory
  ``<frag>/sweep/<out-arm>/transcript_{A,B}.txt`` (+ ``.json``, + a copy of
  ``transcript_mixture.txt`` for mixture-floor parity, + provenance) so
  ``scripts/rescore_stratified.py`` scores it like any campaign arm.

Assignment policy (``whisperx.diarize.assign_word_speakers`` semantics,
re-implemented explicitly so the knobs are visible):

* ``--level word`` (default): each aligned word → speaker with the largest
  time intersection; a word without timestamps inherits its Whisper segment's
  dominant speaker. ``--level segment``: the whole Whisper segment → its
  dominant speaker (the coarser recipe).
* ``--unassigned nearest`` (default): a word/segment overlapping no turn goes
  to the nearest turn (``fill_nearest=True`` — drop nothing); ``drop`` reproduces
  whisperx's default of leaving such words speaker-less (they are then lost).

Overlap regions are where this recipe structurally fails — one transcript
stream cannot carry two simultaneous speakers — which is exactly what the
separation contrast is meant to expose; nothing here is tuned to hide it.

No GPU, no model: pure bookkeeping over existing JSONs. Test split refused
without ``--allow-test`` (one-shot discipline: policy is chosen on dev,
pre-registered, then computed on test once).

Usage::

    venv/bin/python scripts/wxdiar_baseline.py --split dev \\
        --source-arm v41_merge --out-arm v41_wxdiar_raw
    venv/bin/python scripts/wxdiar_baseline.py --split dev \\
        --source-arm v41_merge --transcript-arm mixref_enh_oa050 \\
        --out-arm v41_wxdiar_enh
    venv/bin/python scripts/rescore_stratified.py --split dev \\
        --anchor v41_merge --configs v41_wxdiar_raw v41_merge_nosep --mixture
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = Path("/home/user/datasets/eval/clarin_fragments")
LABELS = ("A", "B")


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def load_split(split: str) -> list[str]:
    txt = (REPO / "asr_pipeline" / "eval" / f"clarin_{split}.txt").read_text().split()
    return list(dict.fromkeys(t for t in txt if t))


# ---------------------------------------------------------------------------
# Diarization turns → two labelled speakers
# ---------------------------------------------------------------------------
def load_turns(diar_json: Path) -> list[tuple[float, float, str]]:
    d = json.loads(diar_json.read_text(encoding="utf-8"))
    turns = [(float(t["start"]), float(t["end"]), str(t["speaker"]))
             for t in d["turns"] if float(t["end"]) > float(t["start"])]
    return sorted(turns)


def speaker_labels(turns) -> dict[str, str]:
    """Map diarization speaker ids → 'A'/'B' by total speech (top-2).

    Under the v41_merge instrument (Sortformer + merge-fold) exactly two ids
    appear; the guard for >2 keeps the two longest and reports the rest, whose
    words are then re-assigned by nearest kept turn (never silently dropped).
    """
    dur: Counter = Counter()
    for s, e, spk in turns:
        dur[spk] += e - s
    ranked = [spk for spk, _ in dur.most_common()]
    mapping = {spk: lab for spk, lab in zip(ranked[:2], LABELS)}
    return mapping


def assign_interval(start: float, end: float, turns, mapping,
                    unassigned: str) -> tuple[str | None, str]:
    """Speaker label for [start, end]: dominant intersection, else nearest/None.

    Returns (label_or_None, how) with how ∈ {overlap, nearest, dropped}.
    """
    inter: defaultdict = defaultdict(float)
    for s, e, spk in turns:
        if spk not in mapping:
            continue
        ov = min(end, e) - max(start, s)
        if ov > 0:
            inter[mapping[spk]] += ov
    if inter:
        return max(inter.items(), key=lambda kv: kv[1])[0], "overlap"
    if unassigned == "drop":
        return None, "dropped"
    mid = 0.5 * (start + end)
    best, best_d = None, float("inf")
    for s, e, spk in turns:
        if spk not in mapping:
            continue
        d = 0.0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
        if d < best_d:
            best, best_d = mapping[spk], d
    return best, "nearest"


# ---------------------------------------------------------------------------
# Transcript → per-speaker word streams → utterances
# ---------------------------------------------------------------------------
def attribute(transcript: dict, turns, mapping, level: str, unassigned: str):
    """Yield (label, start, end, word, how) in transcript order."""
    for seg in transcript.get("segments", []):
        s0, s1 = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        seg_lab, seg_how = assign_interval(s0, s1, turns, mapping, unassigned)
        words = seg.get("words") or []
        if not words:
            # no word list (should not happen with word_timestamps on) — treat
            # the segment text as one unit at segment level
            text = (seg.get("text") or "").strip()
            if text:
                yield seg_lab, s0, s1, text, seg_how
            continue
        for w in words:
            word = str(w.get("word", "")).strip()
            if not word:
                continue
            ws, we = w.get("start"), w.get("end")
            if level == "segment" or ws is None:
                lab, how = seg_lab, (seg_how if level == "segment" else "segment_inherit")
                ws = float(ws) if ws is not None else s0
                we = float(we) if we is not None else s1
            else:
                ws, we = float(ws), float(we if we is not None else ws)
                lab, how = assign_interval(ws, we, turns, mapping, unassigned)
            yield lab, ws, we, word, how


def to_utterances(stream: list[tuple[float, float, str]]):
    """Group consecutive words into utterances; split on gaps > 1.0 s."""
    utts = []
    cur = None
    for ws, we, word in stream:
        if cur is not None and ws - cur[1] <= 1.0:
            cur[1] = max(cur[1], we)
            cur[2].append(word)
        else:
            if cur is not None:
                utts.append(cur)
            cur = [ws, we, [word]]
    if cur is not None:
        utts.append(cur)
    return [(s, e, " ".join(w)) for s, e, w in utts]


def write_txt(path: Path, utts):
    lines = [f"[{s:6.2f} → {e:6.2f}]  {t}" for s, e, t in utts]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--source-arm", default="v41_merge",
                    help="arm whose diarization.json is used (default v41_merge)")
    ap.add_argument("--transcript-arm", default=None,
                    help="arm whose transcript_mixture.json is used "
                         "(default: the source arm = raw-mixture pass)")
    ap.add_argument("--out-arm", required=True,
                    help="new arm name written under <frag>/sweep/")
    ap.add_argument("--level", choices=["word", "segment"], default="word")
    ap.add_argument("--unassigned", choices=["nearest", "drop"], default="nearest")
    ap.add_argument("--eval-root", type=Path, default=EVAL)
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test (one-shot discipline)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("refusing --split test without --allow-test (policy is chosen on "
                 "dev and pre-registered; test is computed once)")
    tarm = args.transcript_arm or args.source_arm
    frags = load_split(args.split)
    head = _git_head()
    tot = Counter()
    rows = []
    for fid in frags:
        src = args.eval_root / fid / "sweep" / args.source_arm
        tsrc = args.eval_root / fid / "sweep" / tarm
        out = args.eval_root / fid / "sweep" / args.out_arm
        diar = src / "diarization.json"
        tjson = tsrc / "transcript_mixture.json"
        ttxt = tsrc / "transcript_mixture.txt"
        if not diar.exists() or not tjson.exists():
            print(f"[{fid}] MISSING {diar if not diar.exists() else tjson} — skipped")
            continue
        if (out / "transcript_A.txt").exists() and not args.force:
            print(f"[{fid}] exists, skip (use --force)")
            continue
        turns = load_turns(diar)
        mapping = speaker_labels(turns)
        transcript = json.loads(tjson.read_text(encoding="utf-8"))
        streams = {lab: [] for lab in LABELS}
        how = Counter()
        for lab, ws, we, word, h in attribute(transcript, turns, mapping,
                                             args.level, args.unassigned):
            how[h] += 1
            if lab is None:
                continue
            streams[lab].append((ws, we, word))
        out.mkdir(parents=True, exist_ok=True)
        utts_all = {}
        for lab in LABELS:
            utts = to_utterances(sorted(streams[lab]))
            utts_all[lab] = utts
            write_txt(out / f"transcript_{lab}.txt", utts)
            (out / f"transcript_{lab}.json").write_text(json.dumps({
                "segments": [{"start": s, "end": e, "text": t} for s, e, t in utts]},
                ensure_ascii=False, indent=1), encoding="utf-8")
        if ttxt.exists():
            shutil.copyfile(ttxt, out / "transcript_mixture.txt")
        shutil.copyfile(tjson, out / "transcript_mixture.json")
        for extra in ("diarization.json", "routing.json"):
            if (src / extra).exists():
                shutil.copyfile(src / extra, out / extra)
        n_words = {lab: sum(len(t.split()) for _s, _e, t in utts_all[lab]) for lab in LABELS}
        meta = {
            "arm": args.out_arm,
            "derived": "scripts/wxdiar_baseline.py — offline WhisperX+diarization "
                       "word-assignment baseline (no separation)",
            "git_head": head,
            "source_arm_diarization": args.source_arm,
            "transcript_arm": tarm,
            "level": args.level, "unassigned": args.unassigned,
            "speaker_map": mapping,
            "n_diar_speakers": len({spk for _s, _e, spk in turns}),
            "assignment_counts": dict(how),
            "words_per_stream": n_words,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
        tot.update(how)
        rows.append((fid, dict(how), n_words, len({spk for _s, _e, spk in turns})))
        print(f"[{fid}] spk={len({spk for _s,_e,spk in turns})} {dict(how)} words A/B="
              f"{n_words['A']}/{n_words['B']}")
    print(f"\n[done] {len(rows)} fragments → sweep/{args.out_arm}  "
          f"(level={args.level}, unassigned={args.unassigned}, transcript={tarm}, "
          f"diarization={args.source_arm}); assignment totals {dict(tot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
