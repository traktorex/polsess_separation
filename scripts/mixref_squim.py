"""FLOOR_GEN3 WP6 — SQUIM audio-quality map over streams and mixtures.

Non-intrusive TorchAudio-SQUIM (STOI / PESQ / SI-SDR estimates) under the
Layer-2 protocol — chunked at 30 s, speech-presence filtered, mean-aggregated —
for five sources per eval fragment::

    mixture             <eval>/<frag>/<frag>.wav              the raw mixture
    stream_A/stream_B   <eval>/<frag>/sweep/<arm>/stream_*.wav   pipeline output
    mixref_enh_oa050    <eval>/<frag>/sweep/mixref_enh_oa050/enhanced.wav
    mixref_enh_pure     <eval>/<frag>/sweep/mixref_enh_pure/enhanced.wav

The last two come from `scripts/mixref_enhanced.py` (WP1) and are simply skipped
where they do not exist yet, so this can be run before or after the full WP1
sweep.

The scoring is `asr_pipeline.eval.layer2.squim_chunked` — imported, not
reimplemented — with one SQUIM model loaded for the whole run (the same
"load once, reuse across recordings" contract `compute_layer2` documents).

B7 caveat (`thesis-log/sweep_plan/B7_SQUIM_PREREG.md`): SQUIM is a COARSE
ranking instrument here — within-system correlation with true metrics is
ρ≈0.49 — so read these columns as an ordering of audio quality across sources,
not as calibrated quality numbers.

Output: `<eval>/_forensics/floor_gen3/wp6_squim.csv`, appended in place and
idempotent per (fragment, source) — an existing row is kept unless `--force`.

Usage::

    python scripts/mixref_squim.py --split dev --limit 2
    python scripts/mixref_squim.py --split all
    python scripts/mixref_squim.py --split test --sources mixture stream_A stream_B
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import torch
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.eval.layer2 import (                                # noqa: E402
    _load_mono,
    load_squim_model,
    squim_chunked,
    unload_squim_model,
)
from scripts.eval_harness import eval_root, load_split                # noqa: E402


DEFAULT_ARM = "v41_merge"
SAMPLE_RATE = 16_000

# source name -> path relative to <eval_root>/<frag>/ ("{arm}" is filled in).
SOURCES: dict[str, str] = {
    "mixture": "{frag}.wav",
    "stream_A": "sweep/{arm}/stream_A.wav",
    "stream_B": "sweep/{arm}/stream_B.wav",
    "mixref_enh_oa050": "sweep/mixref_enh_oa050/enhanced.wav",
    "mixref_enh_pure": "sweep/mixref_enh_pure/enhanced.wav",
}

FIELDS = [
    "frag_id", "source", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "n_chunks", "duration_s", "path", "scored_utc",
]


def _existing_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {(r["frag_id"], r["source"]) for r in csv.DictReader(f)}


def _drop_rows(csv_path: Path, keys: set) -> None:
    """Rewrite the CSV without `keys` — the `--force` path, so a re-score
    replaces its rows instead of duplicating them."""
    if not csv_path.exists() or not keys:
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        kept = [r for r in csv.DictReader(f) if (r["frag_id"], r["source"]) not in keys]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--eval-root", default=None, help=f"default: {eval_root()}")
    ap.add_argument("--split", default=None, choices=["dev", "test", "all"])
    ap.add_argument("--fragments", nargs="*", default=None,
                    help="explicit fragment ids (overrides --split)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--arm", default=DEFAULT_ARM,
                    help="sweep arm the stream_A/stream_B rows are read from")
    ap.add_argument("--out", default=None,
                    help="default: <eval-root>/_forensics/floor_gen3/wp6_squim.csv")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--force", action="store_true",
                    help="re-score (and replace) rows that already exist")
    args = ap.parse_args()

    root = Path(args.eval_root).expanduser() if args.eval_root else eval_root()
    if args.fragments:
        frags = list(args.fragments)
    elif args.split == "all":
        frags = load_split("dev") + load_split("test")
    elif args.split:
        frags = load_split(args.split)
    else:
        sys.exit("mixref_squim: pass --split {dev,test,all} or --fragments ...")
    if args.limit:
        frags = frags[: args.limit]

    out_csv = (Path(args.out).expanduser() if args.out
               else root / "_forensics" / "floor_gen3" / "wp6_squim.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build the work list first so the model is only loaded when there is work.
    todo: list[tuple[str, str, Path]] = []
    missing = 0
    have = _existing_keys(out_csv)
    for frag in frags:
        for source in args.sources:
            path = root / frag / SOURCES[source].format(frag=frag, arm=args.arm)
            if not path.exists():
                missing += 1
                continue
            if (frag, source) in have and not args.force:
                continue
            todo.append((frag, source, path))

    print(f"mixref_squim: {len(frags)} fragment(s) x {len(args.sources)} source(s) "
          f"-> {len(todo)} to score ({len(have)} already in CSV, "
          f"{missing} source file(s) absent)")
    print(f"  out: {out_csv}")
    if not todo:
        return 0

    if args.force:
        _drop_rows(out_csv, {(f, s) for f, s, _ in todo})

    write_header = not out_csv.exists()
    model, device = load_squim_model(args.device)
    print(f"  SQUIM loaded on {device}")
    if str(device).startswith("cuda"):
        # Same no-spill guard as mixref_enhanced.py: on WSL2/WDDM the driver
        # spills past VRAM into shared memory (host BSOD) instead of OOM-ing.
        torch.cuda.set_per_process_memory_fraction(0.82)
    t0 = time.perf_counter()
    n_done = 0
    try:
        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            for i, (frag, source, path) in enumerate(todo, 1):
                t = time.perf_counter()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                    free_b, total_b = torch.cuda.mem_get_info()
                    used_mib = (total_b - free_b) / 2**20
                    if used_mib > 10800:
                        raise SystemExit(
                            f"[gpu] ABORT at row {i}: {used_mib:.0f} MiB used — "
                            "refusing to risk a WDDM sysmem spill (host BSOD). "
                            "CSV rows written so far are kept; re-run to resume."
                        )
                    if i % 25 == 1:
                        print(f"    [gpu] {used_mib:.0f} MiB in use")
                # _load_mono fails loud on a sample-rate mismatch — every file in
                # the tree is 16 kHz by construction, and a resampled one would
                # silently shift SQUIM's estimates.
                audio = _load_mono(path, SAMPLE_RATE)
                res = squim_chunked(audio, SAMPLE_RATE, model, device)
                w.writerow({
                    "frag_id": frag,
                    "source": source,
                    "squim_stoi": res["squim_stoi"],
                    "squim_pesq": res["squim_pesq"],
                    "squim_si_sdr": res["squim_si_sdr"],
                    "n_chunks": res["n_chunks"],
                    "duration_s": round(len(audio) / SAMPLE_RATE, 3),
                    "path": str(path),
                    "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                fh.flush()          # appendable + crash-safe: rows land as scored
                n_done += 1
                print(f"  [{i}/{len(todo)}] {frag}/{source}: "
                      f"stoi={res['squim_stoi']:.3f} pesq={res['squim_pesq']:.3f} "
                      f"sisdr={res['squim_si_sdr']:6.2f} "
                      f"({res['n_chunks']} chunk(s), {time.perf_counter()-t:.2f}s)")
    finally:
        unload_squim_model(model)

    wall = time.perf_counter() - t0
    print(f"\nscored {n_done} row(s) in {wall:.1f}s "
          f"({wall/max(n_done,1):.2f}s per row) -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
