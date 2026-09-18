"""FLOOR_GEN3 WP1 — enhanced-mixture reference arms.

Produces, per eval fragment, a *single-stream* transcript of the ENHANCED
mixture: the pipeline's own enhancement stage applied to the raw mixture, then
the pipeline's own mixture-transcription path on the result. Two variants:

  mixref_enh_oa050  observation_mix_ratio = 0.50 — the deployed blend
                    (identical to the shipped/v41_merge enhancement stage)
  mixref_enh_pure   observation_mix_ratio = 0.00 — pure FRCRN output, no
                    observation adding

Outputs land beside the sweep arms so `asr_pipeline.eval.layer3.read_mixture`
reads them unchanged::

    <eval_root>/<frag>/sweep/<variant>/
        enhanced.wav              16-bit WAV — EXACTLY the audio transcribed
        transcript_mixture.txt    written by asr_pipeline.io.write_pipeline_outputs
        transcript_mixture.json
        metadata.json             config snapshot (as for every arm dir)
        run_meta.json             provenance + per-phase wall-clock

**These arms are reference FLOORS for the WP2/WP7 decomposition, declared
NON-ADOPTABLE (FLOOR_GEN3_PLAN.md guardrail 2). They may be cited like the fair
mixture baseline, never as candidate configs.**

Parity, not re-implementation
-----------------------------
The config is `fresh_eval_cfg(default.yaml)` + `sweep_pipeline.CONFIGS[<arm>]`
— literally the object that produced `<frag>/sweep/<arm>/` on disk — and the
work is done by the pipeline's own stage objects driven through
`Pipeline.run_stage`:

  * enhancement  → `stages/enhancement.py:EnhancementStage.run` (FRCRN forward,
    Hann overlap-add, then the OA blend). The only thing this script varies
    between the two variants is `enhancement.observation_mix_ratio`, which is
    not part of `EnhancementStage.load_signature()`, so the model is loaded once.
  * transcription → `stages/transcription.py:TranscriptionStage.run`, whose
    `transcribe_mixture` branch calls `_WhisperXBackend.transcribe` — i.e. the
    collapse retry, `loop_retry`, `loop_retry_phrase` and the length bound are
    live, exactly as they are for the per-arm `transcript_mixture.txt`.
  * determinism → `Pipeline.__init__` applies `config.deterministic` (cuDNN
    deterministic, benchmark off) before anything runs.

Only the *feed* differs from a normal run: transcription is handed the enhanced
mixture instead of `ctx.audio`, and no other stage runs (no diarization / no
routing / no separation / no assembly). `metadata.json`'s config snapshot is
therefore the whole arm config, of which only the `enhancement` and
`transcription` blocks were exercised; `run_meta.json` says so explicitly.

Phase-major over the whole fragment set (12 GB card, one model at a time):
enhance everything with FRCRN resident, release it, then transcribe everything
with WhisperX resident. A per-fragment reload of either model would dominate.

Idempotent per (fragment, variant): the enhance phase skips a unit whose
`enhanced.wav` exists, the transcribe phase one whose `transcript_mixture.txt`
exists. `--force` recomputes. Per-unit failures are isolated (SCOPE §4.2): the
run continues and writes `_forensics/floor_gen3/mixref_failures.csv`.

Usage::

    python scripts/mixref_enhanced.py --fragments 065a9896__seg00 --limit 2
    python scripts/mixref_enhanced.py --split test
    python scripts/mixref_enhanced.py --split dev --variants mixref_enh_pure
    python scripts/mixref_enhanced.py --split test --phases asr   # resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# eval_harness is dependency-light (no torch, no asr_pipeline) — safe to import
# before the debug-log path is pinned below.
from scripts.eval_harness import eval_root, load_split                # noqa: E402

# `asr_pipeline.debug_log` resolves its log path at IMPORT time, so this must be
# set before any asr_pipeline import. A durable per-campaign log keeps the
# loop-retry / phrase-loop-retry decisions citable after the run (the default
# /tmp path is truncated by every `Pipeline.load_audio`; we never call it).
_FORENSICS = eval_root() / "_forensics" / "floor_gen3"
_FORENSICS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ASR_PIPELINE_DEBUG_LOG", str(_FORENSICS / "mixref_debug.log"))

from asr_pipeline.context import PipelineContext                      # noqa: E402
from asr_pipeline.eval.layer3 import read_mixture                     # noqa: E402
from asr_pipeline.io import load_audio_as_mono, write_pipeline_outputs  # noqa: E402
from asr_pipeline.pipeline import Pipeline                            # noqa: E402
from scripts.sweep_pipeline import CONFIGS, _build_cfg, _git_head     # noqa: E402


# variant name -> enhancement.observation_mix_ratio
VARIANTS: dict[str, float] = {
    "mixref_enh_oa050": 0.50,   # the deployed blend (must equal the arm's value)
    "mixref_enh_pure": 0.00,    # pure FRCRN, no observation adding
}

# The sweep-registry arm whose enhancement + transcription blocks we mirror.
DEFAULT_ARM = "v41_merge"

FAILURES_CSV = _FORENSICS / "mixref_failures.csv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config(arm: str):
    """The arm's PipelineConfig + fail-loud parity assertions.

    `_build_cfg` is `sweep_pipeline`'s own builder (`fresh_eval_cfg(default.yaml)`
    + the registry overrides), so this is the same object the arm dirs on disk
    were produced with. The assertions pin the properties this script depends
    on: if the registry row is ever edited, the mismatch surfaces here instead
    of quietly producing a non-comparable reference floor.
    """
    if arm not in CONFIGS:
        sys.exit(f"mixref: unknown arm {arm!r} (not in sweep_pipeline.CONFIGS)")
    cfg = _build_cfg(CONFIGS[arm])

    problems = []
    if not cfg.enhancement.enabled:
        problems.append("enhancement.enabled is False")
    if cfg.enhancement.backend != "frcrn_se_16k":
        problems.append(f"enhancement.backend={cfg.enhancement.backend!r} != 'frcrn_se_16k'")
    if abs(cfg.enhancement.observation_mix_ratio - VARIANTS["mixref_enh_oa050"]) > 1e-12:
        problems.append(
            f"enhancement.observation_mix_ratio={cfg.enhancement.observation_mix_ratio} "
            f"!= mixref_enh_oa050's {VARIANTS['mixref_enh_oa050']}"
        )
    if cfg.transcription.backend != "whisperx":
        problems.append(f"transcription.backend={cfg.transcription.backend!r} != 'whisperx'")
    if not cfg.transcription.transcribe_mixture:
        problems.append("transcription.transcribe_mixture is False")
    if not cfg.transcription.loop_retry:
        problems.append("transcription.loop_retry is False")
    if not cfg.transcription.loop_retry_phrase:
        problems.append("transcription.loop_retry_phrase is False")
    if not cfg.deterministic:
        problems.append("deterministic is False")
    if problems:
        sys.exit(
            f"mixref: arm {arm!r} no longer matches the WP1 reference contract:\n  "
            + "\n  ".join(problems)
            + "\nFix the registry row or update VARIANTS/this check deliberately."
        )
    return cfg


# ---------------------------------------------------------------------------
# run_meta.json (read-modify-write: the two phases update it independently)
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_run_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_run_meta(path: Path, updates: dict) -> dict:
    """Merge `updates` into the unit's run_meta.json (phases merge, not replace)."""
    meta = _read_run_meta(path)
    phases = dict(meta.get("phases", {}))
    phases.update(updates.pop("phases", {}))
    meta.update(updates)
    meta["phases"] = phases
    # `seconds` mirrors the arm run_meta key read by sweep_pipeline._read_run_seconds:
    # inference wall-clock only (model loads are amortised across the whole run).
    meta["seconds"] = sum(float(p.get("run_s", 0.0)) for p in phases.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _stage_timings(meta: dict) -> list:
    """The per-stage rows for metadata.json, in the shape batch.py records."""
    rows = []
    for phase, stage in (("enh", "enhancement"), ("asr", "transcription")):
        p = meta.get("phases", {}).get(phase)
        if p:
            rows.append({"stage": stage, "load_s": p.get("load_s"), "run_s": p.get("run_s")})
    return rows


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


class _StageTimer:
    """Collector for `Pipeline`'s stage_end events (load/run split)."""

    def __init__(self) -> None:
        self.last: dict = {}

    def __call__(self, event: dict) -> None:
        if event.get("event") == "stage_end":
            self.last = dict(event)


def _unit_dir(root: Path, frag: str, variant: str) -> Path:
    return root / frag / "sweep" / variant


# On WSL2/WDDM the driver silently overcommits VRAM into shared system memory
# instead of raising OOM; the observed result (twice, 2026-07-28) is a hard
# host BSOD, not an error. Three layers so a spill cannot happen: (a) cap the
# PyTorch caching allocator well below VRAM, (b) release cached blocks between
# units so variable-length fragments cannot ratchet reserved memory upward,
# (c) enforce a driver-level watermark via mem_get_info() — which also sees
# ctranslate2's (WhisperX) non-PyTorch allocations — and abort loudly instead.
# Both phases are idempotent, so an abort is resumable by re-running.
_GPU_MEM_FRACTION = 0.82      # PyTorch allocator cap (~10.0 of 12.0 GiB)
_GPU_WATERMARK_MIB = 10800    # driver-level abort threshold


def _gpu_cap() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_GPU_MEM_FRACTION)


def _gpu_housekeep(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    used_mib = (total_b - free_b) / 2**20
    print(f"    [gpu] {tag}: {used_mib:.0f} MiB in use")
    if used_mib > _GPU_WATERMARK_MIB:
        raise SystemExit(
            f"[gpu] ABORT at {tag}: {used_mib:.0f} MiB used > watermark "
            f"{_GPU_WATERMARK_MIB} MiB — refusing to risk a WDDM sysmem spill "
            "(host BSOD). Outputs written so far are complete; re-run the same "
            "command to resume."
        )


def phase_enhance(pipe, cfg, arm, timer, frags, variants, root, force, failures) -> dict:
    """FRCRN resident: write `enhanced.wav` for every pending (fragment, variant)."""
    stats = {"done": 0, "skipped": 0, "run_s": 0.0, "audio_s": 0.0}
    for i, frag in enumerate(frags, 1):
        wav = root / frag / f"{frag}.wav"
        pending = [
            v for v in variants
            if force or not (_unit_dir(root, frag, v) / "enhanced.wav").exists()
        ]
        if not pending:
            stats["skipped"] += len(variants)
            print(f"  [{i}/{len(frags)}] {frag}: enhanced.wav present for all variants — skip")
            continue
        if not wav.exists():
            for v in pending:
                failures.append((frag, v, "enhance", f"missing mixture {wav}"))
            print(f"  [{i}/{len(frags)}] {frag}: MISSING {wav}")
            continue
        stats["skipped"] += len(variants) - len(pending)
        _gpu_housekeep(f"enh [{i}/{len(frags)}]")
        try:
            audio = load_audio_as_mono(str(wav), target_sr=cfg.sample_rate)
        except Exception as exc:                       # noqa: BLE001 — §4.2 isolation
            for v in pending:
                failures.append((frag, v, "enhance", f"load failed: {exc!r}"))
            print(f"  [{i}/{len(frags)}] {frag}: LOAD FAILED {exc!r}")
            continue
        dur = len(audio) / cfg.sample_rate
        for variant in pending:
            out_dir = _unit_dir(root, frag, variant)
            # The stage reads `self.config.observation_mix_ratio` — the same
            # object as cfg.enhancement — and OA is not in load_signature(), so
            # this re-blends without touching the resident FRCRN model.
            cfg.enhancement.observation_mix_ratio = VARIANTS[variant]
            ctx = PipelineContext(input_path=wav, sample_rate=cfg.sample_rate)
            ctx.audio = audio
            try:
                pipe.run_stage("enhancement", ctx)
                if ctx.enhanced_full is None or len(ctx.enhanced_full) != len(audio):
                    got = None if ctx.enhanced_full is None else len(ctx.enhanced_full)
                    raise RuntimeError(
                        f"enhanced length {got} != input length {len(audio)}"
                    )
                out_dir.mkdir(parents=True, exist_ok=True)
                peak = float(np.max(np.abs(ctx.enhanced_full)))
                if peak > 0.999:
                    # PCM_16 would clip. Loud, never silent — the enhanced
                    # reference must not quietly acquire clipping distortion.
                    print(f"  [WARN] {frag}/{variant}: enhanced peak {peak:.4f} "
                          "≳ full scale — PCM_16 write will clip")
                # Same writer call as io.write_pipeline_outputs uses for the
                # stream wavs → PCM_16, the tree's convention. Deliberate over
                # float32: it puts the enhanced mixture in the SAME 16-bit
                # domain as the raw mixture the fair-mixture baseline was
                # transcribed from (no float-precision advantage for this arm),
                # and float WAVs carry a libsndfile PEAK chunk whose embedded
                # timestamp would make otherwise-identical reruns differ.
                sf.write(
                    out_dir / "enhanced.wav",
                    ctx.enhanced_full.astype(np.float32),
                    cfg.sample_rate,
                )
            except Exception as exc:                   # noqa: BLE001 — §4.2 isolation
                failures.append((frag, variant, "enhance", repr(exc)))
                print(f"  [{i}/{len(frags)}] {frag}/{variant}: ENHANCE FAILED {exc!r}")
                continue
            run_s = float(timer.last.get("run_s", float("nan")))
            _write_run_meta(out_dir / "run_meta.json", {
                "fragment": frag,
                "variant": variant,
                "arm_base": arm,
                "observation_mix_ratio": VARIANTS[variant],
                "enhancement_backend": cfg.enhancement.backend,
                "stages_run": ["enhancement", "transcription"],
                "input_wav": str(wav),
                "audio_duration_s": dur,
                "sample_rate": cfg.sample_rate,
                "git_sha": _git_head(),
                "host": platform.node(),
                "phases": {"enh": {
                    "load_s": float(timer.last.get("load_s", float("nan"))),
                    "run_s": run_s,
                    "utc": _now(),
                }},
            })
            stats["done"] += 1
            stats["run_s"] += run_s
            stats["audio_s"] += dur
            print(f"  [{i}/{len(frags)}] {frag}/{variant}: enhanced "
                  f"{dur:6.1f}s audio in {run_s:5.2f}s")
    return stats


def phase_transcribe(pipe, cfg, timer, frags, variants, root, force, failures) -> dict:
    """WhisperX resident: transcript_mixture for every pending (fragment, variant)."""
    stats = {"done": 0, "skipped": 0, "run_s": 0.0, "audio_s": 0.0, "empty": 0}
    for i, frag in enumerate(frags, 1):
        for variant in variants:
            out_dir = _unit_dir(root, frag, variant)
            if (out_dir / "transcript_mixture.txt").exists() and not force:
                stats["skipped"] += 1
                print(f"  [{i}/{len(frags)}] {frag}/{variant}: transcript present — skip")
                continue
            enh = out_dir / "enhanced.wav"
            if not enh.exists():
                failures.append((frag, variant, "transcribe", f"missing {enh}"))
                print(f"  [{i}/{len(frags)}] {frag}/{variant}: MISSING enhanced.wav")
                continue
            _gpu_housekeep(f"asr [{i}/{len(frags)}] {variant}")
            # Keep the snapshot honest: metadata.json must report the OA ratio
            # this variant's audio was produced with.
            cfg.enhancement.observation_mix_ratio = VARIANTS[variant]
            try:
                audio = load_audio_as_mono(str(enh), target_sr=cfg.sample_rate)
                ctx = PipelineContext(input_path=enh, sample_rate=cfg.sample_rate)
                ctx.audio = audio
                pipe.run_stage("transcription", ctx)
                run_s = float(timer.last.get("run_s", float("nan")))
                meta = _write_run_meta(out_dir / "run_meta.json", {
                    "phases": {"asr": {
                        "load_s": float(timer.last.get("load_s", float("nan"))),
                        "run_s": run_s,
                        "utc": _now(),
                    }},
                })
                write_pipeline_outputs(
                    ctx,
                    out_dir=root / frag,
                    subdir_name=f"sweep/{variant}",
                    config_snapshot=asdict(cfg),
                    stage_timings=_stage_timings(meta),
                )
                # Round-trip through the eval reader that will consume these files.
                utts = read_mixture(out_dir)
                if utts is None:
                    raise RuntimeError(
                        "read_mixture() returned None after writing "
                        f"{out_dir/'transcript_mixture.txt'}"
                    )
            except Exception as exc:                   # noqa: BLE001 — §4.2 isolation
                failures.append((frag, variant, "transcribe", repr(exc)))
                print(f"  [{i}/{len(frags)}] {frag}/{variant}: TRANSCRIBE FAILED {exc!r}")
                continue
            dur = len(audio) / cfg.sample_rate
            n_words = sum(len((u.text or "").split()) for u in utts)
            if not utts:
                stats["empty"] += 1
            stats["done"] += 1
            stats["run_s"] += run_s
            stats["audio_s"] += dur
            print(f"  [{i}/{len(frags)}] {frag}/{variant}: {len(utts):3d} utt / "
                  f"{n_words:4d} words in {run_s:5.1f}s (rtf {dur/max(run_s,1e-9):4.1f}x)")
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_failures(failures: list) -> None:
    if not failures:
        return
    with open(FAILURES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fragment", "variant", "phase", "error"])
        w.writerows(failures)
    print(f"\n{len(failures)} failure(s) -> {FAILURES_CSV}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--eval-root", default=None, help=f"default: {eval_root()}")
    ap.add_argument("--split", default=None, choices=["dev", "test", "all"],
                    help="frozen fragment split (all = dev + test)")
    ap.add_argument("--fragments", nargs="*", default=None,
                    help="explicit fragment ids (overrides --split)")
    ap.add_argument("--limit", type=int, default=None, help="first N fragments only")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--phases", default="enh,asr",
                    help="comma list of phases to run: enh,asr (default both)")
    ap.add_argument("--arm", default=DEFAULT_ARM,
                    help="sweep_pipeline.CONFIGS row whose enhancement + "
                         "transcription blocks are mirrored")
    ap.add_argument("--force", action="store_true",
                    help="recompute units whose outputs already exist")
    args = ap.parse_args()

    root = Path(args.eval_root).expanduser() if args.eval_root else eval_root()
    if args.fragments:
        frags = list(args.fragments)
    elif args.split == "all":
        frags = load_split("dev") + load_split("test")
    elif args.split:
        frags = load_split(args.split)
    else:
        return _die("pass --split {dev,test,all} or --fragments ...")
    if args.limit:
        frags = frags[: args.limit]
    variants = list(args.variants)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in phases:
        if p not in ("enh", "asr"):
            return _die(f"unknown phase {p!r} (valid: enh, asr)")

    cfg = build_config(args.arm)

    test_frags = sorted(set(frags) & set(load_split("test")))
    if test_frags:
        # Not a gate: FLOOR_GEN3_PLAN.md schedules WP1 over all 141 fragments as
        # pure diagnosis on existing test outputs (guardrail 1), and the arms are
        # declared NON-ADOPTABLE (guardrail 2). Loud so the exposure is on record.
        print(f"[NOTE] {len(test_frags)} FROZEN-TEST fragment(s) included. WP1 arms "
              f"are reference FLOORS, non-adoptable per FLOOR_GEN3_PLAN.md.")

    print(f"mixref_enhanced: arm={args.arm}  fragments={len(frags)}  "
          f"variants={variants}  phases={phases}  force={args.force}")
    print(f"  eval root : {root}")
    print(f"  debug log : {os.environ['ASR_PIPELINE_DEBUG_LOG']}")
    print(f"  enhancement: {cfg.enhancement.backend} "
          f"(OA {[VARIANTS[v] for v in variants]}, "
          f"max_segment_length_s={cfg.enhancement.max_segment_length_s}, "
          f"resample={cfg.enhancement.resample_quality})")
    print(f"  transcription: {cfg.transcription.backend} {cfg.transcription.model_name} "
          f"lang={cfg.transcription.language} loop_retry={cfg.transcription.loop_retry} "
          f"loop_retry_phrase={cfg.transcription.loop_retry_phrase} "
          f"retry_collapsed_chunk_size={cfg.transcription.retry_collapsed_chunk_size}")
    print(f"  deterministic={cfg.deterministic} device={cfg.device}")

    timer = _StageTimer()
    pipe = Pipeline(cfg, on_event=timer)     # applies the determinism flags
    _gpu_cap()
    failures: list = []
    t0 = time.perf_counter()
    enh_stats = asr_stats = None
    try:
        if "enh" in phases:
            print(f"\n=== phase 1/2: enhancement ({cfg.enhancement.backend}) ===")
            enh_stats = phase_enhance(pipe, cfg, args.arm, timer, frags, variants,
                                      root, args.force, failures)
        if "asr" in phases:
            print(f"\n=== phase 2/2: transcription "
                  f"({cfg.transcription.model_name}) ===")
            asr_stats = phase_transcribe(pipe, cfg, timer, frags, variants, root,
                                         args.force, failures)
    finally:
        pipe.unload()
    wall = time.perf_counter() - t0

    print(f"\n--- summary ({wall/60:.1f} min wall) ---")
    for name, st in (("enhance", enh_stats), ("transcribe", asr_stats)):
        if st is None:
            continue
        per = f"{st['run_s'] / st['done']:.2f}s" if st["done"] else "n/a"
        print(f"  {name:10s} done={st['done']:4d} skipped={st['skipped']:4d}  "
              f"run={st['run_s']/60:6.1f} min  per-unit={per:>7s}  "
              f"audio={st['audio_s']/60:6.1f} min")
    _write_failures(failures)
    return 1 if failures else 0


def _die(msg: str) -> int:
    print(f"mixref_enhanced: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
