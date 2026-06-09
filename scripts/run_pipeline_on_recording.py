"""Run the ASR pipeline against one recording in three ablation modes.

Drives the WER-ablation table the eval module's Layer 3 reads:

    pipeline/         full pipeline (default config + transcribe_mixture)
    pipeline_nosep/   separation.enabled = false
    pipeline_noenh/   enhancement.enabled = false

Phase-major: each mode runs as a fresh ``Pipeline``; the previous pipeline
is dropped before the next mode starts so GPU memory is fully released
between modes. The three modes write their outputs to per-mode subdirs
under ``<eval_root>/<dataset>/<recording_id>/``.

Usage::

    python scripts/run_pipeline_on_recording.py \
        --recording-dir ~/datasets/eval/clarin/442dd69e

The recording's ``<id>.wav`` (or legacy ``mixture.wav``) is the input;
the writer places outputs beside it. The driver doesn't touch the ``reference/`` subdir (that's
``scripts/prepare_eval_references.py``'s job).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline import Pipeline                                # noqa: E402
from asr_pipeline.config import (                                # noqa: E402
    PipelineConfig,
    load_pipeline_config_from_yaml,
)
from asr_pipeline.io import write_pipeline_outputs               # noqa: E402


def _disable_both(cfg: "PipelineConfig") -> None:
    cfg.separation.enabled = False
    cfg.enhancement.enabled = False


def _nosep_with_mossformer(cfg: "PipelineConfig") -> None:
    cfg.separation.enabled = False
    cfg.enhancement.backend = "mossformer_gan_se_16k"


# Ablation / bootstrap modes: (subdir name, override-applier).
#
# - pipeline:          full chain
# - pipeline_nosep:    enhancement on, separation off (ablation row)
# - pipeline_noenh:    enhancement off, separation on (ablation row)
# - pipeline_minimal:  both off — diarize + slice + transcribe on the raw
#                      mixture. Used for GT bootstrap: enhancement
#                      sometimes suppresses the quieter speaker in
#                      overlap, which would propagate into the
#                      hand-corrected GT. With both off, the audio in
#                      each per-speaker stream is verbatim raw mixture
#                      content sliced by diarization. Also doubles as
#                      the strictest ablation baseline ("what if we
#                      only diarize?").
MODES: list[tuple[str, callable]] = [
    ("pipeline",                    lambda cfg: None),
    ("pipeline_nosep",              lambda cfg: setattr(cfg.separation,  "enabled", False)),
    ("pipeline_noenh",              lambda cfg: setattr(cfg.enhancement, "enabled", False)),
    ("pipeline_minimal",            _disable_both),
    # GT-bootstrap candidate: keep enhancement to recover the quieter
    # speaker during overlap, but switch backend from the default MP-SENet
    # to MossFormerGAN (less aggressive suppression of the non-dominant
    # speaker in mixed regions).
    ("pipeline_nosep_mossformer",   _nosep_with_mossformer),
]


def _fresh_cfg(yaml_path: Path) -> PipelineConfig:
    """Load default.yaml + force eval-friendly overrides.

    - ``transcribe_mixture=True`` so L3 ORC-WER has the single-stream baseline.
    - ``output_mode='full_length'`` so per-speaker streams stay on the
      mixture timeline. Transcripts produced from these streams have
      timestamps that line up with the original recording — required for
      tcpWER (time-constrained WER) scoring and for hand-correcting GT
      against the source audio.
    - ``routing.min_overlap_dur=0`` so every pyannote-detected overlap
      goes through the separator (no quiet backchannels get dropped at
      routing time). Eval should be apples-to-apples across modes; we
      let the separator decide what to do with short overlaps.
    """
    cfg = load_pipeline_config_from_yaml(str(yaml_path))
    cfg.transcription.transcribe_mixture = True
    cfg.assembly.output_mode = "full_length"
    cfg.routing.min_overlap_dur = 0.0
    return cfg


def _drop_pipeline(p: Pipeline) -> None:
    """Free GPU references held by the pipeline + its stages."""
    p.unload()
    del p
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_mode(
    mixture_path: Path, recording_dir: Path, subdir: str,
    apply_overrides, yaml_path: Path, skip_existing: bool,
) -> tuple[bool, float]:
    """Run one mode. Returns (ran, elapsed_seconds). ``ran=False`` when
    skipped because the target dir already exists."""
    target_dir = recording_dir / subdir
    if skip_existing and target_dir.exists() and any(target_dir.iterdir()):
        print(f"  [{subdir}] skip — {target_dir} already populated")
        return False, 0.0

    print(f"  [{subdir}] starting...")
    t0 = time.perf_counter()
    cfg = _fresh_cfg(yaml_path)
    apply_overrides(cfg)
    cfg.__post_init__()

    p = Pipeline(cfg)
    try:
        ctx = p.run(str(mixture_path))
        write_pipeline_outputs(
            ctx, recording_dir,
            config_snapshot=asdict(cfg),
            subdir_name=subdir,
        )
    finally:
        _drop_pipeline(p)
    elapsed = time.perf_counter() - t0
    print(f"  [{subdir}] done in {elapsed:.1f}s")
    return True, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recording-dir", type=Path, required=True,
        help="Per-recording dir, e.g. ~/datasets/eval/clarin/442dd69e/. "
             "Must contain <id>.wav or mixture.wav.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml",
        help="Base pipeline config (overrides applied per mode).",
    )
    parser.add_argument(
        "--modes", nargs="+", default=[m[0] for m in MODES],
        choices=[m[0] for m in MODES],
        help="Which ablation modes to run (default: all three).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a mode if its output dir already contains files.",
    )
    args = parser.parse_args()

    recording_dir = args.recording_dir.expanduser().resolve()
    # Same resolution order as asr_pipeline.eval.recordings.load_recording:
    # new convention is `<dir>/<dir.name>.wav`, legacy is `mixture.wav`.
    mixture = recording_dir / f"{recording_dir.name}.wav"
    if not mixture.exists():
        mixture = recording_dir / "mixture.wav"
    if not mixture.exists():
        print(
            f"error: neither {recording_dir.name}.wav nor mixture.wav "
            f"found in {recording_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"recording: {recording_dir}")
    print(f"mixture:   {mixture}")
    print(f"config:    {args.config}")
    print(f"modes:     {args.modes}")
    print()

    requested = {m: o for m, o in MODES if m in args.modes}
    if not requested:
        print("no modes selected", file=sys.stderr)
        return 1

    total_t = 0.0
    for subdir, apply_overrides in MODES:
        if subdir not in requested:
            continue
        ran, elapsed = run_one_mode(
            mixture, recording_dir, subdir, apply_overrides,
            args.config, args.skip_existing,
        )
        total_t += elapsed

    print(f"\ntotal pipeline runtime: {total_t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
