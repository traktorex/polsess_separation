"""Run the ASR pipeline against one recording in its ablation modes.

Thin wrapper over `asr_pipeline.batch.run_batch` (the shared batch runner). Drives
the WER-ablation table the eval module's Layer 3 reads:

    pipeline/          full pipeline (default config + transcribe_mixture)
    pipeline_nosep/    separation.enabled = false
    pipeline_noenh/    enhancement.enabled = false
    pipeline_minimal/  both off — diarize + slice + transcribe only

(plus the GT-bootstrap-only ``pipeline_nosep_mossformer``, which L3 does
not score). Phase-major: each mode is a fresh ``Pipeline`` built inside
``run_batch``; the previous pipeline's GPU memory is fully released between
modes by the shared teardown. Modes write their outputs to per-mode subdirs
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
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.batch import run_batch                          # noqa: E402
from asr_pipeline.config import PipelineConfig                    # noqa: E402
from asr_pipeline.eval.config_presets import fresh_eval_cfg       # noqa: E402

# Back-compat alias: the shared eval-config preset used to live here as
# `_fresh_cfg`. It now lives in `asr_pipeline.eval.config_presets` so the
# sweep driver imports the same policy without a cross-script hack.
_fresh_cfg = fresh_eval_cfg


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
#
# The full/no_sep/no_enh/minimal override sets are the same ones
# `asr_pipeline.batch.MODE_PRESETS` encodes as dotted dicts (a test cross-checks
# the two agree); the extra ``pipeline_nosep_mossformer`` GT-bootstrap mode is
# script-local. Kept as (name, applier) so the applier-state test keeps pinning
# each mode's (sep, enh) effect.
MODES: list[tuple[str, callable]] = [
    ("pipeline",                    lambda cfg: None),
    ("pipeline_nosep",              lambda cfg: setattr(cfg.separation,  "enabled", False)),
    ("pipeline_noenh",              lambda cfg: setattr(cfg.enhancement, "enabled", False)),
    ("pipeline_minimal",            _disable_both),
    # GT-bootstrap candidate: keep enhancement to recover the quieter
    # speaker during overlap, but switch backend from the default FRCRN
    # to MossFormerGAN (less aggressive suppression of the non-dominant
    # speaker in mixed regions).
    ("pipeline_nosep_mossformer",   _nosep_with_mossformer),
]


def run_one_mode(
    mixture_path: Path, recording_dir: Path, subdir: str,
    apply_mode, yaml_path: Path, skip_existing: bool,
) -> bool:
    """Run one mode via ``run_batch``. Returns ``ran`` (False when skipped).

    Writes into ``<recording_dir>/<subdir>/`` — the recording id is pinned to
    ``recording_dir.name`` (independent of the mixture filename, so a legacy
    ``mixture.wav`` still lands under the right dir). The mixture is the input
    already present in the dir, so it is not copied.
    """
    cfg = _fresh_cfg(yaml_path)
    apply_mode(cfg)
    cfg.__post_init__()
    report = run_batch(
        cfg,
        [(recording_dir.name, mixture_path)],
        out_root=recording_dir.parent,
        subdir_name=subdir,
        skip_existing=skip_existing,
        # Preserve the script's historical sentinel: a non-empty target dir means
        # done (metadata.json is a superset of that, so this never re-runs a tree
        # the newer runner would consider complete).
        is_complete=lambda d: d.exists() and any(d.iterdir()),
        copy_mixture=False,
    )
    return bool(report.succeeded)


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
        help="Which ablation modes to run (default: all).",
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

    requested = {m for m in args.modes}
    if not requested:
        print("no modes selected", file=sys.stderr)
        return 1

    t_total = time.perf_counter()
    for subdir, apply_mode in MODES:
        if subdir not in requested:
            continue
        print(f"[{subdir}]")
        run_one_mode(
            mixture, recording_dir, subdir, apply_mode,
            args.config, args.skip_existing,
        )

    print(f"\ntotal pipeline runtime: {time.perf_counter() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
