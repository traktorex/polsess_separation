"""CLI entry point.

Usage::

    python -m asr_pipeline run   --config <yaml> --input <wav> [--set k=v ...]
                                 [--output <spill_dir>] [--write-outputs <eval_root>]
    python -m asr_pipeline batch --split clarin_test [--mode no_enh] [--set k=v ...]
    python -m asr_pipeline batch --manifest paths.txt --out-root <dir> [--mode ...]
    python -m asr_pipeline score --eval-root <dir> --out-dir <dir>

``run`` processes ONE recording (fail-loud, dies on error — SCOPE §4.2 single).
``--output`` enables the legacy per-stage intermediate spill; ``--write-outputs``
materialises the per-recording eval layout under ``<eval_root>/<id>/pipeline/``
and copies the mixture to ``<eval_root>/<id>/<id>.wav`` for eval discovery.

``batch`` processes MANY recordings (a split / manifest / glob / explicit list)
via `asr_pipeline.batch.run_batch`: one recording's failure is recorded in
``failures.csv`` and the batch continues. ``--mode`` selects an ablation preset
(full / no_sep / no_enh / minimal) written into the eval-tree subdir the L3 table
reads. The base config is the eval preset (``fresh_eval_cfg``: full_length +
transcribe_mixture + min_overlap_dur=0) so the outputs are directly L3-scorable —
the same policy the scripts this subsumes used.

``score`` walks an eval tree, runs the two-layer eval, and writes the summary
tables (L2 intrusive / L2 SQUIM / L3) as CSVs — notebook-free scoring.

``--set`` overrides config knobs (repeatable, YAML-typed values; shared fail-loud
policy with the sweep registry via `apply_overrides`). A preflight check (env vars
/ checkpoints the configured backends need) runs BEFORE any model load on ``run``
and ``batch``, so a missing ``$SORTFORMER_VENV_PY`` / checkpoint fails in seconds
instead of after minutes of loading (SCOPE §4).
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
import time
from pathlib import Path

from asr_pipeline.batch import MODE_PRESETS, run_batch, write_run_outputs
from asr_pipeline.config import (
    apply_overrides,
    load_pipeline_config_from_yaml,
    parse_cli_overrides,
)
from asr_pipeline.pipeline import Pipeline
from asr_pipeline.preflight import check_preflight

# Package-shipped base config for `batch` (matches the scripts it subsumes).
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


class _StageProgress:
    """``on_event`` sink: prints per-stage progress + records stage timings.

    Wired as ``Pipeline(cfg, on_event=self)``. Each completed stage contributes
    one ``{"stage", "load_s", "run_s"}`` row to ``self.timings``, which the run
    outputs (``run_meta.json`` + ``metadata.json``) record.
    """

    def __init__(self) -> None:
        self.timings: list = []

    def __call__(self, event: dict) -> None:
        kind = event.get("event")
        stage = event.get("stage", "?")
        if kind == "stage_start":
            print(f"[stage] {stage} — running...", flush=True)
        elif kind == "stage_end":
            load_s = float(event.get("load_s", 0.0))
            run_s = float(event.get("run_s", 0.0))
            print(
                f"[stage] {stage} — done "
                f"(load {load_s:.1f}s, run {run_s:.1f}s)",
                flush=True,
            )
            self.timings.append(
                {"stage": stage, "load_s": load_s, "run_s": run_s}
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asr_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---------------------------------------------------------------
    run = sub.add_parser("run", help="Run the pipeline on one recording.")
    run.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    run.add_argument("--input", required=True, help="Path to input audio file.")
    run.add_argument(
        "--output",
        default=None,
        help="Legacy per-stage spill directory (intermediates). Enables spill.",
    )
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="stage.knob=value",
        help="Override a config knob (repeatable). Value is YAML-typed: "
             "true/false -> bool, 0.5 -> float, quoted or bare text -> str.",
    )
    run.add_argument(
        "--write-outputs",
        dest="write_outputs",
        default=None,
        metavar="EVAL_ROOT",
        help="Write the per-recording eval layout under <EVAL_ROOT>/<id>/ and "
             "copy the mixture to <EVAL_ROOT>/<id>/<id>.wav (for eval discovery).",
    )

    # --- batch -------------------------------------------------------------
    batch = sub.add_parser(
        "batch",
        help="Run the pipeline over many recordings (split / manifest / glob / list).",
    )
    batch.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="Base pipeline YAML (eval preset applied on top). Default: the "
             "package default.yaml.",
    )
    batch.add_argument(
        "--mode", choices=list(MODE_PRESETS), default="full",
        help="Ablation preset: full | no_sep | no_enh | minimal. Selects the "
             "eval-tree output subdir the L3 table reads.",
    )
    batch.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="stage.knob=value",
        help="Override a config knob (repeatable, YAML-typed). Applied after the "
             "--mode preset, so an explicit --set wins on a conflict.",
    )
    batch.add_argument(
        "--out-root", default=None,
        help="Root the per-recording <id>/ dirs are written under. Required "
             "for --manifest/--glob/--inputs; defaults to the eval-tree root "
             "for --split.",
    )
    src = batch.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--split", choices=["clarin_dev", "clarin_test"],
        help="A frozen fragment split (reuses scripts/eval_harness loaders); "
             "inputs read from <eval-root>/<id>/<id>.wav.",
    )
    src.add_argument("--manifest", help="File with one audio path per line (# comments ok).")
    src.add_argument("--glob", help="Glob pattern selecting the input audio files.")
    src.add_argument("--inputs", nargs="+", help="Explicit list of audio file paths.")
    batch.add_argument("--force", action="store_true", help="Re-run even if complete.")

    # --- score -------------------------------------------------------------
    score = sub.add_parser(
        "score",
        help="Walk an eval tree, run L2+L3 eval, write the summary CSVs.",
    )
    score.add_argument(
        "--eval-root", required=True,
        help="Root passed to walk_eval_tree (yields <eval-root>/<dataset>/<id>/).",
    )
    score.add_argument(
        "--dataset", default=None,
        help="Restrict to one dataset dir under --eval-root (default: all).",
    )
    score.add_argument(
        "--out-dir", required=True,
        help="Directory the summary CSVs are written to.",
    )
    return parser


def _run_command(args) -> int:
    config = load_pipeline_config_from_yaml(args.config)
    if args.overrides:
        apply_overrides(config, parse_cli_overrides(args.overrides))
    if args.output is not None:
        config.artifact_dir = args.output
        config.spill_intermediate = True
        config.__post_init__()   # re-validate now that spill settings changed

    # Preflight BEFORE any model load (A2 / SCOPE §4): a missing
    # $SORTFORMER_VENV_PY, gated HF token, or absent checkpoint should fail in
    # seconds, not after minutes of loading.
    check_preflight(config)

    progress = _StageProgress()
    pipeline = Pipeline(config, on_event=progress)
    print(pipeline)
    t0 = time.perf_counter()
    ctx = pipeline.run(args.input)
    total_seconds = time.perf_counter() - t0
    print(f"pipeline finished in {total_seconds:.1f}s")

    if args.write_outputs is not None:
        pipeline_dir = write_run_outputs(
            ctx, config, args.write_outputs, progress.timings, total_seconds,
        )
        print(f"wrote pipeline outputs to {pipeline_dir}")
    return 0


def _resolve_batch_sources(args):
    """``args`` → ``(recordings, out_root)`` for `run_batch`.

    Recording sources (mutually exclusive at the argparse level): a frozen split
    name, a manifest file, a directory glob, or an explicit list of paths. The
    split path reuses the harness loaders (`scripts.eval_harness`) rather than
    reimplementing the frozen-list read; it is imported lazily so the other three
    sources never touch the parent-repo scripts package.
    """
    if args.split:
        from scripts.eval_harness import eval_root as _eval_root, load_split
        ids = load_split(args.split.removeprefix("clarin_"))
        root = Path(args.out_root).expanduser() if args.out_root else _eval_root()
        recordings = [(rid, root / rid / f"{rid}.wav") for rid in ids]
        return recordings, root

    if args.out_root is None:
        raise SystemExit("--out-root is required for --manifest/--glob/--inputs.")
    out_root = Path(args.out_root).expanduser()

    if args.manifest:
        lines = Path(args.manifest).expanduser().read_text(encoding="utf-8").splitlines()
        paths = [Path(ln.strip()).expanduser() for ln in lines
                 if ln.strip() and not ln.strip().startswith("#")]
    elif args.glob:
        paths = [Path(p) for p in sorted(_glob.glob(str(Path(args.glob).expanduser())))]
    else:  # args.inputs (argparse guarantees one source was given)
        paths = [Path(p).expanduser() for p in args.inputs]
    return paths, out_root


def _batch_command(args) -> int:
    # Base = the eval preset (full_length + transcribe_mixture + min_overlap_dur=0)
    # so the outputs are directly L3-scorable — the policy the scripts this
    # subsumes (run_pipeline_on_recording / batch_pipeline_noenh) all used.
    from asr_pipeline.eval.config_presets import fresh_eval_cfg

    subdir_name, mode_overrides = MODE_PRESETS[args.mode]
    config = fresh_eval_cfg(Path(args.config).expanduser())
    # --mode preset first, explicit --set second (an explicit --set wins).
    overrides = {**mode_overrides, **parse_cli_overrides(args.overrides)}
    apply_overrides(config, overrides)

    recordings, out_root = _resolve_batch_sources(args)
    check_preflight(config)   # fail loud before any model load

    print(f"batch: mode={args.mode} subdir={subdir_name} "
          f"out_root={out_root} n={len(recordings)}")
    report = run_batch(
        config, recordings, out_root, subdir_name,
        skip_existing=not args.force,
    )
    print(f"batch done: {len(report.succeeded)} run, "
          f"{len(report.skipped)} skipped, {len(report.failed)} failed")
    if report.failed:
        print(f"  failures recorded in {report.failures_csv}")
    return 0 if not report.failed else 1


def _score_command(args) -> int:
    # Import-only consumption of the eval public API (a parallel agent owns the
    # eval package internals; these interfaces stay byte-compatible). Lazy so a
    # `run`/`batch` invocation never imports the eval package.
    from asr_pipeline.eval import (
        evaluate_many,
        summarize_layer2_intrusive,
        summarize_layer2_squim,
        summarize_layer3,
        walk_eval_tree,
    )

    root = Path(args.eval_root).expanduser()
    recordings = list(walk_eval_tree(root, dataset=args.dataset))
    if not recordings:
        print(f"no recordings found under {root}"
              + (f" (dataset={args.dataset})" if args.dataset else ""),
              file=sys.stderr)
        return 1
    print(f"scoring {len(recordings)} recording(s) under {root}...")
    cards = evaluate_many(recordings)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Do NOT hardcode a column list — write whatever each summarizer returns (a
    # parallel agent is adding cpCER columns to summarize_layer3).
    tables = {
        "layer2_intrusive": summarize_layer2_intrusive(cards),
        "layer2_squim": summarize_layer2_squim(cards),
        "layer3": summarize_layer3(cards),
    }
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows, {len(df.columns)} cols)")
    return 0


_DISPATCH = {
    "run": _run_command,
    "batch": _batch_command,
    "score": _score_command,
}


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
