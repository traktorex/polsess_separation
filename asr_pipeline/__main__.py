"""CLI entry point.

Usage:
    python -m asr_pipeline run --config <yaml> --input <wav> [--output <dir>]

`--output` enables intermediate spill and points the artefact directory.
"""

from __future__ import annotations

import argparse
import sys

from asr_pipeline.config import load_pipeline_config_from_yaml
from asr_pipeline.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asr_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline on one recording.")
    run.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    run.add_argument("--input", required=True, help="Path to input audio file.")
    run.add_argument(
        "--output",
        default=None,
        help="Artefact directory. If set, intermediates are spilled here.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "run":
        # argparse already errors on missing subcommand, but be explicit.
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2

    config = load_pipeline_config_from_yaml(args.config)
    if args.output is not None:
        config.artifact_dir = args.output
        config.spill_intermediate = True
        # Re-run validation now that we mutated the spill settings.
        config.__post_init__()

    pipeline = Pipeline(config)
    print(pipeline)
    pipeline.run(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
