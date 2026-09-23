"""Shared eval-harness paths + small readers for the ASR scoring scripts.

One home for three things that used to be pasted across the scoring scripts
(``rescore_stratified``, ``dump_sweep_results``, ``sweep_pipeline``,
``score_attribution_purity``, ``compare_asr``):

- the eval-tree root (``<root>/<frag_id>/...``), env-overridable so a non-default
  tree or a test fixture can be pointed at without editing code;
- the frozen fragment-split loader (``asr_pipeline/eval/clarin_<split>.txt``),
  which errors loudly on a missing or empty list;
- the attribution-purity CSV reader.

Keeping them here means every script resolves the SAME paths and raises the SAME
error on an empty/missing split. No heavy deps (no torch), so it stays cheap to
import from every harness.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Env override so a non-default eval tree (or a test fixture) can be pointed at
# without editing code; unset → the committed default, so behaviour is unchanged.
_EVAL_ROOT_ENV = "ASR_EVAL_ROOT"
_DEFAULT_EVAL_ROOT = "~/datasets/eval/clarin_fragments"


def eval_root() -> Path:
    """The eval-tree root (``<root>/<frag_id>/...``). Override with
    ``$ASR_EVAL_ROOT``; default ``~/datasets/eval/clarin_fragments`` (unchanged
    from the hardcoded literals this replaces)."""
    return Path(os.environ.get(_EVAL_ROOT_ENV, _DEFAULT_EVAL_ROOT)).expanduser()


def load_split(split: str | None = None, frag_file: str | None = None) -> list[str]:
    """The active fragment list — a named split or an explicit file.

    ``frag_file`` (when given) overrides ``split``; otherwise the frozen list at
    ``asr_pipeline/eval/clarin_<split>.txt``. Both the space-separated dev list and
    the newline-separated test list ``.split()`` cleanly. Exits loudly on a missing
    or empty file (the same behaviour the scripts had inline)."""
    if frag_file:
        path = Path(frag_file).expanduser()
    else:
        path = REPO / "asr_pipeline" / "eval" / f"clarin_{split}.txt"
    if not path.exists():
        sys.exit(f"fragment list not found: {path}")
    frags = path.read_text().split()
    if not frags:
        sys.exit(f"no fragments in {path}")
    return frags


def load_purity(root: Path | None = None) -> dict:
    """``{(frag_id, config): (pure, total)}`` from ``score_attribution_purity.py``'s
    CSV, or ``{}`` if absent (the purity table/Δ are then omitted; the WER/CER gaps
    still print). Window counts micro-average like errors/length."""
    path = (root or eval_root()) / "_attribution_purity.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["frag_id"], r["config"])] = (int(r["pure"]), int(r["total"]))
    return out
