#!/usr/bin/env python
"""Split-disjointness audit — proves train never shares source audio with val/test.

WHAT THIS PROVES
----------------
The PolSESS mixtures in each split are rendered from source recordings named in
the split's corpus CSV (``corpus_<name>_<subset>_final.csv``). The columns ending
in ``OryginalPath`` (sic — the corpus keeps the original Polish-influenced
spelling) hold the provenance of each layer's *source* file:

    speech1OryginalPath, speech2OryginalPath, sceneOryginalPath, eventOryginalPath

A speaker/scene/event source that appears in both train and test would leak the
test distribution into training. This script computes the pairwise set
intersection of every ``OryginalPath`` column across the three splits and, in
addition, the intersection of the *combined speaker source set*
(speech1 ∪ speech2) — because the same speaker recording can appear as
``speech1`` in one row and ``speech2`` in another, a column-by-column check alone
would miss a speaker that swapped roles between splits.

    * ANY train↔val or train↔test overlap => hard failure (non-zero exit).
    * val↔test overlap is reported as INFORMATIONAL: the two eval splits are
      known to share a pool of scene source files (harmless for test validity —
      neither leaks into training — but worth documenting).

HOW TO CITE
-----------
    scripts/audit_split_leakage.py — proves zero train↔{val,test} source-audio
    overlap on every provenance column of PolSESS_C_new_64, pre-empting the
    reviewer's first dataset-integrity question with numbers.

USAGE
-----
    python scripts/audit_split_leakage.py
    python scripts/audit_split_leakage.py --data-root /path/to/PolSESS_C_new_64/...

Pure pandas, no audio IO. Exits non-zero on any train↔{val,test} overlap.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PolSESSParams  # noqa: E402

# Columns holding source-file provenance. The corpus spells it "OryginalPath".
ORIGINAL_PATH_RE = re.compile(r"OryginalPath$")

# The two speaker-source columns are pooled for the role-swap-aware speaker check.
SPEAKER_COLUMNS = ["speech1OryginalPath", "speech2OryginalPath"]


def find_original_path_columns(df: pd.DataFrame) -> list:
    """Return the corpus columns holding source-file provenance."""
    return [c for c in df.columns if ORIGINAL_PATH_RE.search(c)]


def source_set(df: pd.DataFrame, columns) -> set:
    """Union of non-null string values across one or more columns."""
    if isinstance(columns, str):
        columns = [columns]
    values = set()
    for col in columns:
        if col in df.columns:
            values |= set(df[col].dropna().astype(str))
    return values


def load_corpora(data_root: Path, splits) -> dict:
    """Load each split's corpus CSV. Raises FileNotFoundError if one is missing."""
    corpus_name = data_root.name
    corpora = {}
    for split in splits:
        csv_path = data_root / split / f"corpus_{corpus_name}_{split}_final.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Corpus CSV not found: {csv_path}")
        corpora[split] = pd.read_csv(csv_path)
    return corpora


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit train/val/test source-audio disjointness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=PolSESSParams().data_root,
        help="PolSESS dataset root (defaults to POLSESS_DATA_ROOT / config default).",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    splits = ["train", "val", "test"]
    try:
        corpora = load_corpora(data_root, splits)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Split-disjointness audit")
    print("=" * 72)
    print(f"data root : {data_root}")
    for split in splits:
        print(f"  {split:5s}: {len(corpora[split]):>6d} rows")

    columns = find_original_path_columns(corpora["train"])
    print(f"\nProvenance columns audited: {columns}")
    print(f"Speaker role-swap-aware check pools: {SPEAKER_COLUMNS}")

    # Build the checkable set collection: each OriginalPath column individually,
    # plus a pooled speaker set (speech1 ∪ speech2).
    check_specs = [(c, c) for c in columns]
    check_specs.append(("speech(any)=speech1∪speech2", SPEAKER_COLUMNS))

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    print("\nPairwise source-set intersections (|A ∩ B|):")
    header = f"  {'column':<32}" + "".join(f"{a[:2]}∩{b[:2]:<8}" for a, b in pairs)
    print(header)
    print("  " + "-" * (32 + 12 * len(pairs)))

    violations = []          # train↔{val,test} overlaps -> hard fail
    informational = []       # val↔test overlaps -> documented, not a failure

    for label, cols in check_specs:
        sets = {s: source_set(corpora[s], cols) for s in splits}
        cells = []
        for a, b in pairs:
            inter = sets[a] & sets[b]
            cells.append(len(inter))
            if inter:
                if a == "train" or b == "train":
                    violations.append((label, a, b, inter))
                else:
                    informational.append((label, a, b, inter))
        row = f"  {label:<32}" + "".join(f"{n:<12d}" for n in cells)
        print(row)

    print("\n" + "-" * 72)
    if informational:
        print("INFORMATIONAL — val↔test source sharing (expected; does NOT leak "
              "into training):")
        for label, a, b, inter in informational:
            example = sorted(inter)[0]
            print(f"  {label}: {a}∩{b} = {len(inter)} shared "
                  f"(e.g. {Path(str(example)).name})")
    else:
        print("INFORMATIONAL — no val↔test source sharing found.")

    print("\n" + "=" * 72)
    if violations:
        print("RESULT: FAIL — train shares source audio with an eval split:")
        for label, a, b, inter in violations:
            print(f"  {label}: {a}∩{b} = {len(inter)} overlapping source(s)")
        return 1

    print("RESULT: PASS — zero train↔val and train↔test source-audio overlap "
          "on every provenance column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
