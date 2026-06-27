"""Rebuild ``GT_REVIEW_INDEX.md`` from the per-fragment ``gt_review/review.md`` files.

The second-pass GT QA aid writes one ``review.md`` per fragment under
``<drive>/<frag>/gt_review/review.md`` (bundle inputs come from
``scripts/build_gt_review_inputs.py``; the per-fragment eyeball pass produces the
``review.md`` itself). This script walks those reviews, parses each one's
severity counts, flag types, and one-line ``Overall`` summary, and regenerates
the top-level roll-up index — ranked by *attention score* (HIGH×100 + MED×10 +
LOW) so the fragments most worth a re-listen float to the top.

Idempotent: safe to re-run after every batch of fragments is reviewed (e.g. as
the delegated fragments come back). The previous index is backed up to
``GT_REVIEW_INDEX.md.bak`` before each write.

Usage::

    python scripts/build_gt_review_index.py                          # /mnt/f/clarin_fragments
    python scripts/build_gt_review_index.py --drive-root /some/where
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

DROVE = "/mnt/f/clarin_fragments"
KNOWN_TYPES = (
    "WORD_MISMATCH", "NZR_CANDIDATE", "ORTHOGRAPHY", "FORMATTING",
    "MISSED_SPEECH", "ATTRIBUTION_Q", "SEGMENTATION",
)

_FLAGS_RE = re.compile(
    r"\*\*Flags:\*\*\s*(\d+)\s*HIGH\s*/\s*(\d+)\s*MED\s*/\s*(\d+)\s*LOW")
_OVERALL_RE = re.compile(r"(?m)^\*\*Overall:\*\*\s*(.+?)\s*$")
_TYPE_RE = re.compile(r"\[([A-Z_]+)\s*@")


@dataclass
class Review:
    frag: str
    high: int
    med: int
    low: int
    overall: str
    types: dict[str, int]

    @property
    def attention(self) -> int:
        return self.high * 100 + self.med * 10 + self.low


def parse_review(path: Path) -> Review | None:
    """Parse one review.md; return None (with a warning) if it is malformed."""
    frag = path.parent.parent.name  # <frag>/gt_review/review.md
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = _FLAGS_RE.search(text)
    om = _OVERALL_RE.search(text)
    if not fm or not om:
        print(f"  [warn] {frag}: missing Flags/Overall line — skipped")
        return None
    types = {t: 0 for t in KNOWN_TYPES}
    for t in _TYPE_RE.findall(text):
        if t in types:
            types[t] += 1
    return Review(frag, int(fm[1]), int(fm[2]), int(fm[3]), om[1].strip(), types)


def _cell(text: str) -> str:
    """Escape a string for safe inclusion in a Markdown table cell."""
    return (text.replace("|", r"\|")
                .replace("<", r"\<").replace(">", r"\>"))


def render_index(reviews: list[Review]) -> str:
    n = len(reviews)
    total = sum(r.high + r.med + r.low for r in reviews)
    H = sum(r.high for r in reviews)
    M = sum(r.med for r in reviews)
    L = sum(r.low for r in reviews)
    with_high = sum(1 for r in reviews if r.high > 0)
    type_tot: dict[str, int] = {t: 0 for t in KNOWN_TYPES}
    for r in reviews:
        for t, c in r.types.items():
            type_tot[t] += c

    out: list[str] = []
    out.append("# GT Review Index — Second-Pass QA Aid\n")
    out.append(
        f"This index rolls up {n} per-fragment GT review reports. It is a "
        "**second-pass QA aid**: the ASR readings (lv2/lv3) are a *guess menu*, "
        "not ground truth — every flag must be **verified by ear**. The GT was "
        "seeded from lv2, so an independent **lv3 divergence is the signal** "
        "worth attending to; lv2 agreement with the GT carries little "
        "information.\n")
    out.append("## Aggregate stats\n")
    out.append(f"- Fragments reviewed: **{n}**")
    out.append(f"- Total flags: **{total}**")
    out.append(f"- By severity: **HIGH {H}** / **MED {M}** / **LOW {L}**")
    out.append(f"- Fragments with >=1 HIGH flag: **{with_high}**\n")
    out.append("### Flags by type\n")
    out.append("| Type | Count |")
    out.append("|------|------:|")
    for t, c in sorted(type_tot.items(), key=lambda kv: (-kv[1], kv[0])):
        if c:
            out.append(f"| {t} | {c} |")
    out.append("")
    out.append("## All fragments (ranked by attention score)\n")
    out.append("Attention score = HIGH×100 + MED×10 + LOW. Clean fragments sink "
               "to the bottom.\n")
    out.append("| Fragment | HIGH | MED | LOW | Overall | Review |")
    out.append("|----------|-----:|----:|----:|---------|--------|")
    for r in sorted(reviews, key=lambda x: (-x.attention, x.frag)):
        link = f"[{r.frag}/gt_review/review.md]({r.frag}/gt_review/review.md)"
        out.append(f"| `{r.frag}` | {r.high} | {r.med} | {r.low} | "
                   f"{_cell(r.overall)} | {link} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", default=DROVE)
    args = ap.parse_args(argv)
    drive_root = Path(args.drive_root)
    if not drive_root.is_dir():
        raise SystemExit(f"drive root not mounted: {drive_root}")

    paths = sorted(drive_root.glob("*/gt_review/review.md"))
    reviews = [r for p in paths if (r := parse_review(p)) is not None]
    if not reviews:
        raise SystemExit("no review.md files found")

    index_path = drive_root / "GT_REVIEW_INDEX.md"
    if index_path.exists():
        (drive_root / "GT_REVIEW_INDEX.md.bak").write_text(
            index_path.read_text(encoding="utf-8"), encoding="utf-8")
    index_path.write_text(render_index(reviews), encoding="utf-8")

    ranked = sorted(reviews, key=lambda x: (-x.attention, x.frag))
    rank = {r.frag: i + 1 for i, r in enumerate(ranked)}
    print(f"[index] {len(reviews)} fragments -> {index_path}")
    print(f"[index] HIGH {sum(r.high for r in reviews)} / "
          f"MED {sum(r.med for r in reviews)} / "
          f"LOW {sum(r.low for r in reviews)}; "
          f"{sum(1 for r in reviews if r.high)} with >=1 HIGH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
