#!/usr/bin/env python
"""Generate a param-count manifest for the thesis architecture table.

WHAT THIS PROVES / PRODUCES
---------------------------
Instantiates one representative config per architecture / size variant used in
the thesis, on CPU, through the exact production path
(``load_config_from_yaml`` -> ``create_model_from_config``), and records the
parameter count of each. This turns the hand-transcribed counts scattered across
prose + YAML comments into a single regenerable artifact the thesis tables can
cite, and cross-checks the four documented counts (ConvTasNet 8.64M, DPRNN
2.61M, SepFormer 25.68M, MossFormer2 26.41M / 55.74M) so a silent drift shows up
as a MISMATCH here.

Only the parameter count is reported — the "mask vs. mapping" architecture-table
column (SPMamba/SPMamba3 are TF-domain *mapping* models; the rest mask) is thesis
prose and is deliberately NOT emitted here.

HOW TO CITE
-----------
    scripts/model_manifest.py -> docs/generated/model_manifest.{csv,md}
    Parameter counts obtained by CPU instantiation of the listed configs.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" python scripts/model_manifest.py
    CUDA_VISIBLE_DEVICES="" python scripts/model_manifest.py --output-dir docs/generated

Mamba-family models construct on CPU (only their forward pass needs CUDA). A
config that still fails CPU construction is recorded as "skipped: <reason>"
rather than aborting the run. Exits non-zero if any documented count MISMATCHes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config_from_yaml  # noqa: E402
from models.factory import create_model_from_config  # noqa: E402

# One representative config per architecture / size variant used in the thesis.
# (model_type, size_variant, repo-relative config path)
CONFIGS = [
    # ConvTasNet is listed for both tasks: the documented 8.64M is the ES
    # (single-source, C=1) enhancement variant, while the separation table needs
    # the SB (C=2) number (8.71M — the extra 65536 params are the mask 1x1-conv's
    # C*N output channels). Every OTHER model's documented count is already its
    # C=2 separation figure, so those get a single SB row.
    ("convtasnet",   "es (C=1)", "experiments/convtasnet/baseline.yaml"),
    ("convtasnet",   "sb (C=2)", "experiments/convtasnet/sb_task.yaml"),
    ("dprnn",        "default", "experiments/dprnn/dprnn_baseline.yaml"),
    ("sepformer",    "default", "experiments/sepformer/sepformer_baseline_positionalenc.yaml"),
    ("mossformer2",  "matched", "experiments/mossformer2/mossformer2_matched.yaml"),
    ("mossformer2",  "full",    "experiments/mossformer2/mossformer2_full.yaml"),
    ("spmamba",      "reduced", "experiments/spmamba/spmamba_sb_reduced.yaml"),
    ("spmamba",      "full",    "experiments/spmamba/spmamba_sb.yaml"),
    ("mamba_tasnet", "xs",      "experiments/mamba_tasnet/mamba_tasnet_xs.yaml"),
    ("mamba_tasnet", "s",       "experiments/mamba_tasnet/mamba_tasnet_s.yaml"),
    ("mamba_tasnet", "m",       "experiments/mamba_tasnet/mamba_tasnet_m.yaml"),
    ("mamba_tasnet", "l",       "experiments/mamba_tasnet/mamba_tasnet_l.yaml"),
    ("dpmamba",      "xs",      "experiments/dpmamba/dpmamba_xs.yaml"),
    ("dpmamba",      "s",       "experiments/dpmamba/dpmamba_s.yaml"),
    ("dpmamba",      "m",       "experiments/dpmamba/dpmamba_m.yaml"),
    ("dpmamba",      "l",       "experiments/dpmamba/dpmamba_l.yaml"),
]

# Documented counts (millions) from CLAUDE.md / thesis prose, for cross-check.
# Mamba/SPMamba families are documented only as ranges, so they carry no anchor.
DOCUMENTED_MILLIONS = {
    ("convtasnet", "es (C=1)"): 8.64,
    ("dprnn", "default"): 2.61,
    ("sepformer", "default"): 25.68,
    ("mossformer2", "matched"): 26.41,
    ("mossformer2", "full"): 55.74,
}

# Absolute tolerance (millions) for the documented cross-check — documented
# values are quoted to 2 decimals, so ~0.02M covers rounding without hiding drift.
MATCH_TOL_MILLIONS = 0.02


def out_source_count(config) -> int:
    """Read the model's output-source field (C, or n_srcs for SPMamba)."""
    params = getattr(config.model, config.model.model_type)
    return getattr(params, "n_srcs", None) or getattr(params, "C", None)


def build_rows():
    rows = []
    for model_type, size_variant, rel_path in CONFIGS:
        cfg_path = REPO_ROOT / rel_path
        doc = DOCUMENTED_MILLIONS.get((model_type, size_variant))
        base = {
            "model_type": model_type,
            "size_variant": size_variant,
            "config_path": rel_path,
            "documented_millions": doc,
        }
        if not cfg_path.exists():
            rows.append({**base, "task": None, "out_sources": None,
                         "param_count": None, "param_millions": None,
                         "status": "skipped: config not found"})
            continue
        try:
            config = load_config_from_yaml(str(cfg_path))
            model = create_model_from_config(config.model)
            n_params = sum(p.numel() for p in model.parameters())
            millions = n_params / 1e6
            if doc is not None and abs(millions - doc) > MATCH_TOL_MILLIONS:
                status = "MISMATCH"
            else:
                status = "ok"
            rows.append({
                **base,
                "task": config.data.task,
                "out_sources": out_source_count(config),
                "param_count": n_params,
                "param_millions": round(millions, 4),
                "status": status,
            })
        except Exception as exc:  # noqa: BLE001 — record, don't crash the manifest
            rows.append({**base, "task": None, "out_sources": None,
                         "param_count": None, "param_millions": None,
                         "status": f"skipped: {type(exc).__name__}: {exc}"})
    return rows


def write_markdown(df: pd.DataFrame, path: Path):
    lines = [
        "# Model parameter-count manifest",
        "",
        "Generated by `scripts/model_manifest.py` (CPU instantiation of the listed "
        "configs). Regenerate rather than hand-editing.",
        "",
        "| Architecture | Size | Params (M) | Out src | Task | Config | Documented (M) | Status |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for _, r in df.iterrows():
        pm = "" if pd.isna(r["param_millions"]) else f"{r['param_millions']:.2f}"
        oc = "" if pd.isna(r["out_sources"]) else int(r["out_sources"])
        doc = "" if pd.isna(r["documented_millions"]) else f"{r['documented_millions']:.2f}"
        task = "" if pd.isna(r["task"]) else r["task"]
        lines.append(
            f"| {r['model_type']} | {r['size_variant']} | {pm} | {oc} | {task} "
            f"| `{r['config_path']}` | {doc} | {r['status']} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- Param count depends on the output-source count `C`/`n_srcs`, which the "
        "task fixes (ES/EB=1, SB=2). The separation comparison uses C=2.",
        "- ConvTasNet's documented **8.64M** is the ES (C=1) enhancement variant; "
        "the C=2 separation config is **8.71M** (+65536 = 256*256 mask-output "
        "params). Every other model's documented count is already its C=2 figure.",
        "- The 'mask vs. mapping' architecture-table column is thesis prose "
        "(SPMamba is a TF-domain mapping model; the rest mask) and is not emitted "
        "here.",
        "",
    ])
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the thesis model param-count manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "generated"),
        help="Directory for model_manifest.{csv,md}.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    df = pd.DataFrame(rows, columns=[
        "model_type", "size_variant", "config_path", "task", "out_sources",
        "param_count", "param_millions", "documented_millions", "status",
    ])

    csv_path = out_dir / "model_manifest.csv"
    md_path = out_dir / "model_manifest.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)

    print("=" * 78)
    print("Model parameter-count manifest")
    print("=" * 78)
    for _, r in df.iterrows():
        pm = "  n/a " if pd.isna(r["param_millions"]) else f"{r['param_millions']:7.3f}M"
        doc = "" if pd.isna(r["documented_millions"]) else f" (doc {r['documented_millions']:.2f}M)"
        print(f"  {r['model_type']:<13} {r['size_variant']:<8} {pm}{doc:<14}  "
              f"[{r['status']}]")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")

    mismatches = df[df["status"] == "MISMATCH"]
    skipped = df[df["status"].astype(str).str.startswith("skipped")]
    if len(skipped):
        print(f"\nNOTE: {len(skipped)} config(s) skipped (see 'status' column).")
    if len(mismatches):
        print("\nRESULT: FAIL — documented param count(s) do not match instantiation:")
        for _, r in mismatches.iterrows():
            print(f"  {r['model_type']}/{r['size_variant']}: built "
                  f"{r['param_millions']:.4f}M vs documented "
                  f"{r['documented_millions']:.2f}M")
        return 1

    print("\nRESULT: PASS — all documented counts match instantiation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
