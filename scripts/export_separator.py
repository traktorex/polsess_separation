#!/usr/bin/env python
"""Export a training checkpoint as an inference bundle for other projects.

WHAT THIS PRODUCES
------------------
A directory that ``polsess_models.load_separator`` (= ``models.inference``)
loads without this repository:

    <out>/
      weights.safetensors   model weights only (no optimizer, no pickle)
      separator.json        architecture, constructor kwargs, sample rate,
                            source count, training provenance

A training checkpoint is roughly three times larger than its bundle (optimizer +
scheduler state) and needs ``torch.load(weights_only=False)``; a bundle is plain
tensors plus JSON. The legacy-checkpoint fixes of ``models/inference.py`` are
applied here, once, so a bundle always matches the current model classes.

The export verifies itself: the bundle is loaded back and must reproduce the
source model's output on a fixed noise input bit for bit. If that check, or any
step before it, fails, no bundle is left behind and the script exits non-zero.

Mamba-family checkpoints are refused: polsess-models ships without those
architectures (models/README.md), so nothing downstream could load the bundle.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" python scripts/export_separator.py \\
        --checkpoint checkpoints/mossformer2/SB/<run>/mossformer2_SB_best.pt \\
        --out ~/separators/mossformer2_matched_128k_e46

CPU is the default and is enough for every architecture that can be exported.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import MAMBA_MODELS, __version__  # noqa: E402
from models.inference import (  # noqa: E402
    BUNDLE_FORMAT_VERSION,
    BUNDLE_META,
    BUNDLE_WEIGHTS,
    load_model_for_inference,
    load_separator,
    resolve_architecture,
)

# Pre-`data.sample_rate` checkpoints were all trained on the 8 kHz corpus.
LEGACY_SAMPLE_RATE = 8000

# Libraries whose version can change what the same weights compute.
RECORDED_LIBRARIES = ("speechbrain", "rotary-embedding-torch", "einops", "safetensors")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_input(sample_rate: int) -> torch.Tensor:
    """Two seconds of fixed-seed noise, [1, T], for the source count and the
    round-trip check."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(1, 2 * sample_rate, generator=generator)


def export(checkpoint_path: Path, out_dir: Path, device: str) -> dict:
    from safetensors.torch import save_file

    model, checkpoint = load_model_for_inference(str(checkpoint_path), device=device)
    config = checkpoint["config"]
    model_type, model_params = resolve_architecture(config)
    if model_type in MAMBA_MODELS:
        raise SystemExit(
            f"{checkpoint_path}: '{model_type}' is a Mamba-family model. polsess-models "
            "ships without that family, so load_separator could not read the bundle "
            "anywhere but here. See models/README.md, 'What ships'."
        )
    data_config = config.get("data", {})
    per_variant = config.get("training", {}).get("per_variant_validation", False)
    sample_rate = data_config.get("sample_rate") or LEGACY_SAMPLE_RATE

    probe = probe_input(sample_rate).to(device)
    with torch.no_grad():
        reference = model(probe)
    if reference.dim() != 3 or reference.shape[0] != 1 or reference.shape[2] != probe.shape[1]:
        raise SystemExit(
            f"Unexpected output shape {tuple(reference.shape)} for input "
            f"{tuple(probe.shape)}; expected [1, n_src, time]."
        )

    provenance = checkpoint.get("provenance") or {}
    meta = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "polsess_models_version": __version__,
        "model_type": model_type,
        "model_params": model_params,
        "sample_rate": sample_rate,
        "n_src": reference.shape[1],
        "task": data_config.get("task"),
        "epoch": checkpoint.get("epoch"),      # 0-based, as the Trainer stores it
        # The Trainer keeps its monitored validation metric under one key.
        "val_sisdr": checkpoint.get("val_sisdr"),
        "val_metric": "si_sdri_mean_over_variants" if per_variant else "si_sdr",
        "source_checkpoint": {
            "name": f"{checkpoint_path.parent.name}/{checkpoint_path.name}",
            "sha256": sha256_of(checkpoint_path),
        },
        "training": {
            "git_sha": provenance.get("git_sha"),
            "git_dirty": provenance.get("git_dirty"),
            "wandb_run_id": checkpoint.get("wandb_run_id"),
            "dataset_type": data_config.get("dataset_type"),
        },
        "exported_on": date.today().isoformat(),
        # torch.__version__ carries the CUDA build tag, the package metadata does not.
        "exported_with": {
            "torch": torch.__version__,
            **{name: version(name) for name in RECORDED_LIBRARIES},
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / BUNDLE_WEIGHTS
    meta_path = out_dir / BUNDLE_META
    try:
        # clone(): safetensors refuses tensors that share storage (the rotary
        # blocks of MossFormer2 / TF-MossFormer share one `freqs` buffer); saved
        # separately, each copy loads back into the shared buffer unchanged.
        save_file(
            {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
            str(weights_path),
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")

        reloaded, _ = load_separator(str(out_dir), device=device)
        with torch.no_grad():
            roundtrip = reloaded(probe)
        if not torch.equal(roundtrip, reference):
            raise SystemExit(
                "Round-trip check FAILED: the bundle does not reproduce the source "
                f"model (max abs diff {(roundtrip - reference).abs().max().item():.3e}). "
                "Bundle deleted."
            )
    except BaseException:
        # A half-written or unverified bundle must not survive (Ctrl+C included).
        weights_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="training checkpoint (.pt) written by train.py")
    parser.add_argument("--out", required=True, type=Path,
                        help="bundle directory to create")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    out_dir = args.out.expanduser()
    if (out_dir / BUNDLE_META).exists() or (out_dir / BUNDLE_WEIGHTS).exists():
        raise SystemExit(f"{out_dir} already holds a bundle; remove it or pick another --out.")

    meta = export(args.checkpoint.expanduser(), out_dir, args.device)
    size_mb = (out_dir / BUNDLE_WEIGHTS).stat().st_size / 1e6
    print(f"bundle written: {out_dir}")
    print(f"  {meta['model_type']} @ {meta['sample_rate']} Hz, n_src={meta['n_src']}, "
          f"epoch={meta['epoch']} (0-based), {meta['val_metric']}={meta['val_sisdr']}")
    print(f"  weights {size_mb:.0f} MB, round-trip check passed (bit-identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
