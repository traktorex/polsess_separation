# polsess-models

The separator architectures of `polsess_separation` and the loaders for their
checkpoints, installable with pip so that another project can use a separator
trained here without copying model code.

This directory is two things at once:

- inside the repository it is the plain `models` package that `train.py`,
  `evaluate.py` and the tests import from the working directory, as always;
- the root `pyproject.toml` publishes the same files as the distribution
  `polsess-models`, import name `polsess_models`.

There is one copy of the code and no build step for work inside the repository.

## Using a separator in another project

```bash
pip install "polsess-models @ git+https://github.com/traktorex/polsess_separation@polsess-models-v0.1.1"
```

```python
from polsess_models import load_separator

model, info = load_separator("separators/mossformer2_matched_128k_e46", device="cuda")
estimates = model(mixture)      # [batch, time] at info["sample_rate"] -> [batch, info["n_src"], time]
```

A *bundle* is a directory with `weights.safetensors` and `separator.json`. It is
made from a training checkpoint, in this repository:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/export_separator.py \
    --checkpoint checkpoints/mossformer2/SB/<run>/mossformer2_SB_best.pt \
    --out ~/separators/mossformer2_matched_128k_e46
```

The export loads the bundle back and requires bit-identical output before it
reports success; on any failure it leaves no files behind. Bundles are ordinary
files; copy them wherever they are needed.

`separator.json` holds what `load_separator` needs (`model_type`, `model_params`,
`sample_rate`, `n_src`) and provenance: the source checkpoint's name and SHA-256,
the training commit and W&B run id, the library versions of the exporting
environment. Two fields are copied from the checkpoint as the Trainer stores
them: `epoch` is 0-based (the log's "epoch 46" is `45`), and `val_sisdr` is the
monitored validation metric, which `val_metric` names (mean SI-SDRi over the
MM-IPC variants for runs with `per_variant_validation`, SI-SDR otherwise).

| | training checkpoint (`.pt`) | bundle |
|---|---|---|
| written by | `train.py` | `scripts/export_separator.py` |
| read by | `load_model_for_inference` | `load_separator` |
| contents | weights, optimizer, scheduler, pickled config | weights, JSON metadata |
| meant for | this repository (resume, evaluation) | every other project |

## What ships

ConvTasNet, DPRNN, SepFormer, MossFormer2, TF-MossFormer. The Mamba family
(`mamba/`, `spmamba.py`, `mamba_tasnet.py`, `dpmamba.py`) is excluded in
`pyproject.toml`: `mamba/` derives from GPL-3.0 code and needs a CUDA build of
`mamba-ssm`. `scripts/export_separator.py` refuses Mamba-family checkpoints for
the same reason: no installed `polsess-models` could load the bundle. Inside
this repository those models keep working as before, from `.pt` checkpoints.

Shipping the family later means removing the four exclude lines, lifting the
refusal in the export script and bumping the version. It is also a licence
decision, not only a technical one: a wheel that contains `mamba/` is a
derivative of GPL-3.0 code, so the whole distribution, and any project that
redistributes it, falls under GPL-3.0.

`factory.py` is excluded because it belongs to the training side (it imports the
repository's `config.py`).

ConvTasNet caveat: training under AMP patches SpeechBrain's `EPS` to 1e-4
(`utils.apply_eps_patch`); that patch is applied by `train.py` / `evaluate.py`,
not by the model, so a ConvTasNet bundle loaded elsewhere runs with SpeechBrain's
stock 1e-8. No other architecture is affected.

## Maintenance: when does a consumer have to update?

Only `models/` is in the package. Changes to training code, configs, datasets or
scripts never reach a consumer and need no action.

A consumer pins one tag (`polsess-models-vX.Y.Z`) and moves to a newer one only
when it wants a bundle that its installed version cannot load, which
`load_separator` reports as an error naming both versions. In practice that
means a new architecture.

When `models/` changes:

1. Trained architectures stay loadable. A change that renames parameters or
   alters the forward pass of an existing architecture breaks every checkpoint
   of it, here as much as downstream; if it is ever necessary, add the fix to
   `normalize_state_dict` / `resolve_architecture` in `inference.py`, as was done
   for SepFormer and SPMamba.
2. Release: bump `__version__` in `models/__init__.py` (new architecture or
   loader feature: minor; fix: patch), commit, and tag the commit on `main` that
   carries the bump. The tag must equal the version, and tags are not pushed by
   a plain `git push`:

   ```bash
   V=$(python -c "import models; print(models.__version__)")
   git tag "polsess-models-v$V" && git push origin "polsess-models-v$V"
   ```

   A published tag is never moved or deleted: consumers pin it, and pip caches
   what it resolved to. A mistake is fixed with a new patch version. Nothing
   enforces the bump itself, so a change under `models/` that is never tagged
   simply never reaches a consumer; that is safe, only easy to forget.
3. Rules for code in this directory: relative imports only (`from .x import y`,
   never `from models.x`), no imports from the rest of the repository (`config`,
   `utils`, `training`, ...), and no imports of the excluded modules from a
   shipped one. `tests/test_polsess_models_package.py` checks all three
   statically and, where `hatchling` is installed, also builds the wheel and
   imports it from a clean directory.
