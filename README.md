# PolSESS Speech Separation for Polish ASR Preprocessing

Codebase for a master's thesis on monaural speech separation and its use as a
front end for Polish automatic speech recognition. This repository is the
**separation-training half**: PyTorch implementations of ConvTasNet, DPRNN,
SepFormer, MossFormer2, TF-MossFormer, SPMamba, Mamba-TasNet and DPMamba,
trained on the PolSESS corpus, which simulates realistic acoustic conditions
(reverberation, background scenes, event sounds). Thesis chapters 4–5. The
`models/` directory is also published as the pip-installable `polsess-models`
package so other projects can load the trained separators.

The other half — the diarization-guided ASR pipeline evaluated on CLARIN
conversational speech (thesis chapter 6) and its showcase webapp (chapter 7) —
lived here until 2026-10-23 and now develops in
[DGSS-ASR](https://github.com/traktorex/DGSS-ASR). The thesis-era instrument is
preserved at tag `pre-asr-removal-2026-10-23` (hand-in state: `handin-2026-09-05`).

**Results are reported in the thesis, not here.** This repository holds the
code that produced the separation results:
model implementations, experiment configurations, W&B sweep definitions,
generated benchmark and parameter-count artifacts (`docs/generated/`), the
thesis figure scripts (`scripts/thesis_figures/`) and the test suite. Datasets
(PolSESS, Libri2Mix) and trained checkpoints are **not distributed with
the repository** — they are available from the author, and nothing here runs
end-to-end without them. `CLAUDE.md`, the Claude Code instruction file, doubles
as the detailed technical map.

## Repository layout

```
polsess_separation/
├── config.py                  # Config dataclasses: DataConfig / ModelConfig / TrainingConfig
├── train.py, train_sweep.py   # Training entry point + W&B sweep entry point
├── evaluate.py, evaluate_all.py   # One checkpoint / a whole manifest of checkpoints
├── setup.sh, download_dataset.sh  # Cloud-GPU provisioning + corpus download helpers
├── CLAUDE.md                  # Detailed technical map (Claude Code instruction file)
├── THIRD_PARTY.md             # Vendored code, model weights and audio excerpts: licences
├── requirements.txt, requirements-freeze.txt   # Installable core set / exact dev-env versions
├── pyproject.toml             # Packages models/ as `polsess-models` (import name polsess_models)
├── models/                    # Architectures + registry (`get_model` in models/__init__.py)
│   ├── conv_tasnet.py, dprnn.py, sepformer.py, factory.py
│   ├── spmamba.py, mamba_tasnet.py, dpmamba.py   # require mamba-ssm (Linux + CUDA)
│   ├── mamba/                 # BiMamba blocks, adapted from xi-j/Mamba-TasNet
│   ├── mossformer2/           # Vendored from ClearerVoice-Studio + project wrapper
│   └── tf_mossformer/         # Re-implemented from arXiv:2607.21128 on a TF-Locoformer copy
├── datasets/                  # polsess / libri2mix / echoset loaders + registry
├── training/                  # trainer.py (loop, AMP, curriculum) + setup.py (shared builders)
├── utils/                     # common, model_utils, metrics, wandb_logger, logger, warning_filters
├── experiments/               # YAML training configs, one directory per architecture
├── sweeps/                    # W&B sweep definitions + historical run ledger
├── tests/                     # pytest suite (29 files, 468 tests)
├── docs/                      # MM-IPC notes, plans, generated/ artifacts
└── scripts/                   # Benchmarks, audits, export_separator.py, thesis_figures/
```

Registries are dict-based: `get_model("sepformer")`, `get_dataset("polsess")`.
Mamba architectures are excluded automatically when `mamba-ssm` is unavailable.
Measured parameter counts per configuration: `docs/generated/model_manifest.md`.

## Installation

Create a virtualenv named `venv/` at the repository root, then
`venv/bin/pip install -r requirements.txt`. `requirements-freeze.txt` records
the exact versions used for the thesis experiments (it still lists the ASR
pipeline's packages, which nothing here imports any more). Optional extras
(Mamba models) are commented blocks at the end of `requirements.txt`.

## Training

```bash
python train.py --config experiments/dprnn/dprnn_baseline.yaml
python train.py --config experiments/mossformer2/6-final-training/128k_matched_final.yaml
python train.py --config experiments/tf_mossformer/s_8k.yaml
python train.py --config experiments/dprnn/dprnn_baseline.yaml --no-wandb --seed 123
python train.py --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
```

Other CLI overrides: `--model-type`, `--dataset-type`, `--data-root`, `--task`,
`--save-dir`, `--save-all-checkpoints`, `--no-amp`.

**Configuration priority:** defaults → environment variables → YAML → CLI.
`Config.__post_init__` additionally forces the model's output source count to
match the task (ES → 1, SB/EB → 2) regardless of the YAML. Checkpoints land in
`checkpoints/{model_type}/{task}/{run_name}/` beside a `config.yaml` and a
`run_manifest.yaml` recording git SHA, library versions, GPU, hostname, seed,
argv and W&B run id.

## Evaluation

```bash
# All MM-IPC variants on the PolSESS test split
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
python evaluate.py --checkpoint path/to/model.pt --variant SER --no-pesq --no-stoi
python evaluate.py --checkpoint path/to/model.pt \
    --dataset librimix --librimix-root ~/datasets/LibriMix/Libri2Mix --output results.csv
python evaluate_all.py --resume    # manifest: experiments/thesis_eval_manifest.csv
```

Metrics: SI-SDR / SI-SDRi (dB), PESQ-WB (1–5), STOI (0–1).

## Sweeps and tests

```bash
wandb sweep sweeps/3-hyperparam-opt/dprnn/stage1/dprnn.yaml   # returns a sweep id
wandb agent <sweep_id>                                        # run inside tmux
pytest                                                        # 29 files, 468 tests
pytest tests/test_model.py -v
pytest --cov=. --cov-report=html
```

Sweep YAMLs point at `train_sweep.py`. `sweeps/EXPERIMENT_LOG_monolithic.md` is
a **historical log frozen in April 2026** covering the early baseline and HPO
series; most of its W&B links are dead (the original project was deleted in
February 2026). `sweeps/all_runs.csv` is a partial recovery of that ledger,
documented in `sweeps/RESULTS_SCHEMA.md`. Mamba, GPU and dataset-dependent tests skip
automatically on a CPU-only machine without the corpora.

## Technical notes

**MM-IPC augmentation.** MM-IPC (Mix Modification by Inverted Phase
Cancellation) varies background complexity during training by subtracting audio
layers from the full mix through inverted phase cancellation. Letters denote what
is present — S = scene, E = event, R = reverberation, C = clean — giving **SER,
SR, ER, R, C** indoors (with reverb) and **SE, S, E, C** outdoors (without); `C`,
clean speech only, belongs to both families. Implemented in
`datasets/polsess_dataset.py` with lazy loading, so only the layers the selected
variant needs are read; validation uses deterministic per-sample selection.
`allowed_variants` restricts the set (`None` = all) and the trainer mutates it in
place for curriculum learning — which is why `persistent_workers` must stay off
while a curriculum is active.

**Mixed precision.** AMP is on by default. SpeechBrain's `EPS=1e-8` underflows
to zero in float16 and produces NaN SI-SDR, so `utils.apply_eps_patch` raises it
to `1e-4` for the ConvTasNet SpeechBrain lobe. Most models train in float16 with
a GradScaler; Mamba models, MossFormer2 and TF-MossFormer use bfloat16 without
one (dispatch by `model_type` in `training/trainer.py`).

**Benchmarks and audits.** `scripts/benchmark_inference.py --cross-check` (MACs,
latency, RTF, peak inference VRAM), `scripts/benchmark_training.py` (throughput,
peak training VRAM), `scripts/model_manifest.py` (parameter counts) and the
CPU-only `scripts/audit_mmipc.py` / `scripts/audit_split_leakage.py` all write
into `docs/generated/`, each CSV carrying provenance columns (GPU, torch, CUDA,
ptflops version). Regenerate them rather than editing by hand.

## PolSESS dataset layout

Each of `train/`, `val/` and `test/` holds one directory per audio layer —
`clean/` (target for the ES task), `event/`, `mix/` (full mixture), `scene/`,
`sp1_reverb/`, `sp2_reverb/`, `ev_reverb/` — plus a
`corpus_PolSESS_C_in_<split>_final.csv` index. Tasks: `ES` (enhance one
speaker), `EB` (enhance both), `SB` (separate both — the thesis task).

## Hardware

Training used two local desktop GPUs and rented cloud instances; the 128k-sample
final runs need a CUDA GPU and several days each. Measured per-model throughput,
latency and peak VRAM (RTX 4070, 12 GB) live in `docs/generated/` — consult those
rather than a rule of thumb, since requirements differ by an order of magnitude
across architectures. Mamba architectures additionally require Linux and CUDA.

## Environment variables

| Variable | Purpose |
|---|---|
| `POLSESS_DATA_ROOT` | PolSESS dataset root (falls back to the default in `config.py`) |

## Troubleshooting

- **NaN SI-SDR** — float16 underflow; the EPS patch normally handles it, and the
  trainer skips NaN/Inf batches, aborting after 1000 consecutive ones. If it
  persists set `use_amp: false`, or `residual_in_fp32: true` for deep Mamba-TasNet.
- **Training suddenly far slower per batch** — VRAM exhausted, spilling to system
  RAM; reduce `batch_size`, compensate with `grad_accumulation_steps`.
- **`--resume` extends the epoch budget** — `num_epochs` counts *additional*
  epochs, not the total, and early-stopping patience restarts. Stop the run at
  the intended total manually for budget-matched comparisons.
- **Mamba on Windows** — `mamba-ssm` needs Linux and CUDA; use WSL2 with CUDA 12.4+.

## References

ConvTasNet [1809.07454](https://arxiv.org/abs/1809.07454) · DPRNN
[1910.06379](https://arxiv.org/abs/1910.06379) · SepFormer
[2010.13154](https://arxiv.org/abs/2010.13154) · MossFormer2
[2312.11825](https://arxiv.org/abs/2312.11825) · SPMamba
[2404.02063](https://arxiv.org/abs/2404.02063) · DPMamba
[2403.18257](https://arxiv.org/abs/2403.18257) · Mamba-TasNet
[2407.09732](https://arxiv.org/abs/2407.09732) · TF-MossFormer
[2607.21128](https://arxiv.org/abs/2607.21128) · TF-Locoformer
[2408.03440](https://arxiv.org/abs/2408.03440) ·
[SpeechBrain](https://github.com/speechbrain/speechbrain). MM-IPC follows Kleć
et al.'s approach for PolSESS.

## Third-party code

Upstream implementations vendored so a checkpoint or architecture loads without
pulling in a whole framework:

- `models/mossformer2/` — MossFormer2, from [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Apache-2.0)
- `models/tf_mossformer/` — TF-MossFormer ([arXiv:2607.21128](https://arxiv.org/abs/2607.21128)),
  re-implemented from the paper on a copy of [TF-Locoformer](https://github.com/merlresearch/tf-locoformer), MERL (Apache-2.0) — no upstream code was released
- `models/mamba/` — BiMamba blocks adapted from [xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet) (GPL-3.0), on top of `mamba-ssm` (Apache-2.0)

SpeechBrain (Apache-2.0) supplies the ConvTasNet and SepFormer lobes. Each
vendored module's docstring records its upstream source, revision and any local
patch; the upstream licence text sits next to each vendored tree, and
`THIRD_PARTY.md` lists all vendored code and audio excerpts with their licences.
