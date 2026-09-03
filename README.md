# PolSESS Speech Separation for Polish ASR Preprocessing

Codebase for a master's thesis on monaural speech separation and its use as a
front end for Polish automatic speech recognition. Two halves:

1. **Separation training** — PyTorch implementations of ConvTasNet, DPRNN,
   SepFormer, MossFormer2, SPMamba, Mamba-TasNet and DPMamba, trained on the
   PolSESS corpus, which simulates realistic acoustic conditions (reverberation,
   background scenes, event sounds). Thesis chapters 4–5.
2. **ASR pipeline** — `asr_pipeline/`: diarization → routing → enhancement →
   separation → post-processing → assembly → transcription, evaluated on real
   conversational Polish speech from the CLARIN corpus, plus two demonstration
   front-ends. Thesis chapter 6.

**Results are reported in the thesis, not here** (chapter 5 for separation,
chapter 6 for the pipeline). This repository holds the code that produced them:
model implementations, experiment configurations, W&B sweep definitions,
generated benchmark and parameter-count artifacts (`docs/generated/`), the
thesis figure scripts (`scripts/thesis_figures/`) and the test suite. Datasets
(PolSESS, CLARIN, Libri2Mix) and trained checkpoints are **not distributed with
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
├── models/                    # Architectures + registry (`get_model` in models/__init__.py)
│   ├── conv_tasnet.py, dprnn.py, sepformer.py, factory.py
│   ├── spmamba.py, mamba_tasnet.py, dpmamba.py   # require mamba-ssm (Linux + CUDA)
│   ├── mamba/                 # BiMamba blocks, adapted from xi-j/Mamba-TasNet
│   └── mossformer2/           # Vendored from ClearerVoice-Studio + project wrapper
├── datasets/                  # polsess / libri2mix / echoset loaders + registry
├── training/                  # trainer.py (loop, AMP, curriculum) + setup.py (shared builders)
├── utils/                     # common, model_utils, metrics, wandb_logger, logger, warning_filters
├── experiments/               # YAML training configs, one directory per architecture
├── sweeps/                    # W&B sweep definitions + historical run ledger
├── tests/                     # pytest suite (55 files, 1441 tests)
├── docs/                      # MM-IPC notes, sweep-CSV schema, generated/ artifacts
├── scripts/                   # Benchmarks, audits, CLARIN helpers, thesis_figures/
├── asr_pipeline/              # Pipeline package: stages/, eval/, configs/, vendor/, SCOPE.md
├── asr/                       # Notebooks driving asr_pipeline/ (explore, evaluate, CLARIN)
└── webapp/                    # FastAPI showcase UI over asr_pipeline (read-only consumer)
```

Registries are dict-based: `get_model("sepformer")`, `get_dataset("polsess")`.
Mamba architectures are excluded automatically when `mamba-ssm` is unavailable.
Measured parameter counts per configuration: `docs/generated/model_manifest.md`.

## Installation

Create a virtualenv named `venv/` at the repository root (the webapp launcher
and some commands assume that exact path), then
`venv/bin/pip install -r requirements.txt`. That file covers the main
environment only (`requirements-freeze.txt` records the exact versions used for
the thesis experiments); the ASR pipeline reaches some backends (Sortformer, CohereX,
Brouhaha) through separate isolated venvs whose pins conflict with it — see
`CLAUDE.md`.

## Training

```bash
python train.py --config experiments/dprnn/dprnn_baseline.yaml
python train.py --config experiments/mossformer2/6-final-training/128k_matched_final.yaml
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
pytest                                                        # 55 files, 1441 tests
pytest tests/test_model.py -v
pytest --cov=. --cov-report=html
```

Sweep YAMLs point at `train_sweep.py`. `sweeps/EXPERIMENT_LOG_monolithic.md` is
a **historical log frozen in April 2026** covering the early baseline and HPO
series; most of its W&B links are dead (the original project was deleted in
February 2026). `sweeps/all_runs.csv` is a partial recovery of that ledger,
documented in `sweeps/RESULTS_SCHEMA.md`. The test suite last passed in full on
2026-09-03; roughly half of it covers `asr_pipeline/`, and Mamba, GPU and
dataset-dependent tests skip automatically on a CPU-only machine without the
corpora.

## ASR pipeline

Read `asr_pipeline/SCOPE.md` before changing anything in this package — it is
the scope contract (purpose, error philosophy, fallback ledger). The CLI front
door preflights the environment and checkpoints before loading any model:

```bash
python -m asr_pipeline run --config asr_pipeline/configs/sweep_best_e31_refineplus.yaml \
    --input rec.wav --write-outputs <eval_root>
python -m asr_pipeline batch --split clarin_dev --mode no_enh
python -m asr_pipeline score --eval-root <eval_root> --out-dir <csv_dir>
```

`--set stage.knob=value` (repeatable) overrides any config knob and
`--mode full|no_sep|no_enh|minimal` selects an ablation preset. Evaluation is
two-layer: **L2** audio quality (SI-SDR / PESQ / STOI against oracle channels,
plus non-intrusive TorchAudio-SQUIM) and **L3** ASR accuracy (cpWER, tcpWER,
cpCER via `meeteval`). An earlier one-shot REAL-M / LibriMix evaluation flow was
removed; this pipeline supersedes it. The notebooks in `asr/` drive it
interactively and score its output.

## Front-end

```bash
./webapp/run.sh                                   # FastAPI showcase UI, port ${WEBAPP_PORT:-8871}
venv/bin/python -m webapp.dev_server --port 8899  # GPU-free dev harness (replays a fake run)
```

`webapp/` is a read-only consumer of the pipeline: it runs it server-side, one
job at a time, and its HTTP contract is `webapp/API.md`. A second, experimental
front-end that ran overlap detection, routing and separation entirely in the
browser (ONNX Runtime Web) is not part of this branch — nothing depends on it and
it is kept on branch `experiment/webapp-ondevice`.

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
a GradScaler; Mamba models and MossFormer2 use bfloat16 without one (dispatch by
`model_type` in `training/trainer.py`).

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
| `HF_TOKEN` | HuggingFace token for the pyannote diarization stage |
| `HF_HUB_OFFLINE` | Set to `1` for reproducible offline evaluation runs (webapp default) |
| `AP_BWE_CHECKPOINT` | AP-BWE bandwidth-extension checkpoint (post-separation stage) |
| `SORTFORMER_VENV_PY` | Python of the isolated NeMo venv (Sortformer diarization backend) |
| `COHEREX_VENV_PY` | Python of the isolated CohereX venv (Cohere transcription backend) |
| `BROUHAHA_VENV_PY` | Python of the isolated Brouhaha venv (fragment-acoustics scoring) |
| `ASR_PIPELINE_DEBUG_LOG` | Debug log path (default `/tmp/asr_pipeline_debug.log`) |
| `WEBAPP_PORT`, `WEBAPP_JOBS_ROOT` | Showcase webapp port and job output root |

Missing venv paths fail loudly at preflight rather than falling back silently —
a deliberate rule of `asr_pipeline/SCOPE.md`.

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
[2407.09732](https://arxiv.org/abs/2407.09732) ·
[SpeechBrain](https://github.com/speechbrain/speechbrain). MM-IPC follows Kleć
et al.'s approach for PolSESS.

## Third-party code

Upstream implementations vendored so a checkpoint or architecture loads without
pulling in a whole framework:

- `models/mossformer2/` — MossFormer2, from [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Apache-2.0)
- `models/mamba/` — BiMamba blocks adapted from [xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet) (GPL-3.0), on top of `mamba-ssm` (Apache-2.0)
- `asr_pipeline/vendor/ap_bwe/` — [AP-BWE](https://github.com/yxlu-0102/AP-BWE) (MIT)
- `asr_pipeline/vendor/tiger/` — [TIGER](https://github.com/JusperLee/TIGER) (MIT)
- `asr_pipeline/vendor/tf_locoformer/` — [TF-Locoformer](https://github.com/merlresearch/tf-locoformer), MERL (Apache-2.0)
- `asr_pipeline/vendor/mossformer2_dp/` — [MossFormer2 standalone](https://github.com/alibabasglab/MossFormer2) (MIT)

SpeechBrain (Apache-2.0) supplies the ConvTasNet and SepFormer lobes. Each
vendored module's docstring records its upstream source, revision and any local
patch; the upstream licence text sits next to each vendored tree, and
`THIRD_PARTY.md` lists all vendored code, runtime model weights and audio
excerpts with their licences.
