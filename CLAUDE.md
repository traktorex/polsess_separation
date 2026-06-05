# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PyTorch implementation of speech separation using ConvTasNet, SepFormer, DPRNN, SPMamba, Mamba-TasNet, and DPMamba architectures on the PolSESS dataset. Part of a master's thesis on speech separation for downstream Polish ASR preprocessing.

Thesis prose and experiment logs live in `thesis/` — a symlink to an Obsidian vault on Windows, tracked by its own git repo (ignored here). Read freely; when editing thesis content, `cd thesis/` so `thesis/CLAUDE.md` stacks in on top of this file.

## Style

Do not skip your reasoning when Extended Thinking is enabled. 
Always produce a CoT. 
Don't be afraid of thinking too long - be afraid of thinking too briefly. The amount of thinking should be proportional to the complexity of the task you're given.
Avoid undue verbosity by using CoT to structure your response.

## Dataset Variants

- `PolSESS_C_both` = `C_both_16k_faulty` — old 8k-effective dataset (half was duplicated). Used for early baselines, HPO, and HPO validation runs.
- `PolSESS_C_new_64` = `C_new_64` — correct 64k dataset generated 2026-04-15. Use `train_max_samples=16000` / `32000` / full for 16k / 32k / 64k scaling experiments.

## W&B Projects

- `polsess-separation` — standalone runs (baselines, HPO validation), uses `PolSESS_C_both`.
- `polsess-thesis-experiments` — sweeps.
- `polsess-separation-real16k` / `-32k` / `-64k` — scaling runs on subsets of `PolSESS_C_new_64`.

## Thesis Code Principles

This code will be reviewed by academic supervisors. Prioritize: clarity over cleverness, simplicity over abstraction, reproducibility. Prefer explicit implementations that match cited papers. Don't over-engineer — no factories for <4 variants, no deep inheritance, no speculative features. Experiment logging is handled outside this repo.

## Keeping This File Current

After any substantial change — new top-level script or subsystem, new env var, new dataset/model/task variant, new gotcha worth flagging, removed commands, or changed config precedence — propose a targeted edit to this CLAUDE.md. Update in place; don't rewrite from scratch. Skip for routine bugfixes, refactors, or one-off experiments.

## Key Commands

**Training:**
```bash
python train.py --config experiments/dprnn/dprnn_baseline.yaml
python train.py --config experiments/spmamba/spmamba_baseline.yaml --no-wandb --seed 123
python train.py --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
```

**Evaluation:**
```bash
# PolSESS (all MM-IPC variants)
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
# Specific variant
python evaluate.py --checkpoint path/to/model.pt --variant SER
# Fast (skip PESQ/STOI)
python evaluate.py --checkpoint path/to/model.pt --no-pesq --no-stoi
# Libri2Mix
python evaluate.py --checkpoint path/to/model.pt --dataset librimix --librimix-root /home/user/datasets/LibriMix/Libri2Mix
# Save CSV
python evaluate.py --checkpoint path/to/model.pt --output results.csv
```

**Testing:**
```bash
pytest
pytest --cov=. --cov-report=html
pytest tests/test_model.py -v
```

**Sweeps:**
```bash
# Launch a W&B sweep (the sweep YAML points at train_sweep.py as the program)
wandb sweep sweeps/3-hyperparam-opt/dprnn/stage1/dprnn.yaml
# then run agents against the returned sweep ID
```

**Benchmarks (thesis data):**
```bash
python scripts/benchmark_inference.py
python scripts/benchmark_training.py
```

**Interactive:**
```bash
jupyter notebook test_model_interactive.ipynb
jupyter notebook asr/explore_pipeline.ipynb   # interactive frontend for the asr_pipeline/ package (see ASR section)
```

## Architecture Overview

**Configuration (`config.py`):** Dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`) with nested model/dataset params. Priority: defaults < env vars < YAML < CLI args. Use `get_config_from_args()` for CLI, `load_config_for_run(wandb.config)` for sweeps.

**Model Registry (`models/__init__.py`):** Dict-based. `get_model("name")` returns class. Mamba models auto-excluded without `mamba-ssm`.
- `convtasnet` (~8M), `sepformer` (~26M), `resepformer` (~8M), `mossformer2` (matched ~26M / full ~55.7M), `dprnn` (~2-3M) — cross-platform
  - `resepformer` = RE-SepFormer (Subakan et al. 2022): resource-efficient SepFormer. Thin wrapper reusing SpeechBrain's `ResourceEfficientSeparator` mask network + the same dual_path `Encoder`/`Decoder` as `sepformer`.
  - `mossformer2` = MossFormer2 (Zhao et al. 2023, arXiv:2312.11825): transformer + gated-FSMN hybrid. The model files in `models/mossformer2/` are **vendored** from ClearerVoice-Studio (`train/speech_separation/models/mossformer2/`); `models/mossformer2/__init__.py` is the project wrapper. Pure PyTorch (deps: `einops`, `rotary-embedding-torch`), so cross-platform. Single `N` knob = encoder dim = transformer dim (the two must match upstream); `num_blocks` is GFSMN depth (24 = paper full, 11 ≈ SepFormer-matched); `attn_dropout` (default 0.1, upstream hard-coded) covers attention-path dropout — FSMN-gate dropout stays fixed at 0.1. Sweep override key `dropout` routes to `attn_dropout`. Configs: `experiments/mossformer2/mossformer2_{matched,full}.yaml`.
- `spmamba` (~1.2M), `mamba_tasnet` (XS/S/M/L: 2.2-59.6M), `dpmamba` (XS/S/M/L: 2.3-59.8M) — Linux + CUDA only

**Dataset Registry (`datasets/__init__.py`):** Dict-based. `get_dataset("name")` returns class. Supports: `polsess`, `libri2mix`.

**ASR subsystem (`asr/`):** Notebooks driving the productionised CLARIN pipeline. The pre-CLARIN one-shot REAL-M/LibriMix eval flow (`evaluate_asr.py` + intrusive variant + `dataset.py`/`transcribe.py`/`metrics.py` + its `test_asr.py`) is **archived** under `asr/archive/old-asr/`; the original Gradio POC notebook (`asr_pipeline.ipynb`) + early LibriMix-prep scripts sit in `asr/archive/`. Archived code is parked — its imports reference the old top-level `asr.` package layout and would need rewiring to run.
- `clarin_fragments.ipynb` / `clarin_subset_review.ipynb`: select + review the CLARIN test fragments (uses `scripts/clarin_fragment_finder.py`).
- `explore_pipeline.ipynb`: interactive frontend for the productionised `asr_pipeline/` package — per-stage knobs, re-run any stage in isolation, one model on GPU at a time.
- `evaluate_pipeline.ipynb`: three-layer evaluation of `asr_pipeline/` output against the CLARIN debleed (oracle) channels, backed by `asr_pipeline/eval/`.

**`asr_pipeline/` package** — productionised pipeline. `Pipeline` orchestrator runs seven stages in fixed order:
1. **diarization** — pyannote `speaker-diarization-3.1`, `num_speakers=2`, mono 16 kHz. HF token via `$HF_TOKEN`.
2. **routing** — split overlap vs solo regions.
3. **enhancement** — `mpsenet` default; ClearerVoice backends (`frcrn_se_16k`, `mossformer_gan_se_16k`, `mossformer2_se_48k`) available.
4. **separation** — SepFormer 64k baseline checkpoint by default; runs on overlap fragments only.
5. **post_separation_processing** — VAD mask + optional BWE (`naive` / `ap_bwe` / `flowhigh`). Always-on (downstream depends on its `_gated` arrays); set `backend: naive` to apply only the mask.
6. **assembly** — stitch per-speaker streams, ECAPA anchor for speaker identity across pieces.
7. **transcription** — `whisper` / `whisperx` backends; default = WhisperX `large-v2` + `jonatasgrosman/wav2vec2-large-xlsr-53-polish` alignment (rationale in `asr_pipeline/configs/README.md`).

Phase-major execution (one model on GPU at a time). Config via nested dataclasses + YAML. Configs in `asr_pipeline/configs/`: `default.yaml` (POC-equivalent), `p4_fixed_pad.yaml` / `p5_full_length.yaml` (ablation knobs). Debug log at `/tmp/asr_pipeline_debug.log` (override `ASR_PIPELINE_DEBUG_LOG`) — survives the WSL stdout bridge dropping.

**`asr_pipeline/eval/`** — three-layer scoring. `evaluate_recording(rec) → ScoreCard` runs all three layers for one recording; `evaluate_many` batches with SQUIM loaded once; `walk_eval_tree` yields `Recording` per directory under the eval root.
- **L1 diarization** — DER between `pipeline/diarization.json` and reference RTTM.
- **L2 audio quality** — intrusive SI-SDR / PESQ-WB / STOI (chunked, median-aggregated, speech-presence filtered) when oracle audio is available; non-intrusive TorchAudio-SQUIM (chunked, mean-aggregated) always.
- **L3 ASR** — cpWER + tcpWER per ablation mode (full / no-sep / no-enh), ORC-WER on the mixture baseline. Backed by `meeteval`.

Low-level helpers exported for notebook use: `parse_gt_txt`, `parse_transcript_file`, `parse_rttm`, `compute_der`, `cpwer_meeteval`, `orc_wer_meeteval`.

**ASR datasets**
- `~/datasets/clarin_gotowy/gotowy/` — CLARIN debleed eval set (oracle per-speaker channels). Root = `<id>.wav` stereo inputs; `debleed/<id>_{L,R}.wav` = oracle channels; `debleed_enhanced/` = MossFormerGAN-enhanced oracles; `after_pipeline/<id>_{s1,s2}.wav` = pipeline outputs; `transcripts/<id>.txt` = pipeline transcripts; `eval_cache/` = cached references.
- `~/datasets/clarin_all_2speakers/` — full CLARIN 2-speaker download (no oracle channels). `clarin_download/<id>.wav` raw inputs (+ `Korpus.csv`, `Korpus_with_filename.csv`); `diarization/<id>.json` pyannote outputs; `enhanced_mossformer/<id>.wav` MossFormerGAN-enhanced; `auto_transcription_raw/<id>.{txt,json}` and `auto_transcription_enhanced_mossformer/<id>.{txt,json}` WhisperX transcripts.

**ASR helper scripts (`scripts/`)**
- `run_pipeline_on_recording.py` — full pipeline on one recording in three ablation modes (`pipeline` / `pipeline_nosep` / `pipeline_noenh`); drives the L3 WER table.
- `prepare_eval_references.py` — cache enhanced oracles + GT-style transcripts for the eval module.
- `enhance_clarin_debleed.py` — batch MossFormerGAN_SE_16K on oracle debleed channels.
- `diarize_clarin_2speakers.py` — pyannote over the full 2-speaker download → `diarization/<id>.json`.
- `transcribe_clarin_2speakers.py` — WhisperX over the full 2-speaker download, raw and MossFormerGAN-enhanced.

**Training Flow:** `train.py` → config → dataloaders → `create_model_from_config()` → optional `torch.compile()` → `Trainer` (AMP, grad accumulation, checkpointing, curriculum learning).

**Checkpoints:** Saved to `checkpoints/{model_type}/{task}/{run_name}/`. Run name comes from W&B when available, otherwise timestamp. Each directory includes `config.yaml` for reproducibility. By default only the best checkpoint is kept; `save_all_checkpoints: true` keeps every improvement.

## Experiment Configs

Each model architecture has its own YAML configs in `experiments/`. When creating new experiments, always base them on an existing YAML file from `experiments/` — do not write configs from scratch. The YAML structure mirrors the config dataclasses:

```yaml
data:
  dataset_type: polsess
  batch_size: 16
  task: SB                    # ES=enhance 1 speaker, EB=enhance both, SB=separate both

model:
  model_type: dprnn           # Must match a key in the model registry
  dprnn:                      # Nested params matching the model's param dataclass
    N: 64
    hidden_size: 128
    num_layers: 6

training:
  num_epochs: 100
  lr: 0.00015
  use_amp: true
  use_wandb: true
  curriculum_learning:        # Optional: progressive variant introduction
    - epoch: 1
      variants: ["C", "R"]
    - epoch: 8
      variants: ["R", "SR", "S", "SE", "ER", "E", "SER"]
      lr_scheduler: start     # Optional: gate LR scheduler to this stage
```

## Key Technical Details

- **AMP:** Enabled by default. SpeechBrain EPS patched from 1e-8 to 1e-4 in `utils/common.py` to prevent float16 underflow. Most models use float16 + GradScaler; Mamba models **and MossFormer2** use bfloat16 without GradScaler (dispatch on `model_type` in `training/trainer.py:_setup_amp`). MossFormer2's squared-ReLU attention overflows fp16 once activations sharpen — first seen as NaN val SI-SDR (fixed by fp32 validation), then as training NaNs at low `attn_dropout` / higher LR in the 128k sweep.
- **torch.compile:** Auto-applied on Linux for ~10-20% speedup. Checkpoint loading handles `_orig_mod` prefix. Skipped for Mamba models. `mossformer2` is compiled with `dynamic=False` (per-shape static specialization): its vendored rotary block disables the seq-len cache (`cache_if_possible=False`) and its token-shift/group-rearrange can't be lowered under symbolic shapes — so fixed-length crops compile once, new lengths trigger a one-time static recompile. Per-architecture dispatch lives in `compile_for_model_type` (`utils/model_utils.py`), shared by `train.py` and `train_sweep.py`.
- **MM-IPC (Mix Modification by Inverted Phase Cancellation):** Augmentation that randomly varies background complexity during training by subtracting audio layers from the full mix. Indoor variants (with reverb): SER/SR/ER/R. Outdoor variants (no reverb): SE/S/E/C. Letters indicate what's present: S=scene, E=event, R=reverb, C=clean. Implemented via lazy loading in `datasets/polsess_dataset.py`. Validation uses deterministic selection (seeded by sample index).
- **Curriculum Learning:** Configure in YAML `training.curriculum_learning`. Progressive variant introduction + optional LR scheduler gating. Note: when curriculum learning is active, the LR scheduler is **disabled by default** until a curriculum entry includes `lr_scheduler: start` — omitting this key means the scheduler never runs.
- **Gradient Accumulation:** `training.grad_accumulation_steps` for effective batch scaling.
- **Mamba Models (SPMamba, Mamba-TasNet, DPMamba):** Require Linux + CUDA + `mamba-ssm`. AMP uses bfloat16 (no GradScaler) — Mamba CUDA kernels run float32 internally. Mamba-TasNet/DPMamba come in XS/S/M/L size configs. `models/mamba/` contains BiMamba building blocks adapted from xi-j/Mamba-TasNet.

## PolSESS Dataset Structure

```
PolSESS/
├── train/
│   ├── clean/         # Clean speech (target for ES task)
│   ├── event/         # Event sounds
│   ├── mix/           # Full mixed audio
│   ├── scene/         # Background scene
│   ├── sp1_reverb/    # Speaker 1 with reverb
│   ├── sp2_reverb/    # Speaker 2 with reverb
│   ├── ev_reverb/     # Event with reverb
│   └── corpus_PolSESS_C_in_train_final.csv
├── val/
└── test/
```

MM-IPC works by subtracting layers from the full mix using inverted phase cancellation. For example, the "SR" variant (scene + reverb) is created by removing the event layer from the full SER mix.

## Common Pitfalls

1. **NaN in SI-SDR:** AMP underflow — EPS patch should handle it. If not, `use_amp: false`. The trainer skips NaN/Inf batches; after 1000 consecutive NaN batches it aborts the run (`ConsecutiveNaNError` → `SystemExit(1)`, sweep-friendly — see `MAX_CONSECUTIVE_NAN_BATCHES` in `training/trainer.py`).
2. **Memory overflow:** Reduce `batch_size`, use `grad_accumulation_steps` to compensate.
3. **Config precedence:** CLI > YAML > env vars > defaults. Additionally, `Config.__post_init__` silently forces the model's output source count (`C`/`n_srcs`) to match the task (ES→1, SB/EB→2), overriding whatever the YAML says.
4. **MambaTasNet NaN:** Deep configs need `residual_in_fp32: true`.  `grad_clip_norm: 1.0` (not 5.0) might help too.
5. **Mamba on Windows:** Requires WSL2 + CUDA toolkit 12.4+. Non-Mamba models work natively.
6. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr`, not dict access.

## Virtual Environments

**Main (`venv/`)** — all models except SPMamba3. Alias: `polsess_venv`.

**SPMamba3 (`venv_mamba3/`)** — torch 2.11.0+cu130, triton 3.6.0, Mamba-3 kernels. Clone of main venv with Mamba-3 files manually copied from bare repo clone of `state-spaces/mamba`. Additional deps: `tilelang`, `quack-kernels`, `cuda-bindings`, `nvidia-cutlass-dsl`.

## Cloud Setup

`setup.sh` automates fresh cloud GPU instance setup (Vast.ai / RunPod): clones repo, installs deps, downloads PolSESS from Google Drive, configures env vars. Use `--no-data` if dataset is already mounted.

## Environment Variables

- `POLSESS_DATA_ROOT` — PolSESS dataset path (default in `config.py`)
- `REALM_DATA_ROOT` — REAL-M dataset (default: `~/datasets/REAL-M-v0.1.0/`)
- `LIBRIMIX_ASR_ROOT` — LibriSpeechMixASR (default: `~/datasets/LibriSpeechMixASR/`)
