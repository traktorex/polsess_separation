# Plan: Add Diffusion-Based Post-Processing Refinement ("Separate And Diffuse")

## Context

The user wants to add a diffusion-based post-processing module that can improve any separator model's output, based on the "Separate And Diffuse" paper (arXiv 2301.10752). The module should be self-contained in a separate folder with minimal invasion of existing code. The project trains on PolSESS (8kHz Polish speech) with 12GB VRAM (RTX 4070).

**Paper pipeline (3 stages):**
1. Backbone separator → initial source estimates (already exists — 7 models)
2. Pretrained DiffWave vocoder → refines each estimate via mel-spectrogram conditioning
3. Learned alignment network F (dual-head 6-layer ResNet) → complex-valued frequency-domain mixing of original + refined

Only F is trained (SI-SDR loss). DiffWave is pretrained on clean speech and frozen.

**DiffWave pretraining:** The paper authors trained DiffWave themselves on clean training sources — "the DiffWave was trained separately over the training sets' sources (no mixing)." All publicly available DiffWave checkpoints (lmnt-com, philsyn, SpeechBrain HuggingFace) are at 22,050 Hz on LJSpeech. No 8kHz checkpoints exist. A 22kHz model cannot be used on 8kHz audio (mel spectrogram statistics and temporal resolution are fundamentally different). We must train DiffWave on PolSESS clean speech at 8kHz.

**DiffWave implementation option:** The `diffwave` pip package (lmnt-com) defaults to 22kHz but the sample rate is **configurable** — we could use it with modified params instead of reimplementing from scratch. Decision TBD at implementation time (trade-off: pip package = less code but external dependency; from scratch = ~250 lines but full control + thesis-readable).

---

## File Structure

```
polsess_separation/
├── refinement/                        # NEW — all diffusion refinement code
│   ├── __init__.py                    # Public API: RefinementPipeline, AlignmentNetwork, DiffWaveModel
│   ├── diffwave.py                    # DiffWave vocoder (~250 lines)
│   ├── alignment.py                   # Alignment network F (~100 lines)
│   ├── pipeline.py                    # RefinementPipeline wrapper (~100 lines)
│   ├── mel.py                         # Mel-spectrogram helper (~30 lines)
│   ├── train_diffwave.py              # Script: pretrain DiffWave on clean PolSESS speech
│   └── train_alignment.py             # Script: train alignment network F
├── experiments/refinement/             # NEW
│   ├── diffwave_pretrain.yaml         # DiffWave pretraining config
│   └── alignment_train.yaml           # Alignment training config (per-separator)
├── tests/test_refinement.py           # NEW
└── evaluate.py                        # MODIFIED: +2 CLI args, +10 lines
```

**No changes to:** config.py, models/, training/trainer.py, train.py

---

## New Files (7)

### 1. `refinement/mel.py` — Mel-spectrogram helper (~30 lines)

Thin wrapper around `torchaudio.transforms.MelSpectrogram` with 8kHz defaults:
- `sample_rate=8000`, `n_fft=512`, `hop_length=128`, `n_mels=80`, `f_max=4000`
- Includes `log_mel()` (log1p scaling) for DiffWave conditioning
- Single source of truth for mel parameters — used by both DiffWave training and inference

### 2. `refinement/diffwave.py` — DiffWave vocoder (~250 lines)

**Option A:** Use `diffwave` pip package (lmnt-com) with modified params for 8kHz — less code, but external dependency.
**Option B:** Implement from scratch — ~250 lines, full control, thesis-readable.

**Architecture** (follows arXiv 2009.09761):
- `DiffusionEmbedding` — sinusoidal timestep encoding
- `ResidualBlock` — dilated Conv1d + mel condition + gated activation + 1x1 residual/skip
- `DiffWaveModel(nn.Module)`:
  - `residual_channels=64` (reduced from paper's 256 for 8kHz + 12GB VRAM)
  - `n_layers=30`, `dilation_cycle=10` (dilations 2^0...2^9, repeated 3×)
  - `forward(x_t, t, mel)` — training: predicts noise epsilon
  - `inference(mel, n_steps=50)` — reverse diffusion: mel → waveform
  - Noise schedule: linear beta from 1e-4 to 0.05, stored as buffers

**Mel conditioning:** Upsampled to waveform length via 2 transposed conv layers (total stride = hop_length=128). Added into each residual layer.

**Must be trained from scratch** on PolSESS clean speech at 8kHz (no compatible pretrained checkpoints exist).

**Estimated params:** ~2-3M with 64 channels.

### 3. `refinement/alignment.py` — Alignment network F (~150 lines)

**Corrected to match the paper exactly.** The paper describes F as a "dual 6-layer convolutional neural network with residual connections" (ResNet Head). It has TWO identical sub-networks (heads):

```python
class AlignmentNetwork(nn.Module):
    """Dual-head ResNet for complex-valued STFT combination (paper Section 4).

    Two heads with shared architecture but separate weights:
    - Magnitude head: processes concatenated STFT magnitudes
    - Phase head: processes phase of V_d + relative phase angle(V_g · conj(V_d))

    Outputs complex weights α, β for: V̄ = α·V_d + β·V_g
    """

    def __init__(self, n_fft=512, hop_length=128):
        # Each head: 6 Conv2d layers with residual connections
        # Hidden channels: 32, 32, 64, 64, 64 (paper Section 4)
        # Kernel size: 3×3, padding=1
        # Activations: not specified in paper — use ReLU (standard for ResNets)
        #
        # Magnitude head input: Concat(|V_d|, |V_g|) → [B, 2, F, T]
        # Phase head input: Concat(angle(V_d), angle(V_g·conj(V_d))) → [B, 2, F, T]
        # Each head output: [B, 2, F, T] (2 channels for α and β)

    def forward(self, original_wav, diffwave_wav):
        # 1. STFT both: V_d = STFT(original), V_g = STFT(diffwave)
        # 2. Build magnitude input A = Concat(|V_d|, |V_g|) → [B, 2, F, T]
        # 3. Build phase input ψ = Concat(angle(V_d), angle(V_g·conj(V_d))) → [B, 2, F, T]
        # 4. D1 = magnitude_head(A)  → [B, 2, F, T]
        # 5. D2 = phase_head(ψ)     → [B, 2, F, T]
        # 6. Q = D1 · exp(-j · D2)  → complex [B, 2, F, T]
        # 7. α = Q[:, 0], β = Q[:, 1]  (complex per-bin weights)
        # 8. V̄ = α · V_d + β · V_g  (complex element-wise)
        # 9. ISTFT(V̄) → output waveform [B, T]
```

**Key difference from initial plan:** The paper uses complex-valued combination with learned magnitude AND phase weights, not simple real-valued magnitude blending. This gives F the ability to correct phase alignment between the two signals at each time-frequency bin.

STFT params: `n_fft=512, hop_length=128`. ~100-150K parameters (two heads).

**Paper does NOT specify:** exact STFT parameters, activation functions, normalization, or padding. We use reasonable defaults matching our mel.py.

### 4. `refinement/pipeline.py` — RefinementPipeline (~100 lines)

```python
class RefinementPipeline(nn.Module):
    """Post-processing: DiffWave refinement + alignment network."""

    def __init__(self, diffwave, alignment, inference_steps=50):
        # DiffWave always frozen (eval mode, no_grad)

    @classmethod
    def from_checkpoints(cls, diffwave_path, alignment_path, device='cuda'):
        # Load both models from checkpoint files

    def forward(self, separator_output):
        # separator_output: [B, T] or [B, C, T]
        # For C>1: process each source independently
        # 1. mel = compute_log_mel(source)
        # 2. refined = diffwave.inference(mel)
        # 3. output = alignment(source, refined)
        # Return same shape as input
```

### 5. `refinement/train_diffwave.py` — DiffWave pretraining script

**Standalone script** (not using existing `Trainer` — fundamentally different training loop).

**Data:** Clean speech from `{data_root}/train/clean/` (~48K files). Inline `CleanSpeechDataset`:
- Loads wav files, randomly crops to `segment_length` (4s = 32K samples at 8kHz)
- Computes mel-spectrogram on the fly using `mel.py`
- Returns `(mel, waveform)` pairs

**Training loop:**
1. Sample clean waveform `x_0`
2. Sample random timestep `t ~ Uniform(1, T)`
3. Add noise: `x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε`
4. Model predicts: `ε̂ = DiffWave(x_t, t, mel(x_0))`
5. Loss: `MSE(ε̂, ε)`

**Features:** AMP, wandb logging, checkpoint saving, LR scheduling — follows same conventions as `training/trainer.py`.

**VRAM:** ~3-4GB (batch=8, AMP). Comfortable on 12GB.

**CLI:**
```bash
python -m refinement.train_diffwave --config experiments/refinement/diffwave_pretrain.yaml
```

### 6. `refinement/train_alignment.py` — Alignment training script

**Data pipeline:**
1. Load batch from PolSESS (mix + clean)
2. `torch.no_grad()`: frozen separator → initial estimates
3. `torch.no_grad()`: frozen DiffWave → refined estimates
4. Gradient-tracked: alignment network F → final output
5. SI-SDR loss against clean target, backprop through F only

**Key VRAM optimization:** Separator and DiffWave run in `torch.no_grad()` blocks — no activation storage for backprop. Only F's tiny computational graph is tracked.

**VRAM:** ~5-8GB total (separator inference 1-3GB + DiffWave inference 2GB + F training 0.5GB).

**CLI:**
```bash
python -m refinement.train_alignment --config experiments/refinement/alignment_train.yaml \
    --separator-checkpoint checkpoints/spmamba/SB/.../best.pt \
    --diffwave-checkpoint checkpoints/diffwave/best.pt
```

### 7. `tests/test_refinement.py` — Tests (~80 lines)

All CPU, small dimensions:
- `test_mel_shape()` — correct output shape for 8kHz audio
- `test_diffwave_init()` — instantiation with small config
- `test_diffwave_training_forward()` — noise prediction shape
- `test_diffwave_inference()` — reverse diffusion produces waveform
- `test_alignment_forward_shape()` — output matches input shape
- `test_alignment_dual_head()` — magnitude and phase heads produce correct shapes
- `test_alignment_complex_weights()` — α, β are complex-valued
- `test_pipeline_enhancement()` — handles [B, T]
- `test_pipeline_separation()` — handles [B, C, T]
- `test_diffwave_frozen_in_pipeline()` — requires_grad=False

---

## Modified File (1)

### `evaluate.py` — Add optional refinement (~15 lines)

**New CLI args** (lines ~302):
```python
parser.add_argument("--refinement-checkpoint", help="Path to alignment network checkpoint")
parser.add_argument("--diffwave-checkpoint", help="Path to DiffWave checkpoint")
```

**Pipeline construction in `main()`** (after model loading, ~line 335):
```python
refinement_pipeline = None
if args.refinement_checkpoint:
    from refinement import RefinementPipeline
    refinement_pipeline = RefinementPipeline.from_checkpoints(
        diffwave_path=args.diffwave_checkpoint,
        alignment_path=args.refinement_checkpoint,
        device=device,
    )
```

**Apply in `evaluate_model()`** — add `refinement_pipeline=None` parameter, insert after line 76:
```python
if refinement_pipeline is not None:
    estimates = refinement_pipeline(estimates)
```

**Impact:** When flags are not passed, behavior is 100% identical to current code. Lazy import means `refinement/` is never loaded unless explicitly requested.

---

## Experiment YAMLs (2)

### `experiments/refinement/diffwave_pretrain.yaml`
```yaml
data:
  clean_speech_dir: /home/user/datasets/PolSESS_C_both/PolSESS_C_both
  sample_rate: 8000
  segment_length: 32000    # 4 seconds

diffwave:
  residual_channels: 64
  n_layers: 30
  dilation_cycle: 10

training:
  lr: 2e-4
  batch_size: 8
  num_epochs: 200
  use_amp: true
  save_dir: checkpoints/diffwave
  use_wandb: true
  wandb_project: polsess-separation
  wandb_run_name: diffwave-pretrain
```

### `experiments/refinement/alignment_train.yaml`
```yaml
data:
  dataset_type: polsess
  batch_size: 4
  task: SB

alignment:
  n_fft: 512
  hop_length: 128
  n_conv_layers: 6
  hidden_channels: [32, 32, 64, 64, 64]  # Paper Section 4

training:
  lr: 1e-3
  num_epochs: 50
  use_amp: true
  save_dir: checkpoints/refinement
  use_wandb: true
  wandb_project: polsess-separation
  wandb_run_name: alignment-spmamba
```

---

## Implementation Order

1. `refinement/__init__.py` + `refinement/mel.py` (scaffold + mel helper)
2. `refinement/diffwave.py` (DiffWave model)
3. `refinement/alignment.py` (alignment network F)
4. `refinement/pipeline.py` (wiring)
5. `tests/test_refinement.py` (verify all components)
6. `refinement/train_diffwave.py` + `experiments/refinement/diffwave_pretrain.yaml`
7. `refinement/train_alignment.py` + `experiments/refinement/alignment_train.yaml`
8. `evaluate.py` (add 2 CLI args + pipeline application)

Steps 2-3 are independent. Step 5 can run after steps 1-4.

---

## Usage Flow (After Implementation)

```bash
# Phase 1: Pretrain DiffWave on clean PolSESS speech (~8-24h)
python -m refinement.train_diffwave --config experiments/refinement/diffwave_pretrain.yaml

# Phase 2: Train alignment network for a specific separator (~2-4h)
python -m refinement.train_alignment --config experiments/refinement/alignment_train.yaml \
    --separator-checkpoint checkpoints/spmamba/SB/.../best.pt \
    --diffwave-checkpoint checkpoints/diffwave/best.pt

# Phase 3: Evaluate with refinement
python evaluate.py --checkpoint checkpoints/spmamba/SB/.../best.pt \
    --diffwave-checkpoint checkpoints/diffwave/best.pt \
    --refinement-checkpoint checkpoints/refinement/best.pt

# Compare: evaluate without refinement (existing behavior, unchanged)
python evaluate.py --checkpoint checkpoints/spmamba/SB/.../best.pt
```

---

## Verification

1. `pytest tests/test_refinement.py -v` — all component tests pass (CPU)
2. `pytest` — existing tests unaffected
3. Smoke test DiffWave pretraining (1-2 epochs, check loss decreases)
4. Smoke test alignment training (1-2 epochs with a trained separator)
5. Evaluate with and without refinement — compare SI-SDR (expect +1-1.5 dB improvement per paper)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Train DiffWave from scratch at 8kHz | No 8kHz pretrained checkpoints exist. Paper authors also trained theirs on clean training sources. All available checkpoints are 22kHz LJSpeech — incompatible with 8kHz |
| DiffWave impl: TBD (pip pkg vs from scratch) | `diffwave` pip (lmnt-com) is configurable for sample rate. Trade-off: pip = less code; from scratch = thesis-readable, no dependency |
| Dual-head alignment network (matching paper) | Paper uses separate magnitude + phase heads with complex-valued output weights α, β. Simpler single-head real-valued blending would not match the paper |
| Standalone training scripts (not reusing Trainer) | Fundamentally different training loops (noise prediction vs SI-SDR). Sharing Trainer would require invasive changes |
| 64 residual channels (not 256) | 8kHz has half the bandwidth of 16kHz. 64 channels sufficient + fits 12GB VRAM |
| Lazy import in evaluate.py | Zero overhead when refinement not used |
| One alignment checkpoint per separator | Paper trains F on one separator's outputs. Simplest approach for thesis |

## Files Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `refinement/__init__.py` | **New** | ~10 |
| `refinement/mel.py` | **New** | ~30 |
| `refinement/diffwave.py` | **New** | ~250 |
| `refinement/alignment.py` | **New** | ~100 |
| `refinement/pipeline.py` | **New** | ~100 |
| `refinement/train_diffwave.py` | **New** | ~200 |
| `refinement/train_alignment.py` | **New** | ~200 |
| `tests/test_refinement.py` | **New** | ~80 |
| `experiments/refinement/diffwave_pretrain.yaml` | **New** | ~25 |
| `experiments/refinement/alignment_train.yaml` | **New** | ~25 |
| `evaluate.py` | **Edit** | +15 |

**Existing code reused (unchanged):** `utils/model_utils.py` (load_model_for_inference), all models, training/trainer.py, config.py
