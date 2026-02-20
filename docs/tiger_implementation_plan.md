# TIGER Model Integration Plan

## Background

**TIGER** (Time-frequency Interleaved Gain Extraction and Reconstruction) is a lightweight speech separation model published at **ICLR 2025** by the same research group (Kai Li et al., Tsinghua) that produced SPMamba — which we already have integrated.

- **Paper**: [arxiv.org/abs/2410.01469](https://arxiv.org/abs/2410.01469)
- **GitHub**: [github.com/JusperLee/TIGER](https://github.com/JusperLee/TIGER)
- **Pre-trained models**: Available on HuggingFace (TIGER-speech, TIGER-speech-small, TIGER-speech-tiny)

### Why TIGER?

| Metric | TF-GridNet | SPMamba | TIGER (small) | TIGER (large) |
|--------|-----------|---------|---------------|---------------|
| Params | 14.43M | ~2.0M | **0.82M** | **0.82M** |
| MACs (G/s) | 323.75 | ~50 | **7.65** | **15.27** |
| SI-SDRi (EchoSet) | 12.85 dB | — | 12.58 dB | **13.73 dB** |
| SI-SDRi (Libri2Mix) | 19.24 dB | — | 16.67 dB | 17.97 dB |
| GPU Inference (ms) | 94.30 | — | 42.38 | 74.51 |

TIGER achieves near-SOTA quality with **94.3% fewer parameters** than TF-GridNet. On complex acoustic environments (noise + reverb), it actually **surpasses** TF-GridNet. The small version (B=4 FFI blocks) is ~2.2× faster than TF-GridNet on GPU.

### Does it fit our project?

> [!IMPORTANT]
> TIGER was designed and evaluated at **16kHz**. PolSESS is **8kHz**. The band-split scheme computes sub-band widths dynamically from `sample_rate`, so it will adapt automatically. However, the specific sub-band boundaries (0–1kHz narrow, 1–4kHz medium, 4–8kHz wide) are tuned for 16kHz speech. At 8kHz, the Nyquist limit is 4kHz, meaning the 4–8kHz bands are irrelevant — the model will still work but the band-split distribution may be suboptimal for 8kHz. We should test both:
> 1. The default scheme with `sample_rate=8000` (auto-adapted)
> 2. A custom scheme tuned for 8kHz (fewer bands, all ≤4kHz)

**Architecture compatibility**: TIGER is STFT-based (like SPMamba), takes `[B, T]` waveform input, and outputs `[B, n_sources, T]` — matching our existing interface. No Mamba dependency (pure PyTorch: Conv1d, Conv2d, self-attention).

**Training cost**: With 0.82M params vs SPMamba's ~2M, TIGER should train faster and use less VRAM. The shared-parameter FFI blocks (iterated B times) keep memory low.

---

## Architecture Overview

```
Input waveform [B, T]
    │
    ▼
STFT Encoder → complex spectrogram [B, F, T]
    │
    ▼
Band-Split Module → K sub-bands → bottleneck to N channels → [B, K, N, T]
    │
    ▼
Separator (B × shared FFI blocks):
    ├── Frequency path: MSA (multi-scale conv U-Net) + F³A (self-attention)
    └── Frame path:     MSA (multi-scale conv U-Net) + F³A (self-attention)
    │
    ▼
Band-Restoration Module → complex masks [B, n_src, F, T]
    │
    ▼
Apply masks + iSTFT → separated waveforms [B, n_src, T]
```

**Key components**:
- **MSA** (Multi-Scale Selective Attention): A convolutional U-Net that downsamples frequency/time dims progressively, fuses global+local features via selective attention (sigmoid gating), then decodes back. No attention mechanism — purely convolutional.
- **F³A** (Full-Frequency-Frame Attention): Multi-head self-attention that operates on the full K×T or T×K dimensions by merging channel and time/freq into the embedding dim. This gives cross-band / cross-frame context.
- **FFI block**: Runs frequency-path (MSA→F³A) then frame-path (MSA→F³A), with shared weights iterated B times.

---

## Proposed Changes

### Config

#### [MODIFY] [config.py](file:///home/user/polsess_separation/config.py)

Add `TIGERParams` dataclass and register it in `ModelConfig`:

```python
@dataclass
class TIGERParams:
    """TIGER (Time-frequency Interleaved Gain Extraction and Reconstruction) parameters."""
    out_channels: int = 128        # Feature dimension N per sub-band
    in_channels: int = 512         # Hidden dim in MSA U-Net
    num_blocks: int = 4            # FFI block iterations (4=small, 8=large)
    upsampling_depth: int = 4      # MSA downsampling depth D
    att_n_head: int = 4            # F³A attention heads
    att_hid_chan: int = 4           # F³A hidden channel E per head
    n_fft: int = 256               # STFT window size (256 → 128+1 freq bins at 8kHz)
    hop_length: int = 64           # STFT hop length
    n_srcs: int = 1                # 1=enhancement, 2=separation
    sample_rate: int = 8000        # PolSESS sample rate
```

> [!NOTE]
> The original paper uses `win=640, stride=160` for 16kHz (40ms/10ms). For 8kHz, equivalent timing would be `win=320, stride=80` (also 40ms/10ms). Alternatively, we can use `n_fft=256, hop_length=64` to match SPMamba's existing config. This is a hyperparameter to tune.

---

### Model

#### [NEW] [tiger.py](file:///home/user/polsess_separation/models/tiger.py)

Self-contained TIGER implementation (~500 lines), ported from the official repo with these changes:

1. **Remove `look2hear` dependency**: Inline the two external helpers:
   - `activations.get("prelu")` → just use `nn.PReLU()` directly
   - `normalizations.get("LayerNormalization4D")` → copy the `LayerNormalization4D` class (~20 lines) into the file
   - Remove `BaseModel` inheritance → use plain `nn.Module`

2. **Remove `print(self.band_width)`** debug statement (line 547 in original)

3. **Adapt I/O to match project convention**:
   - Input: `[B, 1, T]` or `[B, T]` → squeeze to `[B, T]` (same as SPMamba)
   - Output: `[B, T]` if `n_srcs=1`, `[B, n_srcs, T]` if `n_srcs>1` (same as SPMamba)
   - Add RMS normalization before STFT and denormalization after iSTFT (same as SPMamba)

4. **Parameterize `n_fft` and `hop_length`** instead of `win` and `stride` (naming consistency with SPMamba)

**The core architecture (band-split, UConvBlock/MSA, MultiHeadSelfAttention2D/F³A, Recurrent/FFI separator, mask estimation) stays identical to the paper.**

Internal classes to include:
- `GlobLN`, `ConvNormAct`, `ConvNorm`, `DilatedConvNorm` — basic conv building blocks
- `Mlp` — MLP with depthwise conv
- `InjectionMultiSum`, `InjectionMulti` — selective attention gating (SA module)
- `UConvBlock` — multi-scale selective attention (MSA module)
- `LayerNormalization4D` — 4D layer norm (from `look2hear/layers/normalizations.py`)
- `ATTConvActNorm` — conv+act+norm wrapper for attention projections
- `MultiHeadSelfAttention2D` — full-frequency-frame attention (F³A module)
- `Recurrent` — FFI block iterated B times (the separator)
- `TIGER` — top-level model class

---

### Model Registry

#### [MODIFY] [\_\_init\_\_.py](file:///home/user/polsess_separation/models/__init__.py)

```diff
 from .spmamba import SPMamba
+from .tiger import TIGER

 MODELS = {
     'convtasnet': ConvTasNet,
     'sepformer': SepFormer,
     'dprnn': DPRNN,
     'spmamba': SPMamba,
+    'tiger': TIGER,
 }
```

---

### YAML Config

#### [NEW] [configs/tiger_small.yaml](file:///home/user/polsess_separation/configs/tiger_small.yaml)

```yaml
model:
  model_type: tiger
  tiger:
    out_channels: 128
    in_channels: 512
    num_blocks: 4          # small version
    upsampling_depth: 4
    att_n_head: 4
    att_hid_chan: 4
    n_fft: 320             # 40ms window at 8kHz
    hop_length: 80          # 10ms hop at 8kHz
    n_srcs: 1               # ES task
    sample_rate: 8000
```

---

### Tests

#### [NEW] [tests/test_tiger.py](file:///home/user/polsess_separation/tests/test_tiger.py)

- `test_tiger_forward_shape_single_source`: Verify `[B, T]` output for `n_srcs=1`
- `test_tiger_forward_shape_two_sources`: Verify `[B, 2, T]` output for `n_srcs=2`
- `test_tiger_param_count`: Verify <1M params for small config
- `test_tiger_band_split_8khz`: Verify band-split produces valid sub-bands at 8kHz
- `test_tiger_registered_in_factory`: Verify `get_model("tiger")` works

---

## Adaptation Notes for 8kHz

The band-split scheme in TIGER is computed dynamically:
```python
bandwidth_25  = floor(25 / (sr/2) * enc_dim)   # ~25Hz per band, 40 bands
bandwidth_100 = floor(100 / (sr/2) * enc_dim)   # ~100Hz per band, 10 bands
bandwidth_250 = floor(250 / (sr/2) * enc_dim)   # ~250Hz per band, 8 bands
bandwidth_500 = floor(500 / (sr/2) * enc_dim)   # ~500Hz per band, 8 bands
```

At 16kHz with `n_fft=640`: `enc_dim=321`, Nyquist=8kHz → 67 sub-bands covering 0–8kHz.

At 8kHz with `n_fft=320`: `enc_dim=161`, Nyquist=4kHz → the 500Hz bands (covering 4–8kHz) map to nothing useful. The code will still produce 67 "sub-bands" but the last ~8 are empty or near-empty.

> [!WARNING]
> **Action needed after initial integration**: Run a quick sanity check that the auto-computed `band_width` list at 8kHz is sensible (i.e., all values ≥1, sum equals `enc_dim`). If any sub-band has width 0, we need to adjust the scheme. A custom 8kHz scheme could be:
> - 0–500Hz: 25Hz bands (20 bands)
> - 500–1kHz: 50Hz bands (10 bands)
> - 1–2kHz: 100Hz bands (10 bands)
> - 2–4kHz: 250Hz bands (8 bands)

---

## Verification Plan

### Automated Tests
```bash
pytest tests/test_tiger.py -v
```

### Manual Verification
1. Create model, run dummy forward pass, confirm output shapes
2. Print `band_width` list at 8kHz and verify all values ≥ 1
3. Count parameters and confirm <1M for small config
4. Run a short training (~5 epochs) on a small subset to confirm loss decreases

---

## Effort Estimate

| Item | Estimate |
|------|----------|
| Port TIGER model (remove deps, adapt I/O) | ~30 min |
| Add config + registry | ~10 min |
| Write tests | ~15 min |
| Verify & debug 8kHz band-split | ~15 min |
| **Total** | **~1 hour** |
