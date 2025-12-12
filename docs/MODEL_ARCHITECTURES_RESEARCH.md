# Speech Separation Model Architectures - Research Report

**Date**: 2025-11-29
**Purpose**: Evaluate model architectures for integration into polsess_separation project
**Status**: Research Complete

---

## Executive Summary

This report analyzes the current ConvTasNet implementation and evaluates three categories of speech separation models for potential integration: Transformer-based, RNN-based, and State-Space models.

### Top 3 Recommendations

1. **SepFormer** (Transformer) - Priority 1: Easiest integration, excellent performance (22.4 dB SI-SNRi)
2. **DPRNN** (RNN) - Priority 2: Good efficiency, established architecture (18.8 dB SI-SNRi)
3. **SPMamba** (State-Space) - Priority 3: Cutting-edge 2024 model (22.5 dB SI-SNRi)

---

## Current Architecture Analysis

### ConvTasNet Implementation

**Location**: `models/conv_tasnet.py`

**Architecture**:
- Encoder: Learnable basis functions (N=256 filters)
- Separation Network: Temporal convolutional network with mask estimation
- Decoder: Reconstruction from masked encoded representations

**Key Parameters**:
```python
N: int = 256          # Encoder filter dimension
B: int = 256          # Bottleneck channels
H: int = 512          # Convolutional block channels
P: int = 3            # Kernel size
X: int = 8            # Blocks per repeat
R: int = 4            # Number of repeats
C: int = 1 or 2       # Output sources (task-dependent)
kernel_size: int = 16 # Encoder kernel
stride: int = 8       # Encoder stride
```

**Performance**: SI-SNRi = 15.3 dB on WSJ0-2Mix

---

## Model Integration Requirements

### Required Model Interface

Every model must implement:

```python
class SeparationModel(nn.Module):
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: [B, T] or [B, 1, T] audio waveform
        Returns:
            separated: [B, T] for C=1 or [B, C, T] for C>1
        """
        pass

    def get_num_sources(self) -> int:
        """Return number of output sources"""
        pass
```

### Data Flow

```
Dataset → DataLoader → Trainer
  [T] → [B,T] → [B,1,T] → Model → [B,T] or [B,C,T] → Loss → Backprop
```

**Key Integration Points**:
1. Input preprocessing: `mix.unsqueeze(1)` in trainer.py line 165
2. Loss function routing: Task-based (ES/EB/SB) in trainer.py lines 55-67
3. Output handling: Variable shape based on task
4. Checkpoint loading: Model reconstruction in evaluate.py

---

## Model Categories

### 1. Transformer-Based Models

#### SepFormer (RECOMMENDED)

**Architecture**: Dual-path transformer with multi-head self-attention

**Performance**:
- WSJ0-2Mix (dynamic mixing): SI-SNRi = 22.4 dB, SDRi = 22.4 dB
- WSJ0-3Mix (dynamic mixing): SI-SNRi = 19.8 dB, SDRi = 19.7 dB
- Improvement over ConvTasNet: +7 dB

**Parameters**: 26.0M
**Computational Cost**: 3.16 MACs

**Implementation**:
```python
from speechbrain.inference.separation import SepformerSeparation

model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir='pretrained_models/sepformer-wsj02mix'
)
est_sources = model.separate_file('mixture.wav')
```

**Integration Difficulty**: EASY
**License**: Apache 2.0 (commercial-friendly)
**Source**: https://github.com/speechbrain/speechbrain
**Pretrained Models**: HuggingFace Hub

**Pros**:
- Excellent SpeechBrain integration (3-4 lines of code)
- Best documentation and community support
- Multiple pretrained models available
- State-of-the-art performance among established models
- Production-ready

**Cons**:
- Larger model size (26M params vs 5.6M for ConvTasNet)
- Higher computational cost
- Slower inference than ConvTasNet

**Estimated Integration Time**: 30 minutes

---

#### DPTNet (Dual-Path Transformer Network)

**Architecture**: Dual-path processing with RNN-augmented transformer

**Performance**:
- WSJ0-2Mix: SI-SNRi = 20.6 dB
- LS-2mix: SI-SNRi = 16.8 dB

**Implementation**:
- Official: https://github.com/ujscjj/DPTNet
- Also available in SpeechBrain recipes

**Integration Difficulty**: MEDIUM
**License**: Check repository

**Pros**:
- Good performance (20.6 dB)
- Dual-path efficiency
- Available in SpeechBrain

**Cons**:
- Lower performance than SepFormer
- Less documentation
- Less mature ecosystem

---

#### MossFormer2

**Architecture**: Hybrid transformer + FSMN (RNN-free recurrent)

**Performance**:
- WSJ0-2Mix: SI-SNRi = 24.1 dB (near state-of-the-art)
- WSJ0-3Mix: State-of-the-art
- WHAM!/WHAMR!: State-of-the-art

**Parameters**: 55.7M

**Implementation**:
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

separation = pipeline(
    Tasks.speech_separation,
    model='damo/speech_mossformer2_separation_temporal_8k'
)
result = separation('mixture.wav')
```

**Integration Difficulty**: MEDIUM-HARD
**Source**: https://github.com/alibabasglab/MossFormer2
**License**: Check repository

**Pros**:
- Highest performance (24.1 dB)
- ModelScope provides simple API
- Multi-platform support

**Cons**:
- Requires ModelScope ecosystem (Alibaba)
- Largest model size (55.7M)
- Not in mainstream toolkits
- Less familiar to Western developers
- Requires libsndfile on Linux

**Recommended**: Only if absolute best performance needed

---

### 2. RNN-Based Models

#### DPRNN (Dual-Path RNN) - RECOMMENDED

**Architecture**: Dual-path recurrent processing with chunk-based computation

**Performance**:
- WSJ0-2Mix: SI-SNRi = 18.8 dB, SDRi = 18.8 dB
- Better than ConvTasNet (+3.5 dB)
- Lower than SepFormer (-3.6 dB)

**Parameters**: 2.7M (very efficient)
**Computational Cost**: 4.36 MACs
**Latency**: 207.76ms CPU, 116ms real-time

**Implementation via Asteroid**:
```python
from asteroid.models import DPRNNTasNet

model = DPRNNTasNet.from_pretrained(
    'JorisCos/DPRNNTasNet-ks16_WHAM_sepclean'
)
separated = model.separate(mixture_waveform)
```

**Implementation via SpeechBrain**: Also available in recipes

**Integration Difficulty**: EASY
**License**: MIT (Asteroid), Apache 2.0 (SpeechBrain)
**Source**:
- https://github.com/asteroid-team/asteroid
- https://github.com/speechbrain/speechbrain

**Pros**:
- Available in two major toolkits
- Smallest parameter count (2.7M)
- Good performance/efficiency trade-off
- Well-established (2019, many citations)
- Easy PyTorch Hub integration
- MIT license

**Cons**:
- Lower performance than Transformer models
- Higher latency than ConvTasNet
- Sequential processing limitations

**Estimated Integration Time**: 1 hour

---

#### LSTM-TasNet / Conv-TasNet Variants

**Note**: Original LSTM-based TasNet variants have been superseded by Conv-TasNet

**Performance**: ConvTasNet outperforms LSTM variants
**Recommendation**: Use Conv-TasNet (already implemented) or DPRNN instead

**Integration Difficulty**: EASY (via Asteroid)

---

### 3. State-Space Models

#### SPMamba - RECOMMENDED

**Architecture**: Bidirectional Mamba modules (state-space model)

**Innovation**: Linear computational complexity for long-range dependencies
**Base**: Built on TF-GridNet, replacing BLSTM with Mamba

**Performance**:
- WSJ0-2Mix: SI-SNRi = 22.5 dB, SDRi = 22.7 dB
- WHAM!: SI-SNRi = 17.4 dB, SDRi = 17.6 dB
- Echo2Mix: SI-SNRi = 15.3 dB, SDRi = 16.1 dB (SOTA)
- Improvement over TF-GridNet: +2.42 dB SI-SNRi

**Parameters**: 6.1M (highly efficient)
**Computational Cost**: 238.21 G/s MACs (half of TF-GridNet)

**Implementation**:
```bash
git clone https://github.com/JusperLee/SPMamba.git
conda env create -f look2hear.yml
conda activate look2hear
```

**Integration Difficulty**: MEDIUM
**License**: Apache 2.0 (commercial-friendly)
**Source**: https://github.com/JusperLee/SPMamba
**Latest Release**: November 2024

**Pros**:
- State-of-the-art efficiency (6.1M params)
- Strong performance (22.5 dB)
- Apache 2.0 license
- Recent checkpoints available
- Well-documented

**Cons**:
- Custom codebase (not in standard toolkit)
- Requires specific conda environment
- Less community adoption (new 2024 model)
- Standalone integration needed

**Estimated Integration Time**: 2-3 hours

---

## Performance Comparison

| Model | SI-SNRi (dB) | Parameters | MACs | Latency (ms) | Year | Toolkit |
|-------|--------------|------------|------|--------------|------|---------|
| MossFormer2 | 24.1 | 55.7M | - | - | 2024 | ModelScope |
| SepFormer | 22.4* | 26.0M | 3.16 | 168.45 | 2020 | SpeechBrain |
| SPMamba | 22.5 | 6.1M | 238.21 | - | 2024 | Standalone |
| DPTNet | 20.6 | - | - | - | 2020 | SpeechBrain |
| DPRNN | 18.8 | 2.7M | 4.36 | 207.76 | 2019 | Asteroid/SB |
| ConvTasNet | 15.3 | 5.6M | 0.40 | 34.03 | 2018 | SpeechBrain |

*With dynamic mixing augmentation

**Key Insights**:
- ConvTasNet: Fastest, lowest latency, good for real-time
- DPRNN: Most efficient (2.7M params), good balance
- SepFormer: Best performance/ease ratio
- SPMamba: Best efficiency/performance (6.1M params, 22.5 dB)
- MossFormer2: Highest performance but complex integration

---

## Toolkit Comparison

### SpeechBrain

**License**: Apache 2.0
**Models**: SepFormer, RE-SepFormer, DPRNN, ConvTasNet, DPTNet
**Integration**: EASIEST (simple API)
**Documentation**: Excellent
**Pretrained Models**: HuggingFace Hub
**Best For**: Production use, quick integration

**Example**:
```python
from speechbrain.inference.separation import SepformerSeparation
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix"
)
```

---

### Asteroid

**License**: MIT (code), CC BY-SA 3.0 (models)
**Models**: ConvTasNet, DPRNN, DPTNet, 13+ others
**Integration**: Easy (PyTorch Hub)
**Documentation**: Good
**Pretrained Models**: Asteroid Hub, HuggingFace
**Note**: Models trained on WSJ0 are non-commercial only
**Best For**: Research, flexibility

**Example**:
```python
from asteroid.models import DPRNNTasNet
model = DPRNNTasNet.from_pretrained('model_id')
```

---

### Standalone (SPMamba, MossFormer2)

**Integration**: Custom setup
**Documentation**: Varies
**Best For**: Cutting-edge performance, research

---

## Integration Requirements

### Input/Output Compatibility

**Common Format**:
- Input: Raw waveform (time-domain)
- Sample Rates: 8kHz or 16kHz (model-specific)
- Format: Single-channel WAV
- Auto-resampling: Most frameworks support

**Standard Pattern**:
```python
import torchaudio

# Load
mixture, sr = torchaudio.load('mixture.wav')

# Separate
sources = model.separate(mixture)

# Save
for i, source in enumerate(sources):
    torchaudio.save(f'source_{i}.wav', source, sr)
```

---

### Dependencies

**SpeechBrain**:
```bash
pip install speechbrain
```

**Asteroid**:
```bash
pip install asteroid
```

**SPMamba**:
```bash
conda env create -f look2hear.yml
```

**MossFormer2**:
```bash
pip install modelscope soundfile
```

---

## License Summary

| Model/Toolkit | Code License | Model License | Commercial Use |
|---------------|--------------|---------------|----------------|
| SpeechBrain | Apache 2.0 | Apache 2.0 | Yes |
| Asteroid | MIT | CC BY-SA 3.0 | Depends* |
| SPMamba | Apache 2.0 | Apache 2.0 | Yes |
| MossFormer2 | Check repo | Check repo | Unknown |

*Asteroid models trained on WSJ0 are non-commercial

---

## Recommended Implementation Priority

### Priority 1: SepFormer (via SpeechBrain)

**Rationale**:
- Easiest integration (30 minutes)
- Excellent performance (+7 dB over ConvTasNet)
- Best documentation
- Apache 2.0 license
- Production-ready

**Quick Start**:
```python
from speechbrain.inference.separation import SepformerSeparation

model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir='pretrained_models/sepformer-wsj02mix'
)
est_sources = model.separate_file('mixture.wav')
```

---

### Priority 2: DPRNN (via Asteroid)

**Rationale**:
- Good performance (+3.5 dB over ConvTasNet)
- Most efficient RNN model (2.7M params)
- Alternative toolkit (diversifies dependencies)
- MIT license
- 1 hour integration

**Quick Start**:
```python
from asteroid.models import DPRNNTasNet

model = DPRNNTasNet.from_pretrained(
    'JorisCos/DPRNNTasNet-ks16_WHAM_sepclean'
)
separated = model.separate(mixture_waveform)
```

---

### Priority 3: SPMamba

**Rationale**:
- Cutting-edge 2024 architecture
- Excellent efficiency (6.1M params, 22.5 dB)
- Apache 2.0 license
- Research showcase value
- 2-3 hour integration

**Challenges**:
- Custom conda environment
- Standalone codebase
- Less community support

---

## Required Code Changes for Multi-Model Support

### 1. Configuration System

**Current**: Hardcoded ModelConfig for ConvTasNet parameters

**Required**:
```python
@dataclass
class ModelConfig:
    model_type: str = "convtasnet"  # selector
    # Model-specific parameters as nested configs
```

---

### 2. Model Registry

**Create**: `models/registry.py`
```python
MODEL_REGISTRY = {
    "convtasnet": ConvTasNet,
    "sepformer": SepFormer,
    "dprnn": DPRNN,
    "spmamba": SPMamba,
}
```

---

### 3. Model Factory

**Create**: `models/model_factory.py`
```python
def create_model(config: ModelConfig) -> SeparationModel:
    model_class = MODEL_REGISTRY[config.model_type]
    return model_class.from_config(config)
```

---

### 4. Base Model Interface

**Create**: `models/base.py`
```python
class SeparationModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def output_sources(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> 'SeparationModel':
        pass
```

---

### 5. Trainer Changes

**Current**: Hardcoded input preprocessing, loss routing

**Required**:
- Abstract input preprocessing
- Flexible loss function selection
- Model interface validation

**Files to Modify**:
- `training/trainer.py` (lines 55-67, 165-169)
- `train.py` (lines 15-40)
- `evaluate.py` (lines 22-65)

---

### 6. Evaluation Changes

**Current**: Hardcoded ConvTasNet instantiation

**Required**:
```python
model_type = checkpoint.get("model_type", "convtasnet")
model_class = MODEL_REGISTRY[model_type]
model = model_class.from_config(checkpoint["config"]["model"])
```

---

## Benchmarking Framework

### Metrics to Track

1. **Performance**:
   - SI-SNRi (primary)
   - SDRi
   - PESQ (optional)
   - STOI (optional)

2. **Efficiency**:
   - Inference time
   - RTF (Real-Time Factor)
   - Peak memory (GPU/CPU)
   - Model parameters

3. **Quality**:
   - Subjective listening tests
   - Perceptual metrics

### Test Datasets

- WSJ0-2Mix (standard benchmark)
- LibriMix (modern alternative)
- PolSESS (project-specific)
- Custom evaluation sets

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Create base model interface
2. Implement model registry
3. Create model factory
4. Update configuration system

### Phase 2: Quick Win (Week 2)
5. Add SepFormer via SpeechBrain
6. Test on PolSESS data
7. Benchmark vs ConvTasNet
8. Document integration

### Phase 3: Diversification (Week 3)
9. Add DPRNN via Asteroid
10. Performance comparison
11. Update evaluation scripts
12. User selection interface

### Phase 4: Innovation (Week 4)
13. Integrate SPMamba
14. Full benchmark suite
15. Documentation update
16. Examples and tutorials

### Optional Phase 5: Maximum Performance
17. Evaluate MossFormer2 if needed
18. Advanced configurations
19. Ensemble methods

---

## Files Requiring Modification

### Core Files to Create:
- `models/base.py` - Abstract base class
- `models/registry.py` - Model registration
- `models/model_factory.py` - Factory function
- `models/sepformer.py` - SepFormer wrapper
- `models/dprnn.py` - DPRNN wrapper
- `models/spmamba.py` - SPMamba wrapper (optional)

### Core Files to Modify:
- `models/__init__.py` - Export registry and factory
- `config.py` - Add model_type selector
- `training/trainer.py` - Abstract preprocessing/loss
- `train.py` - Use model factory
- `evaluate.py` - Use model factory
- `training/trainer_factory.py` - Update signature

### Documentation Files:
- `README.md` - Update with multi-model support
- `docs/MODELS.md` - Create model documentation
- `docs/BENCHMARKS.md` - Performance comparisons

---

## Risk Assessment

### Low Risk:
- SepFormer integration (SpeechBrain is stable)
- DPRNN integration (Asteroid is mature)
- Configuration changes (well-scoped)

### Medium Risk:
- SPMamba integration (custom environment)
- Input/output format compatibility
- Backward compatibility with existing checkpoints

### High Risk:
- MossFormer2 integration (ecosystem dependency)
- Performance degradation on PolSESS data
- Breaking changes to existing workflows

---

## Success Criteria

### Technical:
- All models achieve expected performance on WSJ0-2Mix
- All models work with PolSESS dataset
- Unified API across all models
- Backward compatibility maintained

### Usability:
- Model selection via config file
- Clear documentation for each model
- Example scripts for each model
- Easy switching between models

### Performance:
- SepFormer: >22 dB SI-SNRi
- DPRNN: >18 dB SI-SNRi
- SPMamba: >22 dB SI-SNRi (if implemented)

---

## References

### Papers:
- ConvTasNet: "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation" (2019)
- SepFormer: "Attention is All You Need in Speech Separation" (2020)
- DPRNN: "Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation" (2019)
- SPMamba: "SPMamba: State-space model is all you need in speech separation" (2024)
- MossFormer2: "MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation" (2024)

### Code Repositories:
- SpeechBrain: https://github.com/speechbrain/speechbrain
- Asteroid: https://github.com/asteroid-team/asteroid
- SPMamba: https://github.com/JusperLee/SPMamba
- MossFormer2: https://github.com/alibabasglab/MossFormer2

### Resources:
- Speech Separation Tutorial: https://cslikai.cn/Speech-Separation-Paper-Tutorial/
- SpeechBrain Models: https://huggingface.co/speechbrain
- Asteroid Models: https://asteroid.readthedocs.io/

---

## Conclusion

Adding multi-model support to the polsess_separation project is achievable with moderate effort:

**Recommended Path**:
1. Start with SepFormer (30 min integration, +7 dB improvement)
2. Add DPRNN for RNN diversity (1 hour, established model)
3. Optionally add SPMamba for cutting-edge performance (2-3 hours)

**Total Estimated Effort**: 1-2 weeks for complete multi-model framework

**Key Benefits**:
- Architectural diversity (Transformer, RNN, State-Space)
- Performance range: 18.8 to 22.5 dB SI-SNRi
- Two different toolkits (reduces vendor lock-in)
- All Apache 2.0 or MIT licensed
- Production-ready implementations

**Next Steps**:
1. Review and approve this research
2. Design detailed implementation plan
3. Set up development branch
4. Begin with foundation (base classes, registry)
5. Integrate SepFormer as proof of concept
6. Iterate based on results

---

**Report Status**: Complete
**Research Confidence**: High (verified with official sources)
**Recommendation Confidence**: High (based on community adoption, licensing, ease of integration)
