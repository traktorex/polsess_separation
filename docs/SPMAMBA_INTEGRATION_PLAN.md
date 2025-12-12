# SPMamba Integration Plan

## Goal
Add SPMamba model support to the PolSESS separation project while maintaining backward compatibility with existing ConvTasNet infrastructure.

---

## Background

**SPMamba** (State-space model for speech separation):
- Based on TF-GridNet architecture
- Replaces BLSTM with Bidirectional Mamba modules
- Achieves 2.42 dB improvement over baselines
- 6.14M parameters vs 14.43M for TF-GridNet
- 78.69 G/s vs 445.56 G/s (much more efficient!)
- Works in time-frequency domain (STFT)

**Key Difference from ConvTasNet**:
- ConvTasNet: Time-domain separation (encoder-decoder with 1D convolutions)
- SPMamba: Frequency-domain separation (STFT with GridNet blocks)

---

## Architecture Changes Needed

### 1. **Model Registry System** (NEW)

**File**: `models/model_factory.py`

**Purpose**: Central registration and creation of all model architectures

**Functions**:
```python
@register_model("convtasnet")  # Decorator
class ConvTasNet(nn.Module): ...

@register_model("spmamba")
class SPMamba(nn.Module): ...

get_model(name: str, **kwargs) -> nn.Module
list_models() -> List[str]
```

**Benefits**:
- Easy to add new models
- Type-safe model selection
- Automatic model discovery

---

### 2. **Config System Updates**

**File**: `config.py`

**Add to ModelConfig**:
```python
@dataclass
class ModelConfig:
    # Existing ConvTasNet params
    N: int = 256
    B: int = 256
    H: int = 512
    # ... existing params ...

    # NEW: Model selection
    architecture: str = "convtasnet"  # or "spmamba"

    # NEW: SPMamba-specific params
    spmamba_num_blocks: int = 6
    spmamba_dim: int = 64
    spmamba_chunk_size: int = 250
    # ... other SPMamba params ...
```

**Alternative Approach** (Better):
```python
@dataclass
class BaseModelConfig:
    architecture: str = "convtasnet"
    C: int = 1  # Output sources (shared)

@dataclass
class ConvTasNetConfig(BaseModelConfig):
    N: int = 256
    B: int = 256
    H: int = 512
    # ... ConvTasNet specific ...

@dataclass
class SPMambaConfig(BaseModelConfig):
    num_blocks: int = 6
    dim: int = 64
    chunk_size: int = 250
    # ... SPMamba specific ...

@dataclass
class Config:
    model: Union[ConvTasNetConfig, SPMambaConfig]
    # ... rest ...
```

---

### 3. **Model Implementation**

**File**: `models/spmamba.py` (NEW)

**Required Components**:

1. **SPMamba** main model class
   - STFT encoder/decoder
   - GridNet blocks with Mamba modules
   - Multi-head attention
   - Compatible with same training loop as ConvTasNet

2. **MambaBlock**
   - Bidirectional Mamba implementation
   - State-space modeling
   - Linear complexity

3. **GridNetBlock**
   - Temporal (intra) processing
   - Frequency (inter) processing
   - Attention mechanism

4. **STFTEncoder/Decoder**
   - Time-frequency transformation
   - Inverse STFT for reconstruction

**Implementation Options**:

**Option A**: Full implementation from scratch
- Pros: Full control, no dependencies
- Cons: Complex, time-consuming (~1000+ lines)

**Option B**: Adapt from SPMamba repository
- Pros: Proven implementation
- Cons: Need to extract and adapt dependencies

**Option C**: Simplified/hybrid version
- Pros: Easier to integrate, faster
- Cons: May not match paper performance

**Recommended**: Option B (adapt from repository)

---

### 4. **Training Script Updates**

**File**: `train.py`

**Current**:
```python
def create_model(config, summary_info):
    model = ConvTasNet(N=config.model.N, ...)  # Hardcoded!
    return model
```

**New**:
```python
def create_model(config, summary_info):
    from models.model_factory import get_model

    if config.model.architecture == "convtasnet":
        model = get_model("convtasnet",
            N=config.model.N,
            B=config.model.B,
            H=config.model.H,
            # ... other params ...
        )
    elif config.model.architecture == "spmamba":
        model = get_model("spmamba",
            num_blocks=config.model.spmamba_num_blocks,
            dim=config.model.spmamba_dim,
            # ... other params ...
        )
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")

    # Summary info...
    return model
```

**Even Better** (with model-specific config parsing):
```python
def create_model(config, summary_info):
    from models.model_factory import get_model

    # Model configs provide their own build_kwargs()
    model_kwargs = config.model.get_model_kwargs()
    model = get_model(config.model.architecture, **model_kwargs)

    # Populate summary
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    summary_info["model_params_millions"] = num_params
    summary_info["model_arch"] = config.model.architecture

    return model
```

---

### 5. **Models Package Update**

**File**: `models/__init__.py`

**Current**:
```python
from .conv_tasnet import ConvTasNet
```

**New**:
```python
from .conv_tasnet import ConvTasNet
from .spmamba import SPMamba  # NEW
from .model_factory import (  # NEW
    get_model,
    list_models,
    register_model,
)

__all__ = ["ConvTasNet", "SPMamba", "get_model", "list_models", "register_model"]
```

---

### 6. **Experiment Configs**

**New Files**:

**`experiments/spmamba_es.yaml`**:
```yaml
data:
  task: ES
  batch_size: 4

model:
  architecture: spmamba
  C: 1
  spmamba_num_blocks: 6
  spmamba_dim: 64
  spmamba_chunk_size: 250

training:
  num_epochs: 50
  lr: 0.001
  # ... training params ...
```

**`experiments/spmamba_sb.yaml`**:
```yaml
data:
  task: SB
  batch_size: 4

model:
  architecture: spmamba
  C: 2  # Two speakers
  # ... SPMamba params ...
```

---

## Implementation Steps

### Phase 1: Infrastructure (30 min)
1. ✅ Create `models/model_factory.py` with registry system
2. ✅ Update `models/__init__.py` to export factory functions
3. ✅ Add `architecture` field to `config.py` ModelConfig
4. ✅ Update `train.py` create_model() to use factory

### Phase 2: SPMamba Core Implementation (2-3 hours)
5. ⏳ Create `models/spmamba.py` with basic structure:
   - STFTEncoder/Decoder classes
   - MambaBlock (can use mamba-ssm library or implement)
   - GridNetBlock
   - SPMamba main class
6. ⏳ Register SPMamba with decorator
7. ⏳ Add SPMamba-specific config fields

### Phase 3: Testing & Integration (1 hour)
8. ⏳ Create simple test to verify SPMamba instantiation
9. ⏳ Test with dummy data (forward pass)
10. ⏳ Create experiment config `spmamba_es.yaml`
11. ⏳ Run short training test (1-2 epochs)

### Phase 4: Optimization (optional)
12. ⏳ Tune SPMamba hyperparameters
13. ⏳ Create sweep configs for SPMamba
14. ⏳ Compare performance vs ConvTasNet

---

## Dependencies

**New Required Packages**:
```bash
# For Mamba state-space models
pip install mamba-ssm  # Official Mamba implementation
# OR
pip install causal-conv1d>=1.1.0  # If building from source

# For STFT operations (should already have)
# torch, torchaudio
```

**Add to `requirements.txt`**:
```
mamba-ssm>=1.1.0  # State-space models
```

---

## Backward Compatibility

**Guaranteed**:
- All existing configs work (default architecture="convtasnet")
- All existing training scripts unchanged
- All existing checkpoints loadable
- No breaking changes to API

**Migration Path**:
- Old configs automatically use ConvTasNet
- New configs specify `model.architecture = "spmamba"`
- Can mix models in same experiment directory

---

## File Structure After Implementation

```
polsess_separation/
├── models/
│   ├── __init__.py              # Updated: export factory
│   ├── conv_tasnet.py           # Existing: with @register_model
│   ├── spmamba.py               # NEW: SPMamba implementation
│   └── model_factory.py         # NEW: registry system
├── experiments/
│   ├── baseline.yaml            # Existing: ConvTasNet
│   ├── sb_task.yaml             # Existing: ConvTasNet SB
│   ├── spmamba_es.yaml          # NEW: SPMamba ES task
│   └── spmamba_sb.yaml          # NEW: SPMamba SB task
├── train.py                     # Updated: use model factory
├── config.py                    # Updated: add architecture field
└── ...
```

---

## Risks & Mitigations

### Risk 1: Mamba dependency issues
**Mitigation**: Provide fallback to simplified Mamba or use existing attention

### Risk 2: Different input/output shapes
**Mitigation**: Create adapter layer to match ConvTasNet interface

### Risk 3: Training instability
**Mitigation**: Start with proven hyperparameters from paper

### Risk 4: Performance worse than ConvTasNet
**Mitigation**: This is research - compare both, document findings

---

## Success Criteria

✅ **Minimum Viable**:
- SPMamba model instantiates without error
- Forward pass works with dummy data
- Can run 1 epoch of training

✅ **Full Success**:
- SPMamba trains to completion
- Achieves reasonable SI-SDR (> baseline)
- Model selection via config works seamlessly
- Documentation complete

✅ **Stretch Goals**:
- SPMamba outperforms ConvTasNet
- Hyperparameter sweep for SPMamba
- Published comparison results

---

## Next Steps

**Immediate**:
1. Review this plan
2. Decide on SPMamba implementation approach (A/B/C)
3. Install mamba-ssm dependency
4. Start Phase 1 (infrastructure)

**Questions to Answer**:
- Use full SPMamba or simplified version?
- Copy from repo or implement from paper?
- Need GPU for mamba-ssm or CPU-compatible?
