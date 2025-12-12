# Future Enhancements Analysis

Potential improvements that are NOT yet implemented.

---

## Summary of Already Implemented Features

The following features have been FULLY IMPLEMENTED and are no longer "future" enhancements:

✅ **YAML Configuration Files** - Fully functional with `experiments/` folder containing multiple configs
✅ **Logging System** - `utils/logger.py` with colored console output and file logging
✅ **Weights & Biases Integration** - `utils/wandb_logger.py` with comprehensive tracking
✅ **EB Task (Enhance Both Speakers)** - Supported in config, experiments/eb_task.yaml exists
✅ **SB Task (Speech Separation)** - Fully implemented with PIT loss, multiple sweep configs

See the respective modules and configuration files for usage.

---

## Remaining Future Enhancements

### 1. TensorBoard Integration

**Status:** Not implemented (W&B is implemented, but TensorBoard specifically is not)

**What it does:**
- Visualize training curves (loss, metrics over time) locally
- Compare multiple runs side-by-side
- View model architecture
- Display images/audio samples
- Works offline without cloud dependency

**Pros:**
- ✅ Free and open-source
- ✅ Works offline (unlike W&B which requires internet)
- ✅ No account needed
- ✅ PyTorch native support (`torch.utils.tensorboard`)
- ✅ Lighter weight than W&B
- ✅ Good for quick local experiments

**Cons:**
- ⚠️ Less feature-rich than W&B
- ⚠️ Manual experiment organization
- ⚠️ No built-in collaboration features

**Implementation Effort:** 1-2 hours

**Implementation:**
```python
# trainer.py
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, ..., log_dir='runs'):
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):
        # ... training code ...

        # Log metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('SI-SDR/train', train_sisdr, epoch)

    def validate(self, epoch):
        # ... validation code ...

        self.writer.add_scalar('SI-SDR/val', val_sisdr, epoch)
```

**Usage:**
```bash
# Install (if needed)
pip install tensorboard

# Train model (logs automatically)
python train.py --config experiments/baseline.yaml

# View in browser
tensorboard --logdir=runs
```

**When to use:**
- Quick local experiments
- Prefer working offline
- Don't need cloud collaboration
- Want lightweight solution

**Note:** W&B is already implemented and provides more features. TensorBoard would be a complementary option for offline work.

---

### 2. Additional Model Architectures

**Status:** Only ConvTasNet is implemented

**Potential additions:**
- **SPMamba** - State-space model (see docs/SPMAMBA_INTEGRATION_PLAN.md for details)
- **MossFormer2** - Recent SOTA architecture
- **SepFormer** - Transformer-based separation
- **DPTNet** - Dual-path transformer

**Benefit:** Compare different architectures, potentially better performance

**Effort:** 4-8 hours per architecture (depends on complexity and available implementations)

---

### 3. Real-Time Inference

**Status:** Not implemented

**What it would include:**
- Streaming audio processing
- Low-latency separation
- Microphone input support
- Real-time visualization

**Use cases:**
- Live demonstrations
- Interactive applications
- Real-world deployment testing

**Effort:** 6-10 hours

**Challenges:**
- ConvTasNet requires sufficient context (chunk-based processing needed)
- Latency optimization
- Buffer management

---

### 4. Additional Evaluation Metrics

**Status:** SI-SDR, PESQ, STOI are implemented

**Potential additions:**
- **SDR, SAR, SIR** - Source separation metrics
- **WER** - Word Error Rate (with ASR model)
- **MOS prediction** - Perceptual quality prediction
- **DNSMOS** - Deep noise suppression MOS

**Benefit:** More comprehensive quality assessment

**Effort:** 2-3 hours

---

### 5. Data Augmentation

**Status:** MM-IPC is implemented, but no additional augmentation

**Potential additions:**
- **SpecAugment** - Frequency/time masking
- **Pitch shifting** - Voice variation
- **Time stretching** - Speed variation
- **Room impulse response augmentation** - More reverb variety
- **Noise augmentation** - Additional noise types

**Benefit:** Potentially better generalization

**Effort:** 3-5 hours

**Note:** May or may not improve performance - needs experimental validation

---

### 6. Model Compression

**Status:** No compression implemented

**Techniques:**
- **Pruning** - Remove less important weights
- **Quantization** - Reduce precision (int8, float16)
- **Knowledge distillation** - Train smaller model from larger
- **ONNX export** - For deployment

**Benefit:** Faster inference, smaller model size, deployment-ready

**Effort:** 4-6 hours

---

### 7. Cross-Dataset Evaluation

**Status:** Libri2Mix is implemented, but no systematic cross-dataset testing

**Potential additions:**
- Test on DNS Challenge datasets
- Test on WHAM! dataset
- Test on other Polish speech datasets
- Domain adaptation experiments

**Benefit:** Assess generalization capabilities

**Effort:** 2-3 hours per dataset

---

## Recommended Priorities

### High Priority (High Value, Low Effort)
1. **TensorBoard** - 1-2 hours, huge benefit for offline experimentation

### Medium Priority (Moderate Effort, Good Value)
2. **Additional Evaluation Metrics** - 2-3 hours, better quality assessment
3. **Cross-Dataset Evaluation** - 2-3 hours per dataset, assess generalization

### Low Priority (High Effort or Uncertain Value)
4. **Additional Model Architectures** - 4-8 hours each, experimental
5. **Data Augmentation** - 3-5 hours, uncertain benefit
6. **Real-Time Inference** - 6-10 hours, depends on use case
7. **Model Compression** - 4-6 hours, depends on deployment needs

---

## Quick Wins

If you have 2-3 hours and want maximum impact:

1. **Add TensorBoard** (1-2 hours)
   - Visualize training offline
   - Compare experiments easily
   - No cloud dependency

2. **Add SDR/SAR/SIR metrics** (1 hour)
   - More comprehensive evaluation
   - Standard in separation literature

**Total: 2-3 hours for significant improvement**

---

## Notes

- Most major features have already been implemented
- Focus should now be on:
  - Using the system for actual experiments
  - Publishing results
  - Writing papers/thesis
- Additional enhancements are optional and depend on specific research goals
