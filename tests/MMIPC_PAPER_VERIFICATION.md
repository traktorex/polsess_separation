# MM-IPC Implementation Verification Against Paper

This document verifies that the implementation in `datasets/polsess_dataset.py` correctly implements the MM-IPC (Mix Modification by Inverted Phase Cancellation) technique as described in the paper "A Solution for Developing Corpora for Polish Speech Enhancement in Complex Acoustic Environments" by Klec et al.

## Paper Definition Summary

From **Table 2** (page 10) and **Section 2.1** (pages 3-4):

### Core Concept
**MM-IPC allows removing components from the mix by adding their phase-inverted versions.**

Variant letters indicate which components to **KEEP**:
- **S** = keep Scene
- **E** = keep Event
- **R** = keep Reverb
- **C** = Clean (remove all background)

### Indoor Variants (has_reverb=True)

| Variant | Keep | Remove from mix |
|---------|------|----------------|
| **SER** | S+E+R | Nothing (full complexity) |
| **SR** | S+R | Event + ev_reverb |
| **ER** | E+R | Scene |
| **R** | R | Scene + Event + ev_reverb |
| **C** | None | Scene + Event + sp1_reverb + sp2_reverb + ev_reverb |

### Outdoor Variants (has_reverb=False)

| Variant | Keep | Remove from mix |
|---------|------|----------------|
| **SE** | S+E | Nothing (full complexity) |
| **S** | S | Event |
| **E** | E | Scene |
| **C** | None | Scene + Event |

### Task-Specific Rules

From **Section 3.2** and **Figure 3** (page 8):

1. **ES (Extract Single Speaker)**: Remove speaker2 + sp2_reverb (if has_reverb)
2. **EB (Extract Both Speakers)**: Keep both speakers, remove background only
3. **SB (Separate Both Speakers)**: Keep both speakers, return as [2, T] tensor

## Implementation Verification

### `_lazy_load()` Method

The implementation loads files that will be **removed** from the mix (via `_apply_mmipc`).

#### Logic Analysis:

```python
# Lines 145-147: Always load mix and both speakers
audio["mix"], _ = torchaudio.load(paths["mix"])
audio["speaker1"], _ = torchaudio.load(paths["speaker1"])
audio["speaker2"], _ = torchaudio.load(paths["speaker2"])
```
✅ **Correct**: All tasks need the mix and speakers loaded.

```python
# Lines 150-151: Load scene if "S" NOT in variant
if "S" not in variant:
    audio["scene"], _ = torchaudio.load(paths["scene"])
```
✅ **Correct**:
- SER, SE: "S" is present → don't load scene (keep it)
- SR, S: "S" is present → don't load scene (keep it)
- ER, E, R, C: "S" not present → load scene (remove it)

```python
# Lines 153-156: Load event if "E" NOT in variant
if "E" not in variant:
    audio["event"], _ = torchaudio.load(paths["event"])
    if has_reverb:
        audio["ev_reverb"], _ = torchaudio.load(paths["ev_reverb"])
```
✅ **Correct**:
- SER, SE: "E" is present → don't load event (keep it)
- ER, E: "E" is present → don't load event (keep it)
- SR, S, R, C: "E" not present → load event + ev_reverb if indoor (remove them)

```python
# Lines 159-166: Load speaker reverbs
if has_reverb and "R" not in variant:
    if variant == "C":
        audio["sp1_reverb"], _ = torchaudio.load(paths["sp1_reverb"])
        audio["sp2_reverb"], _ = torchaudio.load(paths["sp2_reverb"])
elif has_reverb and self.task == "ES":
    audio["sp2_reverb"], _ = torchaudio.load(paths["sp2_reverb"])
```

Let me trace through all cases:

**Indoor + C variant** (any task):
- `has_reverb=True and "R" not in variant` → True and True = **True**
- `variant == "C"` → **True**
- Loads: sp1_reverb, sp2_reverb ✅

**Indoor + SER/SR/ER/R + ES task**:
- `has_reverb=True and "R" not in variant` → True and False/False/False/False = **False**
- `elif has_reverb and self.task == "ES"` → True and True = **True**
- Loads: sp2_reverb ✅

**Indoor + SER/SR/ER/R + EB/SB task**:
- First if: **False**
- elif: True and False = **False**
- Loads: nothing ✅

**Outdoor + any variant + any task**:
- `has_reverb=False` → both conditions are False
- Loads: nothing ✅

✅ **Correct**: All reverb loading logic matches paper requirements.

### `_apply_mmipc()` Method

```python
def _apply_mmipc(self, audio, has_reverb):
    """Apply MM-IPC by removing components from mix."""
    mix = audio["mix"]
    if has_reverb and "sp1_reverb" in audio:
        mix = mix - audio["sp1_reverb"]
    if has_reverb and "sp2_reverb" in audio:
        mix = mix - audio["sp2_reverb"]

    if self.task == "ES":
        mix = mix - audio["speaker2"]

    if "scene" in audio:
        mix = mix - audio["scene"]
    if "event" in audio:
        mix = mix - audio["event"]
    if "ev_reverb" in audio:
        mix = mix - audio["ev_reverb"]

    return mix
```

✅ **Correct**: Subtracts all loaded components from mix. This implements the phase cancellation described in the paper (Section 2.1):
> "Users can mute specific components from the mix by adding their phase-inverted versions."

Since subtraction is equivalent to adding the phase-inverted signal: `mix - component = mix + (-component)`

### `_compute_clean()` Method

```python
def _compute_clean(self, audio):
    """Compute clean target based on task."""
    if self.task == "EB":
        return audio["speaker1"] + audio["speaker2"]
    elif self.task == "ES":
        return audio["speaker1"]
    elif self.task == "SB":
        return torch.stack([audio["speaker1"], audio["speaker2"]])
```

✅ **Correct**: Matches Figure 3 (page 8) exactly:
- ES: Returns single speaker1
- EB: Returns sum of both speakers
- SB: Returns stacked speakers [2, T]

## Test Coverage Verification

The test suite in `tests/test_mmipc.py` covers:

1. ✅ **All task × variant combinations** (27 combinations)
2. ✅ **Field loading verification** for each combination
3. ✅ **Component subtraction arithmetic** verification
4. ✅ **Clean target computation** verification
5. ✅ **Indoor/outdoor filtering** verification
6. ✅ **End-to-end pipeline** verification

## Paper Requirements Checklist

| Requirement | Paper Reference | Status |
|-------------|----------------|--------|
| Load only components to remove | Section 2.1 | ✅ Verified |
| Variant letters indicate what to KEEP | Table 2 | ✅ Verified |
| Phase cancellation via subtraction | Section 2.1 | ✅ Verified |
| ES removes speaker2 always | Figure 3, Table 2 | ✅ Verified |
| EB keeps both speakers | Figure 3, Table 2 | ✅ Verified |
| SB returns stacked speakers | Figure 3 | ✅ Verified |
| Indoor variants load reverb components | Table 2 | ✅ Verified |
| Outdoor variants never load reverb | Table 2 | ✅ Verified |
| C variant removes all background | Table 2 | ✅ Verified |
| Event reverb loaded with event (indoor) | Implementation detail | ✅ Verified |

## Critical Implementation Details

### 1. Two-Stage Process

The paper describes (Section 2.1, Equation 1):
- Original mix: `y(t) = Σ(si(t) + ri(t)) + b(t) + e(t) + re(t)`
- MM-IPC modifies this by removing specific components

The implementation achieves this via:
1. **Stage 1 (`_lazy_load`)**: Load only components to remove
2. **Stage 2 (`_apply_mmipc`)**: Subtract them from mix

✅ This matches the paper's description of "adding phase-inverted versions."

### 2. Speaker2 Handling for ES Task

From the paper (page 8, Figure 3):
> "The first model enhances single speech (ES) by separating it from complex noise"

The code correctly interprets "complex noise" to include speaker2 for ES task.

For indoor samples with ES task:
- Speaker2 is removed: ✅
- Speaker2's reverb is also removed: ✅ (lines 165-166)

This is **critical** because removing speaker2 but keeping its reverb would create an inconsistency.

### 3. C Variant Special Handling

Table 2 shows C variant removes ALL background components:
- Indoor C: Removes scene + event + ev_reverb + sp1_reverb + sp2_reverb
- Outdoor C: Removes scene + event

The code handles this correctly:
- Lines 161-163: For C variant, loads both speaker reverbs
- Combined with ES task logic: Also loads sp2_reverb for ES

✅ **Verified**: C variant indeed creates "clean" speech as described.

## Conclusion

**The implementation is FULLY CORRECT and matches the paper's specification exactly.**

All aspects of the MM-IPC technique as described in Klec et al.'s paper are properly implemented:

1. ✅ Variant semantics (letters indicate what to KEEP)
2. ✅ Phase cancellation via subtraction
3. ✅ Task-specific behavior (ES/EB/SB)
4. ✅ Indoor/outdoor distinction
5. ✅ Reverb handling
6. ✅ Component interdependencies (e.g., event + ev_reverb)

The test suite comprehensively verifies all 27 task×variant combinations plus edge cases, providing confidence that the implementation will work correctly in production use.

## Minor Observation

The paper mentions (page 7, Section 3.1):
> "Each corpus contains 4,000 training examples (approximately 4.4 hours)"

This suggests ~4 second samples, which matches our test fixture:
```python
duration = 4.0  # seconds
sr = 8000  # Hz
```

✅ Test parameters match the paper's experimental setup.
