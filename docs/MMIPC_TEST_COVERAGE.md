# MM-IPC Test Coverage Documentation

This document describes the comprehensive test coverage for the PolSESS dataset's MM-IPC (Multi-Modal Indoor/outdoor and Physical Complexity) variant system implemented in `tests/test_mmipc.py`.

## Overview

The tests verify that the `_lazy_load()`, `_apply_mmipc()`, and `_compute_clean()` methods in `PolSESSDataset` correctly handle all combinations of:
- **Tasks**: ES (single speaker extraction), EB (both speakers), SB (speaker separation)
- **Indoor variants** (with reverb): SER, SR, ER, R, C
- **Outdoor variants** (no reverb): SE, S, E, C

## Variant Semantics

Variant letters indicate which components to **KEEP** in the mix:
- **S**: Keep Scene (remove scene from mix)
- **E**: Keep Event (remove event from mix)
- **R**: Keep Reverb (remove reverb from mix)
- **C**: Clean (remove all: scene, event, reverb)

## Verified Logic for `_lazy_load()`

The tests verify that `_lazy_load()` loads exactly the audio files that need to be **removed** from the mix.

### ES Task (Extract Speaker 1)

**Indoor variants (has_reverb=True):**
- **SER**: Loads `{mix, speaker1, speaker2, sp2_reverb}` - Only speaker2 and its reverb removed
- **SR**: Loads `{mix, speaker1, speaker2, sp2_reverb, event, ev_reverb}` - Remove speaker2, event+reverb
- **ER**: Loads `{mix, speaker1, speaker2, sp2_reverb, scene}` - Remove speaker2, scene
- **R**: Loads `{mix, speaker1, speaker2, sp2_reverb, scene, event, ev_reverb}` - Remove speaker2, all background
- **C**: Loads `{mix, speaker1, speaker2, sp1_reverb, sp2_reverb, scene, event, ev_reverb}` - Remove everything

**Outdoor variants (has_reverb=False):**
- **SE**: Loads `{mix, speaker1, speaker2}` - Only speaker2 removed
- **S**: Loads `{mix, speaker1, speaker2, event}` - Remove speaker2, event
- **E**: Loads `{mix, speaker1, speaker2, scene}` - Remove speaker2, scene
- **C**: Loads `{mix, speaker1, speaker2, scene, event}` - Remove speaker2, all background

### EB Task (Extract Both Speakers)

**Indoor variants:**
- **SER**: Loads `{mix, speaker1, speaker2}` - No removal (keep all)
- **SR**: Loads `{mix, speaker1, speaker2, event, ev_reverb}` - Remove event+reverb
- **ER**: Loads `{mix, speaker1, speaker2, scene}` - Remove scene
- **R**: Loads `{mix, speaker1, speaker2, scene, event, ev_reverb}` - Remove all background
- **C**: Loads `{mix, speaker1, speaker2, sp1_reverb, sp2_reverb, scene, event, ev_reverb}` - Remove all

**Outdoor variants:**
- **SE**: Loads `{mix, speaker1, speaker2}` - No removal
- **S**: Loads `{mix, speaker1, speaker2, event}` - Remove event
- **E**: Loads `{mix, speaker1, speaker2, scene}` - Remove scene
- **C**: Loads `{mix, speaker1, speaker2, scene, event}` - Remove all background

### SB Task (Separate Both Speakers)

SB task has identical loading behavior to EB task.

## Verified Logic for `_apply_mmipc()`

Tests verify that `_apply_mmipc()` correctly subtracts loaded components from the mix:

**Examples tested:**
1. **ES+SER (indoor)**: mix - speaker2 - sp2_reverb
2. **ES+C (indoor)**: mix - sp1_reverb - sp2_reverb - speaker2 - scene - event - ev_reverb
3. **EB+C (indoor)**: mix - sp1_reverb - sp2_reverb - scene - event - ev_reverb (keeps both speakers)
4. **ES+SE (outdoor)**: mix - speaker2

The tests use mock audio with distinguishable constant values to verify exact arithmetic.

## Verified Logic for `_compute_clean()`

Tests verify correct target computation:

1. **ES task**: Returns `speaker1` only (1D tensor)
2. **EB task**: Returns `speaker1 + speaker2` (1D tensor)
3. **SB task**: Returns `stack([speaker1, speaker2])` (2D tensor with shape [2, T])

## Test Structure

### Test Classes

1. **TestLazyLoadFieldsES**: 10 tests covering all ES task variants
2. **TestLazyLoadFieldsEB**: 10 tests covering all EB task variants
3. **TestLazyLoadFieldsSB**: 10 tests covering all SB task variants
4. **TestApplyMMIPC**: 4 tests verifying component subtraction arithmetic
5. **TestComputeClean**: 3 tests verifying target computation for each task
6. **TestVariantSelection**: 3 tests verifying reverb filtering logic
7. **TestEndToEnd**: 27 parameterized tests covering all task×variant combinations

**Total: 67 test cases covering 61 unique scenarios**

### Mock Data

The tests use a pytest fixture `mock_polsess_data` that creates:
- 2 audio samples per subset (train/val)
- 1 indoor sample (with reverb files)
- 1 outdoor sample (without reverb files)
- Sample rate: 8000 Hz
- Duration: 4 seconds
- Each audio component has a unique constant value for arithmetic verification

## Edge Cases Covered

1. **Variant C handling**: Tests separately verify indoor and outdoor "C" variant behavior
2. **Dataset filtering**: Tests verify that indoor/outdoor variants correctly filter samples
3. **Mixed variants**: Tests verify that "C" variant includes both indoor and outdoor samples
4. **Task-specific loading**: Tests verify ES task always loads sp2_reverb, EB/SB don't
5. **Reverb-only loading**: Tests verify C variant loads speaker reverb only for indoor samples

## Running the Tests

```bash
# Run all MM-IPC tests
pytest tests/test_mmipc.py -v

# Run specific test class
pytest tests/test_mmipc.py::TestLazyLoadFieldsES -v

# Run with coverage
pytest tests/test_mmipc.py --cov=datasets.polsess_dataset --cov-report=html
```

## Verification Summary

✅ **All 3 tasks** (ES, EB, SB) tested with all compatible variants
✅ **All 9 variants** tested (5 indoor + 4 outdoor)
✅ **Field loading logic** verified for all 27 task×variant combinations
✅ **Component subtraction** verified with arithmetic checks
✅ **Target computation** verified for all task types
✅ **Variant selection and filtering** verified
✅ **End-to-end integration** verified with shape and value checks

The implementation is **fully tested and verified** to correctly implement the MM-IPC augmentation strategy.
