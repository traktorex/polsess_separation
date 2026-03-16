# ASR Evaluation Module

Evaluates speech separation models as a preprocessing step for ASR. Measures whether separating a 2-speaker mixture improves Whisper transcription accuracy (WER/CER).

## Evaluation Modes

| Mode | Description | Model needed? | Datasets |
|------|-------------|:---:|---------|
| `separation` | Separate → transcribe → WER/CER | Yes | LibriSpeech, REAL-M |
| `mixture` | Transcribe unseparated mix (no-separation baseline) | No | LibriSpeech, REAL-M |
| `baseline` | Transcribe clean source audio (best achievable WER) | No | LibriSpeech only |

## Datasets

**LibriSpeechMixASR** — Synthetic 2-speaker mixes from LibriSpeech at 16kHz.
Clean source audio available. Location: `~/datasets/LibriSpeechMixASR/` (or `LIBRIMIX_ASR_ROOT` env var).

**REAL-M** — 1,436 real-world 2-speaker mixtures at 8kHz with transcriptions.
No clean sources — ASR-based evaluation only. Location: `~/datasets/REAL-M-v0.1.0/` (or `REALM_DATA_ROOT` env var).

## Usage

```bash
# Run from polsess_separation/ directory

# Evaluate separation model on REAL-M
python asr/evaluate_asr.py \
    --checkpoint checkpoints/spmamba/SB/.../best.pt \
    --dataset realm --mode separation --whisper-model large

# Mixture baseline on REAL-M (shows problem: high WER without separation)
python asr/evaluate_asr.py --dataset realm --mode mixture

# Clean source baseline on LibriSpeech
python asr/evaluate_asr.py --dataset librispeech --split dev --mode baseline

# Separation on LibriSpeech (50 samples, specific Whisper model)
python asr/evaluate_asr.py \
    --checkpoint checkpoints/dprnn/SB/.../best.pt \
    --dataset librispeech --split dev --mode separation \
    --num-samples 50 --whisper-model base.en

# Run Whisper on GPU (default: cpu to save GPU memory for separation model)
python asr/evaluate_asr.py --dataset realm --mode separation \
    --checkpoint path/to/model.pt --whisper-device cuda

# Override dataset location
python asr/evaluate_asr.py --dataset realm --dataset-dir /path/to/REAL-M --mode mixture
```

## Module Structure

```
asr/
├── __init__.py          # Public API
├── dataset.py           # LibriSpeechMixDataset + RealMDataset
├── evaluate_asr.py      # Unified evaluation script (3 modes)
├── metrics.py           # WER/CER via jiwer (micro-averaged)
├── transcribe.py        # WhisperTranscriber wrapper
├── archive/             # One-time data prep scripts (already run)
│   ├── prepare_librispeech_mix.py
│   ├── transcribe_sources.py
│   └── create_long_mixes.py
│   └── separate_audio.py    # Standalone batch separation utility
└── *_asr_results.json   # Previous evaluation results
```

## Results

### LibriSpeechMixASR (Whisper large, 500 samples)

| Mode | WER (%) | CER (%) |
|------|---------|---------|
| Clean baseline | 0.13 | 0.06 |
| SPMamba separation | 18.80 | 12.58 |

### REAL-M (SPMamba, 1436 samples)

| Whisper model | WER (%) | CER (%) |
|---------------|---------|---------|
| tiny.en | 86.34 | 66.38 |
| large | 63.31 | 46.63 |

For comparison, the REAL-M paper reports SepFormer-WHAMR! at 60.7% WER (Wav2Vec 2.0 ASR).

### Legacy (LibriSpeechMixASR dev, 50 samples)

| Model | WER (%) | CER (%) |
|-------|---------|---------|
| ConvTasNet | 82.89 | 62.94 |
| DPRNN | 60.65 | 42.02 |
