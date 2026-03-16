"""ASR evaluation datasets: LibriSpeech synthetic mixes and REAL-M real-world mixtures.

Both datasets return the same core contract for the evaluation script:
    - mix: Audio tensor (1, T)
    - transcription1: Speaker 1 ground truth text
    - transcription2: Speaker 2 ground truth text
    - sample_id: Unique identifier
    - sample_rate: Audio sample rate (Hz)

LibriSpeechMixDataset additionally provides clean source audio (s1, s2).
RealMDataset does NOT have clean sources — only mixtures and transcriptions.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torchaudio

logger = logging.getLogger("polsess")

# Default dataset paths (override via environment variables)
DEFAULT_LIBRIMIX_ASR_ROOT = "/home/user/datasets/LibriSpeechMixASR"
DEFAULT_REALM_ROOT = "/home/user/datasets/REAL-M-v0.1.0"

# Early REAL-M sessions have a different CSV format (no index column, .mp3 filenames)
REALM_EARLY_SESSIONS = {"early_collection1", "early_collection2", "early_collection3"}


class LibriSpeechMixDataset:
    """Synthetic 2-speaker LibriSpeech mixes with ground truth transcriptions.

    Created by prepare_librispeech_mix.py (see asr/archive/).
    Each sample is a 4-second chunk at 16kHz with clean source audio available.

    Args:
        dataset_root: Root directory containing dev/, test/, long/, 10sec/ subsets.
            Defaults to LIBRIMIX_ASR_ROOT env var or ~/datasets/LibriSpeechMixASR.
        split: Dataset subset — 'dev', 'test', 'long', or '10sec'.
        max_samples: Limit number of samples (None = all).
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: str = "dev",
        max_samples: Optional[int] = None,
    ):
        root = dataset_root or os.getenv("LIBRIMIX_ASR_ROOT", DEFAULT_LIBRIMIX_ASR_ROOT)
        self.dataset_root = Path(root)
        self.split = split

        valid_splits = ("dev", "test", "long", "10sec")
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

        self.data_dir = self.dataset_root / split

        # Load metadata with Whisper chunk transcriptions
        metadata_path = self.data_dir / "metadata_with_chunks.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Run asr/archive/transcribe_sources.py first to generate ground truth."
            )

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        if max_samples is not None:
            self.metadata = self.metadata[:max_samples]

        self.mix_dir = self.data_dir / "mix"
        self.s1_dir = self.data_dir / "s1"
        self.s2_dir = self.data_dir / "s2"

        logger.info(f"Loaded LibriSpeechMix {split}: {len(self.metadata)} samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Returns dict with: mix (1,T), s1 (1,T), s2 (1,T),
        transcription1, transcription2, sample_id, sample_rate.
        """
        item = self.metadata[idx]
        sample_id = item["id"]

        mix, sr = torchaudio.load(self.mix_dir / f"{sample_id}.wav")
        s1, _ = torchaudio.load(self.s1_dir / f"{sample_id}.wav")
        s2, _ = torchaudio.load(self.s2_dir / f"{sample_id}.wav")

        return {
            "mix": mix,
            "s1": s1,
            "s2": s2,
            "transcription1": item["transcription1_chunk"],
            "transcription2": item["transcription2_chunk"],
            "sample_id": sample_id,
            "speaker1_id": item["speaker1"],
            "speaker2_id": item["speaker2"],
            "sample_rate": sr,
        }


class RealMDataset:
    """REAL-M dataset: real-world 2-speaker mixtures with transcriptions.

    From Subakan et al. (2021) — 1,436 crowd-sourced real-life mixtures.
    Unlike LibriSpeechMixDataset, there are NO clean source signals —
    evaluation is only possible via ASR-based metrics (WER/CER).

    Audio is pre-converted to 8kHz mono WAV, matching our separation models.

    Args:
        dataset_root: Path to REAL-M-v0.1.0/ directory.
            Defaults to REALM_DATA_ROOT env var or ~/datasets/REAL-M-v0.1.0.
        max_samples: Limit number of samples (None = all 1,436).

    CSV format handling:
        - Early collections (3 sessions): columns sentence1, sentence2, WorkerId, filename
          with .mp3 filenames that map to .wav in the converted directory
        - Newer collections (51 sessions): columns index, sentence1, sentence2, filename
    """

    SAMPLE_RATE = 8000

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        root = dataset_root or os.getenv("REALM_DATA_ROOT", DEFAULT_REALM_ROOT)
        self.dataset_root = Path(root)
        self.audio_dir = self.dataset_root / "audio_files_converted_8000Hz"
        self.transcription_dir = self.dataset_root / "transcriptions"

        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}\n"
                f"Set REALM_DATA_ROOT env var or pass dataset_root argument."
            )

        self.samples = self._load_transcriptions()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded REAL-M: {len(self.samples)} samples from {self._session_count} sessions")

    def _load_transcriptions(self) -> List[Dict]:
        """Load all transcription CSVs and build flat sample list."""
        samples = []
        self._session_count = 0

        for csv_path in sorted(self.transcription_dir.glob("*.csv")):
            session_id = csv_path.stem
            is_early = session_id in REALM_EARLY_SESSIONS
            session_samples = 0

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename_csv = row["filename"].strip()

                    # Early collections have .mp3 filenames, audio is .wav
                    if is_early:
                        audio_filename = Path(filename_csv).stem + ".wav"
                    else:
                        audio_filename = filename_csv

                    audio_path = self.audio_dir / session_id / audio_filename

                    if not audio_path.exists():
                        logger.warning(f"Audio file not found, skipping: {audio_path}")
                        continue

                    samples.append({
                        "audio_path": audio_path,
                        "transcription1": row["sentence1"].strip(),
                        "transcription2": row["sentence2"].strip(),
                        "session_id": session_id,
                        "sample_id": Path(audio_filename).stem,
                    })
                    session_samples += 1

            if session_samples > 0:
                self._session_count += 1

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Returns dict with: mix (1,T), transcription1, transcription2,
        sample_id, session_id, sample_rate. No clean sources available.
        """
        sample = self.samples[idx]

        mix, sr = torchaudio.load(sample["audio_path"])

        return {
            "mix": mix,
            "transcription1": sample["transcription1"],
            "transcription2": sample["transcription2"],
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "sample_rate": sr,
        }
