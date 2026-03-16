"""Libri2Mix dataset loader for cross-dataset evaluation.

Supports Libri2Mix for 2-speaker separation (SB task) evaluation.
Two variants available:
  - Libri2Mix-Clean (mix_clean): s1 + s2, no noise
  - Libri2Mix-Noisy (mix_both): s1 + s2 + WHAM noise

Compatible with PolSESS-trained separation models.
"""

import logging
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, List, Dict

logger = logging.getLogger("polsess")


class Libri2MixDataset(Dataset):
    """Libri2Mix dataset for 2-speaker separation evaluation.

    Args:
        data_root: Path to Libri2Mix root (contains wav8k/, wav16k/)
        subset: Dataset split - "test", "dev", or "train-100"
        sample_rate: Target sample rate (8000 or 16000)
        mode: "min" or "max" (utterance length mode)
        mix_type: "mix_clean" (Libri2Mix-Clean) or "mix_both" (Libri2Mix-Noisy)
        max_samples: Limit number of samples (None = all)

    Expected directory structure:
        data_root/wav{sample_rate}k/{mode}/{subset}/
            mix_clean/  - Clean mixture (s1 + s2)
            mix_both/   - Noisy mixture (s1 + s2 + noise)
            s1/         - Speaker 1 target
            s2/         - Speaker 2 target
            noise/      - Noise component

    Returns:
        Dictionary with keys:
            - "mix": Mixed audio tensor [T]
            - "clean": Stacked speaker targets [2, T]
            - "filename": Audio filename
    """

    def __init__(
        self,
        data_root: str,
        subset: Literal["test", "dev", "train-100"] = "test",
        sample_rate: int = 8000,
        mode: Literal["min", "max"] = "min",
        mix_type: Literal["mix_clean", "mix_both"] = "mix_clean",
        max_samples: int = None,
    ):
        self.data_root = Path(data_root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.mode = mode
        self.mix_type = mix_type
        self.max_samples = max_samples

        sr_str = f"wav{sample_rate // 1000}k"
        self.base_path = self.data_root / sr_str / mode / subset

        self.mix_dir = self.base_path / mix_type
        self.s1_dir = self.base_path / "s1"
        self.s2_dir = self.base_path / "s2"

        for name, path in [("Mix", self.mix_dir), ("S1", self.s1_dir), ("S2", self.s2_dir)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} directory not found: {path}\n"
                    f"Expected: {data_root}/{sr_str}/{mode}/{subset}/{path.name}/"
                )

        self.mix_files = sorted(list(self.mix_dir.glob("*.wav")))

        if len(self.mix_files) == 0:
            raise ValueError(f"No .wav files found in {self.mix_dir}")

        if max_samples is not None:
            self.mix_files = self.mix_files[:max_samples]

        variant_name = "Libri2Mix-Clean" if mix_type == "mix_clean" else "Libri2Mix-Noisy"
        logger.info(f"Loaded {variant_name} {subset}: {len(self.mix_files)} samples")

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_path = self.mix_files[idx]
        filename = mix_path.name

        s1_path = self.s1_dir / filename
        s2_path = self.s2_dir / filename

        for label, path in [("s1", s1_path), ("s2", s2_path)]:
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        mix_audio, _ = torchaudio.load(mix_path)
        s1_audio, _ = torchaudio.load(s1_path)
        s2_audio, _ = torchaudio.load(s2_path)

        # Squeeze to 1D [T]
        mix_audio = mix_audio.squeeze(0)
        s1_audio = s1_audio.squeeze(0)
        s2_audio = s2_audio.squeeze(0)

        # Ensure same length
        min_len = min(len(mix_audio), len(s1_audio), len(s2_audio))
        mix_audio = mix_audio[:min_len]
        s1_audio = s1_audio[:min_len]
        s2_audio = s2_audio[:min_len]

        # Stack speakers as [2, T] — same format as PolSESS SB task
        clean = torch.stack([s1_audio, s2_audio])

        return {
            "mix": mix_audio,
            "clean": clean,
            "filename": filename,
        }


def libri2mix_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for variable-length Libri2Mix sequences.

    Pads all sequences to the maximum length in the batch.

    Returns:
        Dictionary with:
            - "mix": [B, T_max]
            - "clean": [B, 2, T_max]
            - "lengths": [B] original lengths
            - "filenames": List of filenames
    """
    max_len = max(sample["mix"].shape[0] for sample in batch)

    mix_padded = []
    clean_padded = []
    lengths = []
    filenames = []

    for sample in batch:
        mix = sample["mix"]
        clean = sample["clean"]
        orig_len = mix.shape[0]

        if orig_len < max_len:
            pad_len = max_len - orig_len
            mix = torch.nn.functional.pad(mix, (0, pad_len))
            clean = torch.nn.functional.pad(clean, (0, pad_len))

        mix_padded.append(mix)
        clean_padded.append(clean)
        lengths.append(orig_len)
        filenames.append(sample["filename"])

    return {
        "mix": torch.stack(mix_padded),
        "clean": torch.stack(clean_padded),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "filenames": filenames,
    }
