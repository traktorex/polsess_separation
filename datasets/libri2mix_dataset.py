"""Libri2Mix dataset loader for cross-dataset evaluation.

Supports Libri2Mix mix_single variant (1 speaker + noise) for ES task evaluation.
Compatible with PolSESS-trained models.
"""

import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, List, Dict


class Libri2MixDataset(Dataset):
    """Libri2Mix dataset for speech enhancement evaluation.

    Loads mix_single variant: 1 speaker + noise mixtures.
    Returns same interface as PolSESSDataset for compatibility.

    Args:
        data_root: Path to Libri2Mix root (e.g., "C:/datasety/LibriMix/generated/Libri2Mix")
        subset: Dataset split - "test", "dev", or "train-100"
        sample_rate: Target sample rate (8000 or 16000)
        mode: "min" or "max" (utterance length mode)
        max_samples: Limit number of samples (None = all samples)

    Expected directory structure:
        data_root/wav{sample_rate}k/{mode}/{subset}/
            mix_single/  - Mixture files (speech + noise)
            s1/          - Clean speech target
            noise/       - Noise component (not used)
            s2/          - Second speaker (not used for mix_single)

    Returns:
        Dictionary with keys:
            - "mix": Mixed audio tensor [T]
            - "clean": Clean speech tensor [T]
            - "filename": Audio filename (for tracking)
    """

    def __init__(
        self,
        data_root: str,
        subset: Literal["test", "dev", "train-100"] = "test",
        sample_rate: int = 8000,
        mode: Literal["min", "max"] = "min",
        max_samples: int = None,
    ):
        self.data_root = Path(data_root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.mode = mode
        self.max_samples = max_samples

        # Construct paths
        sr_str = f"wav{sample_rate // 1000}k"
        self.base_path = self.data_root / sr_str / mode / subset

        self.mix_dir = self.base_path / "mix_single"
        self.clean_dir = self.base_path / "s1"

        # Verify directories exist
        if not self.mix_dir.exists():
            raise FileNotFoundError(
                f"Mix directory not found: {self.mix_dir}\n"
                f"Expected structure: {data_root}/wav{sample_rate//1000}k/{mode}/{subset}/mix_single/"
            )
        if not self.clean_dir.exists():
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}\n"
                f"Expected structure: {data_root}/wav{sample_rate//1000}k/{mode}/{subset}/s1/"
            )

        # Get list of audio files
        self.mix_files = sorted(list(self.mix_dir.glob("*.wav")))

        if len(self.mix_files) == 0:
            raise ValueError(f"No .wav files found in {self.mix_dir}")

        # Apply max_samples limit
        if max_samples is not None:
            self.mix_files = self.mix_files[:max_samples]

        print(f"Loaded Libri2Mix {subset} set: {len(self.mix_files)} samples")
        print(f"  Mix dir: {self.mix_dir}")
        print(f"  Clean dir: {self.clean_dir}")

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        """Load mixture and clean speech pair.

        Returns:
            dict: {"mix": tensor [T], "clean": tensor [T], "filename": str}
        """
        mix_path = self.mix_files[idx]
        filename = mix_path.name

        # Corresponding clean file has same name
        clean_path = self.clean_dir / filename

        if not clean_path.exists():
            raise FileNotFoundError(
                f"Clean file not found: {clean_path}\n"
                f"Mix file: {mix_path}"
            )

        # Load audio files
        mix_audio, mix_sr = torchaudio.load(mix_path)
        clean_audio, clean_sr = torchaudio.load(clean_path)

        # Verify sample rates match
        if mix_sr != self.sample_rate:
            raise ValueError(
                f"Mix file sample rate mismatch: expected {self.sample_rate}, got {mix_sr}\n"
                f"File: {mix_path}"
            )
        if clean_sr != self.sample_rate:
            raise ValueError(
                f"Clean file sample rate mismatch: expected {self.sample_rate}, got {clean_sr}\n"
                f"File: {clean_path}"
            )

        # Convert to mono if needed (should already be mono)
        if mix_audio.shape[0] > 1:
            mix_audio = mix_audio.mean(dim=0, keepdim=True)
        if clean_audio.shape[0] > 1:
            clean_audio = clean_audio.mean(dim=0, keepdim=True)

        # Squeeze to 1D [T]
        mix_audio = mix_audio.squeeze(0)
        clean_audio = clean_audio.squeeze(0)

        # Ensure same length (trim to shorter if needed)
        min_len = min(len(mix_audio), len(clean_audio))
        mix_audio = mix_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        return {
            "mix": mix_audio,
            "clean": clean_audio,
            "filename": filename,
        }


def libri2mix_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for variable-length Libri2Mix sequences.

    Pads all sequences in the batch to the maximum length.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with padded tensors:
            - "mix": [B, T_max] padded mixtures
            - "clean": [B, T_max] padded clean speech
            - "lengths": [B] original lengths before padding
            - "filenames": List of filenames
    """
    # Find max length in batch
    max_len = max(sample["mix"].shape[0] for sample in batch)

    # Pad all sequences to max length
    mix_padded = []
    clean_padded = []
    lengths = []
    filenames = []

    for sample in batch:
        mix = sample["mix"]
        clean = sample["clean"]
        orig_len = mix.shape[0]

        # Pad to max_len
        if orig_len < max_len:
            pad_len = max_len - orig_len
            mix = torch.nn.functional.pad(mix, (0, pad_len), mode="constant", value=0)
            clean = torch.nn.functional.pad(
                clean, (0, pad_len), mode="constant", value=0
            )

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
