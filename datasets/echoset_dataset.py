"""EchoSet dataset loader for cross-dataset evaluation.

EchoSet (Li et al., SonicSim) is a 2-speaker separation corpus rendered with
simulated room impulse responses in Matterport3D scenes — strong, realistic
reverberation, which is why it is used here as the reverberant counterpart to
the (anechoic) Libri2Mix cross-dataset check.

IMPORTANT — target convention differs from PolSESS. EchoSet ships only
``spk1_reverb.wav`` / ``spk2_reverb.wav``: the targets *retain* room reverb, so
the task is separation without dereverberation. PolSESS's SB target is the
**dry** ``clean/`` speech (see ``PolSESSDataset._compute_clean``), so a
PolSESS-trained model is trained to separate *and* dereverberate. Scores from
this loader are therefore NOT comparable to PolSESS or Libri2Mix numbers — a
model is penalised here for removing reverb it was trained to remove. Use it
for the relative comparison between models (all of which carry the same
mismatch), never for an absolute figure.

The mixture also does not sum exactly from the two targets
(``mix - (spk1 + spk2)`` is non-negligible), i.e. EchoSet mixtures carry an
additional noise component; the condition is reverberant *and* noisy.

Compatible with PolSESS-trained separation models (16 kHz, SB task).
"""

import logging
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, List, Dict

from .libri2mix_dataset import libri2mix_collate_fn

logger = logging.getLogger("polsess")

# The published tarball unpacks to this nested path; accept either the split
# parent or the archive root so callers need not spell it out.
_NESTED_ROOT = Path("gpfs-flash/hulab/public_datasets/audio_datasets/EchoSet")


class EchoSetDataset(Dataset):
    """EchoSet dataset for 2-speaker reverberant separation evaluation.

    Args:
        data_root: Path to the directory holding train/val/test (or the archive
            root, which is auto-descended).
        subset: Dataset split - "test", "val", or "train"
        max_samples: Limit number of samples (None = all)

    Expected directory structure:
        data_root/{subset}/{scene}/{room}/{utterance}/
            mix.wav          - Reverberant 2-speaker mixture
            spk1_reverb.wav  - Speaker 1 target, reverb retained
            spk2_reverb.wav  - Speaker 2 target, reverb retained

    Returns:
        Dictionary with keys:
            - "mix": Mixed audio tensor [T]
            - "clean": Stacked speaker targets [2, T]
            - "filename": Split-relative path identifying the utterance
    """

    def __init__(
        self,
        data_root: str,
        subset: Literal["test", "val", "train"] = "test",
        max_samples: int = None,
    ):
        self.data_root = Path(data_root)
        self.subset = subset
        self.max_samples = max_samples

        if not (self.data_root / subset).exists() and (
            self.data_root / _NESTED_ROOT / subset
        ).exists():
            self.data_root = self.data_root / _NESTED_ROOT
            logger.info(f"EchoSet: descended to nested archive root {self.data_root}")

        self.base_path = self.data_root / subset
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"EchoSet split not found: {self.base_path}\n"
                f"Expected: {data_root}/{subset}/<scene>/<room>/<utterance>/mix.wav"
            )

        # Utterance dirs sit three levels down: scene / room / utterance.
        self.utt_dirs = sorted(
            p.parent for p in self.base_path.glob("*/*/*/mix.wav")
        )

        if len(self.utt_dirs) == 0:
            raise ValueError(f"No mix.wav files found under {self.base_path}")

        if max_samples is not None:
            self.utt_dirs = self.utt_dirs[:max_samples]

        logger.info(f"Loaded EchoSet {subset}: {len(self.utt_dirs)} samples")

    def __len__(self):
        return len(self.utt_dirs)

    def __getitem__(self, idx):
        utt_dir = self.utt_dirs[idx]

        mix_path = utt_dir / "mix.wav"
        s1_path = utt_dir / "spk1_reverb.wav"
        s2_path = utt_dir / "spk2_reverb.wav"

        for label, path in [("spk1_reverb", s1_path), ("spk2_reverb", s2_path)]:
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
            "filename": str(utt_dir.relative_to(self.base_path)),
        }


# EchoSet utterances are all exactly 6.00 s, so the padding branch never fires;
# the batch dict contract is identical to Libri2Mix's, so we reuse its collate
# rather than keeping a second copy in sync.
echoset_collate_fn = libri2mix_collate_fn


def echoset_mixture_noise_floor(dataset, n: int = 50) -> float:
    """Report the level of the non-additive component, in dB below the targets.

    ``mix`` is not exactly ``spk1_reverb + spk2_reverb``; this quantifies the
    remainder so the noise condition can be stated rather than assumed.
    Diagnostic helper — not used by evaluation.
    """
    ratios = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        residual = item["mix"] - item["clean"].sum(dim=0)
        target_rms = item["clean"].sum(dim=0).pow(2).mean().sqrt()
        residual_rms = residual.pow(2).mean().sqrt()
        if residual_rms > 0:
            ratios.append(20 * torch.log10(target_rms / residual_rms).item())
    return sum(ratios) / len(ratios) if ratios else float("nan")
