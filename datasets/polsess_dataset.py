"""PolSESS Dataset for Speech Enhancement/Separation with MM-IPC augmentation."""

import pandas as pd
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any


class PolSESSDataset(Dataset):
    """
    Dataset for PolSESS with MM-IPC augmentation.

    Args:
        data_root: Root directory containing train/val/test folders
        subset: 'train', 'val', or 'test'
        task: 'ES' (single speaker) or 'EB' (both speakers)
        allowed_variants: MM-IPC variants (None = all, ['SER'] = specific, ['SER','SR'] = subset)
        max_samples: Maximum number of samples to load (None = all samples)
    """

    INDOOR_VARIANTS = ["SER", "SR", "ER", "R", "C"]
    OUTDOOR_VARIANTS = ["SE", "S", "E", "C"]

    def __init__(
        self,
        data_root,
        subset="train",
        task="ES",
        allowed_variants=None,
        max_samples=None,
    ):
        self.data_root = Path(data_root)
        self.subset = subset
        self.task = task
        self.max_samples = max_samples
        self._allowed_variants = None  # Private attribute

        # Automatically derive CSV filename from data_root
        # Example: "F:\\...\\PolSESS_C_in\\PolSESS_C_in" -> corpus_name = "PolSESS_C_in"
        corpus_name = self.data_root.name
        csv_filename = f"corpus_{corpus_name}_{subset}_final.csv"
        csv_path = self.data_root / subset / csv_filename

        # Load full metadata (unfiltered)
        self.full_metadata = pd.read_csv(csv_path)

        # Apply filtering via property setter
        self.allowed_variants = allowed_variants

    @property
    def allowed_variants(self):
        """Get currently allowed MM-IPC variants."""
        return self._allowed_variants

    @allowed_variants.setter
    def allowed_variants(self, variants):
        """Set allowed variants and re-filter metadata."""
        self._allowed_variants = variants
        self._filter_metadata()

    def _filter_metadata(self):
        """Filter metadata based on allowed_variants and max_samples."""
        # Start with full metadata
        self.metadata = self.full_metadata.copy()

        # Filter by variants if specified
        if self._allowed_variants:
            needs_reverb = any(v in self.INDOOR_VARIANTS for v in self._allowed_variants)
            needs_no_reverb = any(
                v in self.OUTDOOR_VARIANTS for v in self._allowed_variants
            )

            if needs_reverb and not needs_no_reverb:
                self.metadata = self.metadata[
                    self.metadata["reverbForSpeaker1"].notna()
                ].reset_index(drop=True)
            elif needs_no_reverb and not needs_reverb:
                self.metadata = self.metadata[
                    self.metadata["reverbForSpeaker1"].isna()
                ].reset_index(drop=True)

        # Limit dataset size if specified
        if self.max_samples is not None:
            self.metadata = self.metadata.head(self.max_samples).reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        has_reverb = pd.notna(row["reverbForSpeaker1"])

        paths = self._build_paths(row, has_reverb)
        variant = self._choose_variant(has_reverb, idx)
        audio = self._lazy_load(paths, variant, has_reverb)
        mix = self._apply_mmipc(audio, has_reverb)
        clean = self._compute_clean(audio)

        return {"mix": mix, "clean": clean, "background_complexity": variant}

    def _build_paths(self, row, has_reverb):
        """Build dictionary of file paths."""
        paths = {
            "mix": self.data_root / self.subset / "mix" / row["mixFile"],
            "speaker1": self.data_root / self.subset / "clean" / row["speaker1File"],
            "speaker2": self.data_root / self.subset / "clean" / row["speaker2File"],
            "scene": self.data_root / self.subset / "scene" / row["sceneFile"],
            "event": self.data_root / self.subset / "event" / row["eventFile"],
        }

        if has_reverb:
            paths["sp1_reverb"] = (
                self.data_root / self.subset / "sp1_reverb" / row["reverbForSpeaker1"]
            )
            paths["sp2_reverb"] = (
                self.data_root / self.subset / "sp2_reverb" / row["reverbForSpeaker2"]
            )
            paths["ev_reverb"] = (
                self.data_root / self.subset / "ev_reverb" / row["reverbForEvent"]
            )

        return paths

    def _choose_variant(self, has_reverb, idx):
        """Choose MM-IPC variant, respecting reverb compatibility.

        For the val and test subsets, selection is deterministic (seeded by idx)
        so each sample gets the same variant across epochs and across runs
        (reproducible evaluation, survey gap 6). Only training draws randomly.
        """
        # Get compatible variants
        if self.allowed_variants:
            compatible = [
                v
                for v in self.allowed_variants
                if v in (self.INDOOR_VARIANTS if has_reverb else self.OUTDOOR_VARIANTS)
            ]
            if not compatible:
                raise ValueError(
                    f"No compatible variant in {self.allowed_variants} for has_reverb={has_reverb}"
                )
        else:
            compatible = self.INDOOR_VARIANTS if has_reverb else self.OUTDOOR_VARIANTS

        # For val/test, use deterministic selection (seeded by sample index)
        if self.subset in ("val", "test"):
            rng = random.Random(idx)
            return rng.choice(compatible)

        # For training, use random selection
        return random.choice(compatible)

    def _lazy_load(self, paths, variant, has_reverb):
        """Load only files needed for this task and variant.

        Variant letters indicate which components to KEEP:
        - "S" = keep Scene
        - "E" = keep Event
        - "R" = keep Reverb
        - "C" = Clean (remove all: scene, event, reverb)
        """
        audio = {}

        # Always load mix and speakers
        audio["mix"], _ = torchaudio.load(paths["mix"])
        audio["speaker1"], _ = torchaudio.load(paths["speaker1"])
        audio["speaker2"], _ = torchaudio.load(paths["speaker2"])

        # Load components to remove based on variant
        if "S" not in variant:
            audio["scene"], _ = torchaudio.load(paths["scene"])

        if "E" not in variant:
            audio["event"], _ = torchaudio.load(paths["event"])
            if has_reverb:
                audio["ev_reverb"], _ = torchaudio.load(paths["ev_reverb"])

        # Load speaker reverbs to remove (indoor samples only)
        if has_reverb:
            if variant == "C":
                # C variant: remove ALL reverb (cleanest output)
                audio["sp1_reverb"], _ = torchaudio.load(paths["sp1_reverb"])
                audio["sp2_reverb"], _ = torchaudio.load(paths["sp2_reverb"])
            elif self.task == "ES":
                # ES task: always remove sp2_reverb (along with speaker2)
                audio["sp2_reverb"], _ = torchaudio.load(paths["sp2_reverb"])

        for key in audio:
            audio[key] = self._ensure_1d(audio[key])

        return audio

    def _ensure_1d(self, tensor):
        while tensor.dim() > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    # ------------------------------------------------------------------
    # MM-IPC variant algebra (executable proof: scripts/audit_mmipc.py)
    # ------------------------------------------------------------------
    # Every mixture is rendered from an additive layer decomposition
    # (Klec et al., Eq. 1; docs/MMIPC_PAPER_VERIFICATION.md):
    #
    #   indoor  (has_reverb):
    #     mix = sp1_dry + sp1_reverb + sp2_dry + sp2_reverb
    #           + scene + event_dry + event_reverb
    #   outdoor (no reverb):
    #     mix = sp1_dry + sp2_dry + scene + event_dry
    #
    # The scene layer carries no separate reverb tail (it is ambient
    # background rendered directly into the room). Layer -> stored file:
    #     sp1_dry      = clean/speaker1File          sp2_dry      = clean/speaker2File
    #     scene        = scene/sceneFile             event_dry    = event/eventFile
    #     sp1_reverb   = sp1_reverb/reverbForSpeaker1  (indoor only)
    #     sp2_reverb   = sp2_reverb/reverbForSpeaker2  (indoor only)
    #     event_reverb = ev_reverb/reverbForEvent      (indoor only)
    #
    # MM-IPC removes a layer by subtracting its stored file (subtraction ==
    # adding the phase-inverted signal). Variant letters name what is KEPT
    # (S=scene, E=event, R=speaker reverb tail; C=clean, keep none).
    #
    #   Background layers REMOVED per variant (task-independent):
    #     indoor   SER -> (none)
    #              SR  -> event_dry + event_reverb
    #              ER  -> scene
    #              R   -> scene + event_dry + event_reverb
    #              C   -> scene + event_dry + event_reverb + sp1_reverb + sp2_reverb
    #     outdoor  SE  -> (none)
    #              S   -> event_dry
    #              E   -> scene
    #              C   -> scene + event_dry
    #
    #   Speaker layers removed by TASK, on top of the variant removal:
    #     ES      -> also remove sp2_dry (speaker2) and, indoor, sp2_reverb
    #                (speaker 2 is cancelled tail-and-all — never a half-removal)
    #     EB / SB -> keep both speakers (no speaker removal)
    #
    #   Target (_compute_clean):
    #     ES -> sp1_dry     EB -> sp1_dry + sp2_dry     SB -> [sp1_dry, sp2_dry]
    #
    # Key identity: under ES, the C variant cancels every non-sp1 layer, so the
    # mix collapses to exactly sp1_dry (the dry speaker-1 target). That the
    # augmentation is thus numerically lossless — and that every OTHER variant's
    # retained-layer reconstruction is exact too — is verified end-to-end on real
    # data by scripts/audit_mmipc.py (residual vs. the 16-bit PCM floor, ~1e-4 RMS).
    #
    # Not a double subtraction: for ES indoor non-C variants, speaker 2 is removed
    # via BOTH sp2_dry (task rule) and sp2_reverb (task rule) — correct additive
    # algebra, because reverberated speaker 2 = sp2_dry + sp2_reverb.
    def _apply_mmipc(self, audio, has_reverb):
        """Apply MM-IPC by removing components from mix.

        See the variant-algebra comment block directly above this method for the
        full mix decomposition and per-variant removed-layers truth table;
        scripts/audit_mmipc.py is the executable proof that this reconstruction
        is lossless to the 16-bit PCM quantization floor.
        """
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

    def _compute_clean(self, audio):
        """Compute clean target based on task."""
        if self.task == "EB":
            return audio["speaker1"] + audio["speaker2"]
        elif self.task == "ES":
            return audio["speaker1"]
        elif self.task == "SB":
            return torch.stack([audio["speaker1"], audio["speaker2"]])
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def condition_labels(self):
        """Return per-sample acoustic-condition labels for stratified analysis.

        Surfaces the corpus columns describing each mixture's acoustic condition
        — ``sceneClass`` (background scene category), ``eventClass`` (foreground
        event category) and ``SSR`` (speaker-to-speaker ratio in dB) — aligned
        row-for-row with this dataset's active ``metadata`` (i.e. after
        allowed_variants / max_samples filtering), so row *i* here corresponds to
        ``self[i]``. Standalone accessor for per-condition SI-SDRi breakdowns in
        the analysis chapter; deliberately NOT wired into training or evaluation.

        Returns:
            pandas.DataFrame with whichever of ['sceneClass', 'eventClass', 'SSR']
            exist in the corpus, indexed 0..len(self)-1.
        """
        wanted = ["sceneClass", "eventClass", "SSR"]
        present = [c for c in wanted if c in self.metadata.columns]
        return self.metadata[present].reset_index(drop=True)


def polsess_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for PolSESS dataset.

    Args:
        batch: List of samples from PolSESSDataset

    Returns:
        Dictionary with batched tensors:
            - "mix": [B, T] mixed/noisy audio
            - "clean": [B, T] or [B, 2, T] clean target(s)
            - "background_complexity": List of MM-IPC variant strings
    """
    return {
        "mix": torch.stack([sample["mix"] for sample in batch]),
        "clean": torch.stack([sample["clean"] for sample in batch]),
        "background_complexity": [sample["background_complexity"] for sample in batch],
    }
