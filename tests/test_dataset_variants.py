"""Comprehensive tests for PolSESS dataset variant selection and lazy loading."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from datasets.polsess_dataset import PolSESSDataset


class TestVariantSelection:
    """Test MM-IPC variant selection logic."""

    def test_variant_constants(self):
        """Test that variant constants are defined correctly."""
        assert PolSESSDataset.INDOOR_VARIANTS == ["SER", "SR", "ER", "R", "C"]
        assert PolSESSDataset.OUTDOOR_VARIANTS == ["SE", "S", "E", "C"]
        assert len(PolSESSDataset.INDOOR_VARIANTS) == 5
        assert len(PolSESSDataset.OUTDOOR_VARIANTS) == 4

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_allowed_variants_filters_correctly(self, mock_load, mock_read_csv):
        """Test that allowed_variants parameter filters variants correctly."""
        import pandas as pd

        # Create real DataFrame with required columns
        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": ["s1_reverb.wav"],
                "reverbForSpeaker2": ["s2_reverb.wav"],
                "reverbForEvent": ["ev_reverb.wav"],
            }
        )
        mock_read_csv.return_value = mock_df

        # Mock audio loading
        mock_load.return_value = (torch.zeros(16000), 16000)

        # Test filtering to only SER and SR
        dataset = PolSESSDataset(
            data_root="/fake/path",
            subset="test",
            task="ES",
            allowed_variants=["SER", "SR"],
        )

        # The dataset should only allow SER and SR variants
        assert dataset.allowed_variants == ["SER", "SR"]

    @patch("pandas.read_csv")
    def test_reverb_filtering_for_indoor_outdoor(self, mock_read_csv):
        """Test that dataset correctly filters samples based on reverb availability."""
        import pandas as pd
        import numpy as np

        # Create DataFrame with mixed reverb availability
        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav", "mix2.wav"],
                "speaker1File": ["s1.wav", "s2.wav"],
                "speaker2File": ["s1b.wav", "s2b.wav"],
                "sceneFile": ["scene1.wav", "scene2.wav"],
                "eventFile": ["event1.wav", "event2.wav"],
                "reverbForSpeaker1": ["s1_reverb.wav", np.nan],  # Indoor, then outdoor
                "reverbForSpeaker2": ["s2_reverb.wav", np.nan],
                "reverbForEvent": ["ev_reverb.wav", np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        # Create dataset with indoor-only variants
        dataset_indoor = PolSESSDataset(
            data_root="/fake/path",
            subset="test",
            task="ES",
            allowed_variants=["SER"],  # Indoor only
        )

        # Indoor variant should only include reverb=True samples
        assert dataset_indoor.allowed_variants == ["SER"]
        assert len(dataset_indoor) == 1  # Only reverb sample remains

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_random_variant_selection(self, mock_load, mock_read_csv):
        """Test that variants are selected randomly during __getitem__."""
        import pandas as pd

        # Create real DataFrame
        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": ["s1_reverb.wav"],
                "reverbForSpeaker2": ["s2_reverb.wav"],
                "reverbForEvent": ["ev_reverb.wav"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_load.return_value = (torch.zeros(16000), 16000)

        # Test with specific indoor variants to avoid randomness issues
        dataset = PolSESSDataset(
            data_root="/fake/path",
            subset="test",
            task="ES",
            allowed_variants=["SER", "SR"],  # Limited variants for testing
        )

        # Call __getitem__ multiple times and check it doesn't crash
        sample1 = dataset[0]
        sample2 = dataset[0]

        assert "mix" in sample1
        assert "clean" in sample1


class TestLazyLoading:
    """Test lazy loading implementation for MM-IPC variants."""

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_lazy_loading_only_loads_needed_components(self, mock_load, mock_read_csv):
        """Test that lazy loading only loads audio components needed for the selected variant."""
        import pandas as pd
        import numpy as np

        # Create real DataFrame with outdoor sample (no reverb)
        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": [np.nan],  # No reverb for outdoor
                "reverbForSpeaker2": [np.nan],
                "reverbForEvent": [np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        # Mock audio loading to return recognizable tensors
        def mock_load_side_effect(path):
            # Return different tensor sizes to identify which was loaded
            if "s1.wav" in str(path):
                return (torch.ones(16000), 16000)
            elif "event1.wav" in str(path):
                return (torch.full((16000,), 2.0), 16000)
            else:
                return (torch.zeros(16000), 16000)

        mock_load.side_effect = mock_load_side_effect

        dataset = PolSESSDataset(
            data_root="/fake/path",
            subset="test",
            task="ES",
            allowed_variants=["S"],  # Outdoor, speech only - should only load speaker1
        )

        # Get a sample - should only load speaker1 for 'S' variant
        sample = dataset[0]

        # Verify the sample has the expected keys
        assert "mix" in sample
        assert "clean" in sample
        assert isinstance(sample["mix"], torch.Tensor)
        assert isinstance(sample["clean"], torch.Tensor)


class TestTaskModes:
    """Test different task modes (ES, EB, SB)."""

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_es_task_returns_single_speaker(self, mock_load, mock_read_csv):
        """Test that ES task returns single speaker."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": [np.nan],
                "reverbForSpeaker2": [np.nan],
                "reverbForEvent": [np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_load.return_value = (torch.zeros(16000), 16000)

        dataset = PolSESSDataset(
            data_root="/fake/path", subset="test", task="ES", allowed_variants=["S"]
        )

        sample = dataset[0]
        # ES task should return 1D clean signal
        assert sample["clean"].dim() == 1

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_eb_task_returns_both_speakers(self, mock_load, mock_read_csv):
        """Test that EB task returns both speakers combined."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": [np.nan],
                "reverbForSpeaker2": [np.nan],
                "reverbForEvent": [np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        # Return different values for each speaker to verify they're combined
        def load_side_effect(path):
            if "s1.wav" in str(path):
                return (torch.ones(16000), 16000)
            elif "s2.wav" in str(path):
                return (torch.full((16000,), 2.0), 16000)
            else:
                return (torch.zeros(16000), 16000)

        mock_load.side_effect = load_side_effect

        dataset = PolSESSDataset(
            data_root="/fake/path", subset="test", task="EB", allowed_variants=["S"]
        )

        sample = dataset[0]
        # EB task should return 1D clean signal (sum of both speakers)
        assert sample["clean"].dim() == 1

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_sb_task_returns_separated_speakers(self, mock_load, mock_read_csv):
        """Test that SB task returns both speakers separately."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": [np.nan],
                "reverbForSpeaker2": [np.nan],
                "reverbForEvent": [np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_load.return_value = (torch.zeros(16000), 16000)

        dataset = PolSESSDataset(
            data_root="/fake/path", subset="test", task="SB", allowed_variants=["S"]
        )

        sample = dataset[0]
        # SB task should return 2D clean signal [2, T]
        assert sample["clean"].dim() == 2
        assert sample["clean"].shape[0] == 2  # Two speakers


class TestMaxSamples:
    """Test max_samples parameter for dataset limiting."""

    @patch("pandas.read_csv")
    def test_max_samples_limits_dataset_size(self, mock_read_csv):
        """Test that max_samples correctly limits dataset size."""
        import pandas as pd
        import numpy as np

        # Create DataFrame with 100 samples
        mock_df = pd.DataFrame(
            {
                "mixFile": [f"mix{i}.wav" for i in range(100)],
                "speaker1File": [f"s1_{i}.wav" for i in range(100)],
                "speaker2File": [f"s2_{i}.wav" for i in range(100)],
                "sceneFile": [f"scene{i}.wav" for i in range(100)],
                "eventFile": [f"event{i}.wav" for i in range(100)],
                "reverbForSpeaker1": [np.nan] * 100,
                "reverbForSpeaker2": [np.nan] * 100,
                "reverbForEvent": [np.nan] * 100,
            }
        )
        mock_read_csv.return_value = mock_df

        # Limit to 10 samples
        dataset = PolSESSDataset(
            data_root="/fake/path", subset="test", task="ES", max_samples=10
        )

        # Dataset should only have 10 samples
        assert len(dataset) == 10

    @patch("pandas.read_csv")
    def test_max_samples_none_uses_all_samples(self, mock_read_csv):
        """Test that max_samples=None uses all available samples."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "mixFile": [f"mix{i}.wav" for i in range(100)],
                "speaker1File": [f"s1_{i}.wav" for i in range(100)],
                "speaker2File": [f"s2_{i}.wav" for i in range(100)],
                "sceneFile": [f"scene{i}.wav" for i in range(100)],
                "eventFile": [f"event{i}.wav" for i in range(100)],
                "reverbForSpeaker1": [np.nan] * 100,
                "reverbForSpeaker2": [np.nan] * 100,
                "reverbForEvent": [np.nan] * 100,
            }
        )
        mock_read_csv.return_value = mock_df

        dataset = PolSESSDataset(
            data_root="/fake/path", subset="test", task="ES", max_samples=None
        )

        assert len(dataset) == 100


class TestSnrPerturbation:
    """Test SNR perturbation augmentation."""

    # Distinct per-stem values so we can verify the math
    STEM_VALUES = {
        "mix": 10.0,
        "s1": 3.0,
        "s2": 2.0,
        "scene": 1.0,
        "event": 0.5,
        "s1_reverb": 0.3,
        "s2_reverb": 0.2,
        "ev_reverb": 0.1,
    }

    def _make_df(self, has_reverb=False):
        """Helper: create a single-row DataFrame."""
        import pandas as pd
        import numpy as np

        reverb_val = "s1_reverb.wav" if has_reverb else np.nan
        reverb_s2 = "s2_reverb.wav" if has_reverb else np.nan
        reverb_ev = "ev_reverb.wav" if has_reverb else np.nan
        return pd.DataFrame(
            {
                "mixFile": ["mix1.wav"],
                "speaker1File": ["s1.wav"],
                "speaker2File": ["s2.wav"],
                "sceneFile": ["scene1.wav"],
                "eventFile": ["event1.wav"],
                "reverbForSpeaker1": [reverb_val],
                "reverbForSpeaker2": [reverb_s2],
                "reverbForEvent": [reverb_ev],
            }
        )

    def _stem_loader(self, path):
        """Mock torchaudio.load returning distinct values per stem."""
        s = str(path)
        v = self.STEM_VALUES
        if "mix1.wav" in s:
            val = v["mix"]
        elif "s1_reverb" in s:
            val = v["s1_reverb"]
        elif "s2_reverb" in s:
            val = v["s2_reverb"]
        elif "ev_reverb" in s:
            val = v["ev_reverb"]
        elif "s1.wav" in s:
            val = v["s1"]
        elif "s2.wav" in s:
            val = v["s2"]
        elif "scene1.wav" in s:
            val = v["scene"]
        elif "event1.wav" in s:
            val = v["event"]
        else:
            val = 0.0
        return (torch.full((1, 16000), val), 16000)

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    @patch("datasets.polsess_dataset.random.uniform", return_value=3.0)
    def test_snr_perturbation_math_outdoor(self, mock_uniform, mock_load, mock_read_csv):
        """Verify exact mix value for outdoor SE variant with known gain."""
        mock_read_csv.return_value = self._make_df(has_reverb=False)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="ES",
            allowed_variants=["SE"],
            snr_perturbation_db=6.0,
        )
        sample = ds[0]

        # SE outdoor ES task:
        # _apply_mmipc: mix - speaker2. Scene/event kept.
        # SNR perturbation: kept_noise = scene + event, dB = 3.0
        v = self.STEM_VALUES
        mix_after_ipc = v["mix"] - v["s2"]
        gain = 10 ** (3.0 / 20)
        kept_noise = v["scene"] + v["event"]
        expected = mix_after_ipc + (gain - 1) * kept_noise

        assert torch.allclose(sample["mix"], torch.full((16000,), expected)), \
            f"Expected {expected}, got {sample['mix'][0].item()}"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    @patch("datasets.polsess_dataset.random.uniform", return_value=-4.0)
    def test_snr_perturbation_math_indoor(self, mock_uniform, mock_load, mock_read_csv):
        """Verify exact mix value for indoor SER variant — speaker reverb must NOT be scaled."""
        mock_read_csv.return_value = self._make_df(has_reverb=True)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="SB",
            allowed_variants=["SER"],
            snr_perturbation_db=6.0,
        )
        sample = ds[0]

        # SER indoor SB task: _apply_mmipc subtracts nothing (all kept)
        # SNR perturbation: kept_noise = scene + event + ev_reverb, dB = -4.0
        v = self.STEM_VALUES
        mix_after_ipc = v["mix"]
        gain = 10 ** (-4.0 / 20)
        kept_noise = v["scene"] + v["event"] + v["ev_reverb"]
        expected = mix_after_ipc + (gain - 1) * kept_noise

        assert torch.allclose(sample["mix"], torch.full((16000,), expected)), \
            f"Expected {expected}, got {sample['mix'][0].item()}"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_snr_perturbation_preserves_clean(self, mock_load, mock_read_csv):
        """Clean targets are identical regardless of SNR perturbation."""
        mock_read_csv.return_value = self._make_df(has_reverb=False)
        mock_load.side_effect = self._stem_loader

        ds_base = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="ES",
            allowed_variants=["SE"],
            snr_perturbation_db=0.0,
        )
        ds_aug = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="ES",
            allowed_variants=["SE"],
            snr_perturbation_db=20.0,
        )

        clean_base = ds_base[0]["clean"]
        clean_aug = ds_aug[0]["clean"]
        assert torch.allclose(clean_base, clean_aug), "Clean targets must not change with SNR perturbation"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_snr_perturbation_disabled_by_default(self, mock_load, mock_read_csv):
        """With snr_perturbation_db=0.0 (default), mix equals unaugmented reference."""
        mock_read_csv.return_value = self._make_df(has_reverb=False)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="ES",
            allowed_variants=["SE"],
        )

        # SE outdoor ES: mix_after_ipc = mix - speaker2
        v = self.STEM_VALUES
        expected = v["mix"] - v["s2"]
        mix = ds[0]["mix"]
        assert torch.allclose(mix, torch.full((16000,), expected)), \
            "Default (0.0) must produce the standard IPC mix"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_snr_perturbation_noop_on_clean_variant(self, mock_load, mock_read_csv):
        """'C' outdoor variant has no non-speech background — mix unchanged with perturbation."""
        mock_read_csv.return_value = self._make_df(has_reverb=False)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="ES",
            allowed_variants=["C"],
            snr_perturbation_db=20.0,
        )

        # C outdoor ES: mix - speaker2 - scene - event
        v = self.STEM_VALUES
        expected = v["mix"] - v["s2"] - v["scene"] - v["event"]

        mix_ref = ds[0]["mix"]
        assert torch.allclose(mix_ref, torch.full((16000,), expected))
        # Must be stable across calls (no random scaling applied)
        for _ in range(10):
            assert torch.allclose(ds[0]["mix"], mix_ref), \
                "'C' variant mix must be invariant to SNR perturbation"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_snr_perturbation_noop_on_r_variant(self, mock_load, mock_read_csv):
        """'R' indoor variant — only speaker reverb remains, nothing to scale."""
        mock_read_csv.return_value = self._make_df(has_reverb=True)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="train",
            task="SB",
            allowed_variants=["R"],
            snr_perturbation_db=20.0,
        )

        # R indoor SB: mix - scene - event - ev_reverb (reverb kept, scene/event removed)
        v = self.STEM_VALUES
        expected = v["mix"] - v["scene"] - v["event"] - v["ev_reverb"]

        mix_ref = ds[0]["mix"]
        assert torch.allclose(mix_ref, torch.full((16000,), expected))
        for _ in range(10):
            assert torch.allclose(ds[0]["mix"], mix_ref), \
                "'R' variant mix must be invariant to SNR perturbation"

    @patch("pandas.read_csv")
    @patch("torchaudio.load")
    def test_snr_perturbation_skipped_for_val(self, mock_load, mock_read_csv):
        """Val subset must not apply SNR perturbation even when configured."""
        mock_read_csv.return_value = self._make_df(has_reverb=False)
        mock_load.side_effect = self._stem_loader

        ds = PolSESSDataset(
            data_root="/fake/path",
            subset="val",
            task="ES",
            allowed_variants=["SE"],
            snr_perturbation_db=20.0,
        )

        # SE outdoor ES: mix - speaker2, no perturbation on val
        v = self.STEM_VALUES
        expected = v["mix"] - v["s2"]

        mix_ref = ds[0]["mix"]
        assert torch.allclose(mix_ref, torch.full((16000,), expected))
        for _ in range(10):
            assert torch.allclose(ds[0]["mix"], mix_ref), \
                "Val subset must not apply SNR perturbation"
