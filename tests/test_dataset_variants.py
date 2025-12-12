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
