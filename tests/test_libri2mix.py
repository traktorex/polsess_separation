"""Tests for Libri2Mix dataset loader.

LibriMix is used for cross-dataset evaluation to test model generalization.
Tests validate 2-speaker separation (SB task) with mix_clean and mix_both variants.

The directory/getitem/collate tests below build a minimal synthetic Libri2Mix
tree under `tmp_path` and exercise the real loader end to end (no mocking of
`Path`/`torchaudio.load` internals) — replaces the previous
try/except-pass "does it at least not blow up" no-ops (survey gap 17).
"""

import pytest
import torch
import torchaudio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets.libri2mix_dataset import Libri2MixDataset, libri2mix_collate_fn


def _make_libri2mix_root(
    tmp_path: Path,
    sample_rate: int = 8000,
    mode: str = "min",
    subset: str = "test",
    mix_type: str = "mix_clean",
    n_samples: int = 3,
    n_samples_frames: int = 4000,
):
    """Build a minimal on-disk Libri2Mix tree matching the loader's expected
    layout: data_root/wav{sr}k/{mode}/{subset}/{mix_type,s1,s2}/*.wav

    s1/s2 are constant-valued (distinguishable) tensors and mix == s1 + s2,
    matching the real corpus's construction for the mix_clean variant so the
    "mixture is sum of sources" contract is exercised, not just shapes/dtypes.
    Returns (data_root, expected) where expected is a list of
    (filename, s1_value, s2_value) in the sorted order the loader will see.
    """
    sr_str = f"wav{sample_rate // 1000}k"
    base = tmp_path / sr_str / mode / subset
    mix_dir = base / mix_type
    s1_dir = base / "s1"
    s2_dir = base / "s2"
    for d in (mix_dir, s1_dir, s2_dir):
        d.mkdir(parents=True, exist_ok=True)

    expected = []
    for i in range(n_samples):
        filename = f"utt{i}.wav"
        s1_value = 0.1 * (i + 1)
        s2_value = -0.1 * (i + 1)  # distinguishable sign from s1
        s1_audio = torch.full((1, n_samples_frames), s1_value)
        s2_audio = torch.full((1, n_samples_frames), s2_value)
        mix_audio = s1_audio + s2_audio  # mixture = sum of sources

        torchaudio.save(str(s1_dir / filename), s1_audio, sample_rate)
        torchaudio.save(str(s2_dir / filename), s2_audio, sample_rate)
        torchaudio.save(str(mix_dir / filename), mix_audio, sample_rate)
        expected.append((filename, s1_value, s2_value))

    return tmp_path, expected


class TestLibri2MixDataset:
    """Tests for Libri2Mix dataset initialization and loading."""

    def test_libri2mix_requires_valid_directory_structure(self):
        """Test that dataset validates directory structure."""
        # Should raise FileNotFoundError for invalid path
        with pytest.raises(FileNotFoundError, match="Mix directory not found"):
            Libri2MixDataset(
                data_root="/nonexistent/path",
                subset="test",
                sample_rate=8000,
            )

    @pytest.mark.parametrize("subset", ["test", "dev", "train-100"])
    def test_libri2mix_subset_options(self, tmp_path, subset):
        """Each valid subset resolves to its own directory and loads real files."""
        data_root, expected = _make_libri2mix_root(tmp_path, subset=subset, n_samples=2)

        dataset = Libri2MixDataset(
            data_root=str(data_root),
            subset=subset,
            sample_rate=8000,
        )

        assert len(dataset) == 2
        assert dataset.base_path == data_root / "wav8k" / "min" / subset

    @pytest.mark.parametrize("sample_rate", [8000, 16000])
    def test_libri2mix_sample_rate_options(self, tmp_path, sample_rate):
        """8kHz and 16kHz roots resolve to distinct wav{sr}k directories."""
        data_root, expected = _make_libri2mix_root(
            tmp_path, sample_rate=sample_rate, n_samples=2
        )

        dataset = Libri2MixDataset(
            data_root=str(data_root),
            subset="test",
            sample_rate=sample_rate,
        )

        assert len(dataset) == 2
        assert dataset.base_path.parent.parent.name == f"wav{sample_rate // 1000}k"

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_libri2mix_mode_options(self, tmp_path, mode):
        """min/max mode resolves to its own directory and loads real files."""
        data_root, expected = _make_libri2mix_root(tmp_path, mode=mode, n_samples=2)

        dataset = Libri2MixDataset(
            data_root=str(data_root),
            subset="test",
            sample_rate=8000,
            mode=mode,
        )

        assert len(dataset) == 2
        assert dataset.base_path.parent.name == mode

    def test_libri2mix_getitem_returns_correct_format(self, tmp_path):
        """__getitem__ returns real audio in the documented shape/dtype/keys,
        and the mixture equals the sum of the two speaker sources (the
        mix_clean contract)."""
        data_root, expected = _make_libri2mix_root(tmp_path, n_samples_frames=4000)

        dataset = Libri2MixDataset(data_root=str(data_root), subset="test", sample_rate=8000)
        sample = dataset[0]
        filename, s1_value, s2_value = expected[0]

        assert set(sample.keys()) == {"mix", "clean", "filename"}
        assert sample["filename"] == filename
        assert isinstance(sample["mix"], torch.Tensor)
        assert isinstance(sample["clean"], torch.Tensor)
        assert sample["mix"].dtype == torch.float32
        assert sample["clean"].dtype == torch.float32

        # mix: [T], clean: [2, T]
        assert sample["mix"].dim() == 1
        assert sample["clean"].dim() == 2
        assert sample["clean"].shape[0] == 2
        assert sample["mix"].shape[-1] == sample["clean"].shape[-1]

        # Speaker identity: clean[0] is s1, clean[1] is s2 (not swapped).
        assert torch.allclose(
            sample["clean"][0], torch.full_like(sample["clean"][0], s1_value), atol=1e-4
        )
        assert torch.allclose(
            sample["clean"][1], torch.full_like(sample["clean"][1], s2_value), atol=1e-4
        )

        # mixture = sum of sources (mix_clean contract).
        assert torch.allclose(sample["mix"], sample["clean"][0] + sample["clean"][1], atol=1e-4)

    def test_libri2mix_getitem_missing_speaker_file_raises(self, tmp_path):
        """A mix file with no matching s1/s2 file raises FileNotFoundError
        (real failure path, not mocked)."""
        data_root, expected = _make_libri2mix_root(tmp_path, n_samples=1)
        # Delete s2's file so it's missing at __getitem__ time.
        filename = expected[0][0]
        (data_root / "wav8k" / "min" / "test" / "s2" / filename).unlink()

        dataset = Libri2MixDataset(data_root=str(data_root), subset="test", sample_rate=8000)
        with pytest.raises(FileNotFoundError, match="s2 file not found"):
            dataset[0]

    def test_libri2mix_max_samples_limiting(self, tmp_path):
        """max_samples truncates the (sorted) file list, not just accepts the kwarg."""
        data_root, expected = _make_libri2mix_root(tmp_path, n_samples=5)

        dataset = Libri2MixDataset(
            data_root=str(data_root), subset="test", sample_rate=8000, max_samples=2
        )

        assert len(dataset) == 2
        # Truncation keeps the first two in sorted filename order.
        assert [p.name for p in dataset.mix_files] == [expected[0][0], expected[1][0]]

    def test_libri2mix_path_construction(self):
        """Test correct path construction for different configs (error-message
        smoke test on a guaranteed-missing root)."""
        test_cases = [
            (8000, "min", "wav8k/min/test"),
            (16000, "max", "wav16k/max/test"),
            (8000, "max", "wav8k/max/dev"),
        ]

        for sample_rate, mode, expected_path_part in test_cases:
            subset = "test" if "test" in expected_path_part else "dev"
            with pytest.raises(FileNotFoundError, match=expected_path_part.replace("/", r"\/")):
                Libri2MixDataset(
                    data_root="/tmp/libri2mix_definitely_missing",
                    subset=subset,
                    sample_rate=sample_rate,
                    mode=mode,
                )


class TestLibri2MixCollateFunction:
    """Tests for Libri2Mix collate function."""

    def test_libri2mix_collate_padding(self):
        """Test that collate function pads sequences to max length."""
        # Create batch with different lengths — clean is [2, T] (2 speakers)
        batch = [
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "file1.wav"},
            {"mix": torch.randn(10000), "clean": torch.randn(2, 10000), "filename": "file2.wav"},
            {"mix": torch.randn(9000), "clean": torch.randn(2, 9000), "filename": "file3.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # Check that all sequences padded to max length (10000)
        assert result["mix"].shape == (3, 10000)
        assert result["clean"].shape == (3, 2, 10000)
        assert len(result["lengths"]) == 3
        assert len(result["filenames"]) == 3

    def test_libri2mix_collate_lengths_tracking(self):
        """Test that collate function tracks original lengths."""
        batch = [
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "file1.wav"},
            {"mix": torch.randn(12000), "clean": torch.randn(2, 12000), "filename": "file2.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # Verify lengths are tracked correctly
        assert result["lengths"][0] == 8000
        assert result["lengths"][1] == 12000

    def test_libri2mix_collate_preserves_filenames(self):
        """Test that collate function preserves filenames."""
        batch = [
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "test1.wav"},
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "test2.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        assert result["filenames"] == ["test1.wav", "test2.wav"]

    def test_libri2mix_collate_batch_stacking(self):
        """Test that collate function properly stacks batch."""
        batch = [
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "file1.wav"},
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "file2.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # Verify proper batching
        assert isinstance(result["mix"], torch.Tensor)
        assert isinstance(result["clean"], torch.Tensor)
        assert result["mix"].dim() == 2  # [B, T]
        assert result["clean"].dim() == 3  # [B, 2, T]
        assert result["mix"].shape[0] == 2  # Batch size

    def test_libri2mix_collate_zero_padding(self):
        """Test that padding uses zeros."""
        batch = [
            {"mix": torch.ones(5000), "clean": torch.ones(2, 5000), "filename": "file1.wav"},
            {"mix": torch.ones(8000), "clean": torch.ones(2, 8000), "filename": "file2.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # First sample should have zeros in padded region
        padded_region = result["mix"][0, 5000:]
        assert torch.all(padded_region == 0), "Mix padded region should be zeros"
        padded_region_clean = result["clean"][0, :, 5000:]
        assert torch.all(padded_region_clean == 0), "Clean padded region should be zeros"


class TestLibri2MixCrossDatasetCompatibility:
    """Tests for cross-dataset evaluation compatibility."""

    def test_libri2mix_interface_matches_polsess(self):
        """Test that Libri2Mix returns same interface as PolSESS dataset."""
        # Both should return dict with "mix" and "clean" keys
        batch = [
            {"mix": torch.randn(8000), "clean": torch.randn(2, 8000), "filename": "file.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # Same keys as PolSESS collate output
        required_keys = {"mix", "clean"}
        assert required_keys.issubset(set(result.keys()))

    def test_libri2mix_tensor_format_compatibility(self):
        """Test that tensors are in compatible format for models."""
        batch = [
            {"mix": torch.randn(16000), "clean": torch.randn(2, 16000), "filename": "file.wav"},
        ]

        result = libri2mix_collate_fn(batch)

        # mix: [B, T], clean: [B, 2, T]
        assert result["mix"].dim() == 2
        assert result["clean"].dim() == 3
        assert result["mix"].shape[0] == 1  # Batch dimension
        assert result["clean"].shape == (1, 2, 16000)
