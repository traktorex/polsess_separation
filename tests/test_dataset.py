"""Tests for PolSESSDataset."""

import pytest
from pathlib import Path
from datasets import PolSESSDataset


def test_dataset_variant_constants():
    """Test dataset has correct variant constants."""
    assert PolSESSDataset.INDOOR_VARIANTS == ["SER", "SR", "ER", "R", "C"]
    assert PolSESSDataset.OUTDOOR_VARIANTS == ["SE", "S", "E", "C"]


def test_dataset_all_variants():
    """Test that all variants are accounted for."""
    all_variants = PolSESSDataset.INDOOR_VARIANTS + PolSESSDataset.OUTDOOR_VARIANTS
    expected = ["SER", "SR", "ER", "R", "C", "SE", "S", "E", "C"]
    assert sorted(all_variants) == sorted(expected)


@pytest.mark.skipif(
    not Path("F:\\PolSMSE\\EksperymentyMOWA\\BAZY\\MOWA\\PolSESS_C_in\\PolSESS_C_in").exists(),
    reason="PolSESS data not available"
)
def test_dataset_loading_with_data():
    """Test dataset can load samples (requires actual data)."""
    data_root = "F:\\PolSMSE\\EksperymentyMOWA\\BAZY\\MOWA\\PolSESS_C_in\\PolSESS_C_in"

    dataset = PolSESSDataset(
        data_root=data_root,
        subset="train",
        task="ES",
    )

    assert len(dataset) > 0, "Dataset should have samples"


def test_dataset_max_samples_limit(tmp_path):
    """Test max_samples parameter limits dataset size."""
    # This test would require mock data or actual data
    # Skipping if data not available
    pytest.skip("Requires actual dataset for testing")
