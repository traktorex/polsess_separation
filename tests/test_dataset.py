"""Tests for PolSESSDataset."""

import os
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


_POLSESS_ROOT = os.getenv(
    "POLSESS_DATA_ROOT",
    "/home/user/datasets/PolSESS_C_both/PolSESS_C_both",
)


@pytest.mark.skipif(
    not Path(_POLSESS_ROOT).exists(),
    reason="PolSESS data not available (set POLSESS_DATA_ROOT env var)",
)
def test_dataset_loading_with_data():
    """Test dataset can load samples (requires actual data)."""
    dataset = PolSESSDataset(
        data_root=_POLSESS_ROOT,
        subset="train",
        task="ES",
    )

    assert len(dataset) > 0, "Dataset should have samples"


# test_dataset_max_samples_limit was a permanent `pytest.skip("Requires actual
# dataset for testing")` stub (E4 item 10) — deleted rather than implemented:
# max_samples limiting is already fully exercised with mocked pandas data in
# tests/test_dataset_variants.py::TestMaxSamples
# (test_max_samples_limits_dataset_size / test_max_samples_none_uses_all_samples),
# so implementing it here would only duplicate that coverage.
