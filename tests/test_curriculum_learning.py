"""Tests for curriculum learning functionality."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
from pathlib import Path


def test_curriculum_variants_update():
    """Test that training variants are updated correctly based on curriculum schedule."""
    from training.trainer import Trainer

    # Create minimal config with curriculum learning
    config = SimpleNamespace()
    config.data = SimpleNamespace(task="ES", batch_size=2)
    config.training = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=5.0,
        lr_factor=0.5,
        lr_patience=2,
        use_amp=False,
        validation_variants=["SER"],
        curriculum_learning=[
            {"epoch": 1, "variants": ["C"]},
            {"epoch": 2, "variants": ["C", "R"]},
            {"epoch": 5, "variants": ["C", "R", "S", "SR"], "lr_scheduler": "start"},
        ],
    )

    # Create mock model and loaders
    import torch.nn as nn

    model = nn.Linear(10, 5)  # Simple model with actual parameters

    # Create mock dataset with allowed_variants attribute
    mock_train_dataset = Mock()
    mock_train_dataset.allowed_variants = None

    mock_train_loader = Mock()
    mock_train_loader.dataset = mock_train_dataset
    mock_train_loader.batch_size = 2

    mock_val_loader = Mock()
    mock_logger = Mock()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device="cpu",
        logger=mock_logger,
        wandb_logger=None,
    )

    # Test: Epoch 1 - should use ["C"]
    trainer._update_training_variants(1)
    assert mock_train_loader.dataset.allowed_variants == ["C"]
    assert not trainer.lr_scheduler_enabled

    # Test: Epoch 3 - should use epoch 2's variants (most recent)
    trainer._update_training_variants(3)
    assert mock_train_loader.dataset.allowed_variants == ["C", "R"]
    assert not trainer.lr_scheduler_enabled

    # Test: Epoch 5 - should enable LR scheduler
    trainer._update_training_variants(5)
    assert mock_train_loader.dataset.allowed_variants == ["C", "R", "S", "SR"]
    assert trainer.lr_scheduler_enabled


def test_no_curriculum_learning():
    """Test that without curriculum learning, LR scheduler is enabled from start."""
    from training.trainer import Trainer

    # Create config WITHOUT curriculum learning
    config = SimpleNamespace()
    config.data = SimpleNamespace(task="ES", batch_size=2)
    config.training = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=5.0,
        lr_factor=0.5,
        lr_patience=2,
        use_amp=False,
        validation_variants=None,
        curriculum_learning=None,  # No curriculum
    )

    import torch.nn as nn

    model = nn.Linear(10, 5)  # Simple model with actual parameters

    mock_train_loader = Mock()
    mock_train_loader.batch_size = 2
    mock_val_loader = Mock()
    mock_logger = Mock()

    trainer = Trainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device="cpu",
        logger=mock_logger,
        wandb_logger=None,
    )

    # LR scheduler should be enabled from the start
    assert trainer.lr_scheduler_enabled


def test_validation_deterministic_variant_selection():
    """Test that validation dataset produces deterministic variant selection."""
    from datasets.polsess_dataset import PolSESSDataset
    from unittest.mock import patch, MagicMock
    import pandas as pd
    import numpy as np

    # Create mock CSV data with required columns
    mock_metadata = pd.DataFrame({
        "mixFile": ["mix1.wav", "mix2.wav"],
        "speaker1File": ["sp1_1.wav", "sp1_2.wav"],
        "speaker2File": ["sp2_1.wav", "sp2_2.wav"],
        "sceneFile": ["scene1.wav", "scene2.wav"],
        "eventFile": ["event1.wav", "event2.wav"],
        "reverbForSpeaker1": ["reverb1.wav", np.nan],  # First has reverb, second doesn't
        "reverbForSpeaker2": ["reverb2.wav", np.nan],
        "reverbForEvent": ["reverb_ev.wav", np.nan],
    })

    # Mock the CSV loading and file existence checks
    with patch("pandas.read_csv", return_value=mock_metadata):
        with patch.object(Path, "exists", return_value=True):
            # Create validation dataset with allowed variants
            val_dataset = PolSESSDataset(
                data_root="fake/path",
                subset="val",
                task="ES",
                allowed_variants=["SER", "SR", "ER"],
            )

    # Test 1: Validation subset returns same variant for same index
    # Call _choose_variant multiple times with same index
    variants_for_idx_0 = [
        val_dataset._choose_variant(has_reverb=True, idx=0) for _ in range(10)
    ]
    # All should be identical
    assert len(set(variants_for_idx_0)) == 1, "Validation should return same variant for same index"

    variants_for_idx_1 = [
        val_dataset._choose_variant(has_reverb=True, idx=1) for _ in range(10)
    ]
    assert len(set(variants_for_idx_1)) == 1, "Validation should return same variant for same index"

    # Test 2: Different indices can return different variants (but consistently)
    # This is OK - different samples can have different variants, they just need to be consistent
    all_variants = list(set(variants_for_idx_0 + variants_for_idx_1))
    assert all(v in ["SER", "SR", "ER"] for v in all_variants), "Should only return allowed variants"


def test_training_nondeterministic_variant_selection():
    """Test that training dataset uses random (non-deterministic) variant selection."""
    from datasets.polsess_dataset import PolSESSDataset
    from unittest.mock import patch
    import pandas as pd
    import numpy as np

    # Create mock CSV data with required columns
    mock_metadata = pd.DataFrame({
        "mixFile": ["mix1.wav", "mix2.wav"],
        "speaker1File": ["sp1_1.wav", "sp1_2.wav"],
        "speaker2File": ["sp2_1.wav", "sp2_2.wav"],
        "sceneFile": ["scene1.wav", "scene2.wav"],
        "eventFile": ["event1.wav", "event2.wav"],
        "reverbForSpeaker1": ["reverb1.wav", np.nan],
        "reverbForSpeaker2": ["reverb2.wav", np.nan],
        "reverbForEvent": ["reverb_ev.wav", np.nan],
    })

    # Mock the CSV loading and file existence checks
    with patch("pandas.read_csv", return_value=mock_metadata):
        with patch.object(Path, "exists", return_value=True):
            # Create training dataset with allowed variants
            train_dataset = PolSESSDataset(
                data_root="fake/path",
                subset="train",
                task="ES",
                allowed_variants=["SER", "SR", "ER", "R"],
            )

    # For training, calling _choose_variant multiple times with same index
    # should eventually return different variants (non-deterministic)
    # We'll call it 50 times to be statistically confident
    variants_for_idx_0 = [
        train_dataset._choose_variant(has_reverb=True, idx=0) for _ in range(50)
    ]

    # Should have more than one unique variant (proves non-determinism)
    unique_variants = set(variants_for_idx_0)
    assert len(unique_variants) > 1, "Training should return different variants across calls"
    assert all(v in ["SER", "SR", "ER", "R"] for v in unique_variants), "Should only return allowed variants"


def test_variant_selection_respects_reverb_compatibility():
    """Test that variant selection respects indoor/outdoor reverb compatibility."""
    from datasets.polsess_dataset import PolSESSDataset
    from unittest.mock import patch
    import pandas as pd
    import numpy as np

    # Create mock CSV data with required columns
    mock_metadata = pd.DataFrame({
        "mixFile": ["mix1.wav"],
        "speaker1File": ["sp1_1.wav"],
        "speaker2File": ["sp2_1.wav"],
        "sceneFile": ["scene1.wav"],
        "eventFile": ["event1.wav"],
        "reverbForSpeaker1": [np.nan],  # No reverb for outdoor test
        "reverbForSpeaker2": [np.nan],
        "reverbForEvent": [np.nan],
    })

    # Mock the CSV loading and file existence checks
    with patch("pandas.read_csv", return_value=mock_metadata):
        with patch.object(Path, "exists", return_value=True):
            # Create validation dataset with OUTDOOR variants only
            val_dataset = PolSESSDataset(
                data_root="fake/path",
                subset="val",
                task="ES",
                allowed_variants=["SE", "S", "E"],  # Outdoor only
            )

    # has_reverb=True with outdoor variants should raise an error
    with pytest.raises(ValueError, match="No compatible variant"):
        val_dataset._choose_variant(has_reverb=True, idx=0)

    # has_reverb=False with outdoor variants should work
    variant = val_dataset._choose_variant(has_reverb=False, idx=0)
    assert variant in ["SE", "S", "E"]


def test_validation_consistency_across_different_allowed_variants():
    """Test that validation determinism works with different allowed_variants configurations."""
    from datasets.polsess_dataset import PolSESSDataset
    from unittest.mock import patch
    import pandas as pd
    import numpy as np

    # Create mock CSV data with required columns
    mock_metadata = pd.DataFrame({
        "mixFile": ["mix1.wav"],
        "speaker1File": ["sp1_1.wav"],
        "speaker2File": ["sp2_1.wav"],
        "sceneFile": ["scene1.wav"],
        "eventFile": ["event1.wav"],
        "reverbForSpeaker1": ["reverb1.wav"],  # Has reverb
        "reverbForSpeaker2": ["reverb2.wav"],
        "reverbForEvent": ["reverb_ev.wav"],
    })

    # Mock the CSV loading and file existence checks
    with patch("pandas.read_csv", return_value=mock_metadata):
        with patch.object(Path, "exists", return_value=True):
            # Test with single variant
            val_dataset_single = PolSESSDataset(
                data_root="fake/path",
                subset="val",
                task="ES",
                allowed_variants=["SER"],
            )

            # Test with multiple variants
            val_dataset_multi = PolSESSDataset(
                data_root="fake/path",
                subset="val",
                task="ES",
                allowed_variants=["SER", "SR", "ER", "R", "C"],
            )

            # Test with no restriction (all variants)
            val_dataset_all = PolSESSDataset(
                data_root="fake/path",
                subset="val",
                task="ES",
                allowed_variants=None,
            )

    # Single variant should always return that variant
    variants_single = [val_dataset_single._choose_variant(has_reverb=True, idx=0) for _ in range(5)]
    assert all(v == "SER" for v in variants_single)

    # Multiple variants should be deterministic for same index
    variants_multi = [val_dataset_multi._choose_variant(has_reverb=True, idx=0) for _ in range(10)]
    assert len(set(variants_multi)) == 1, "Should return same variant consistently"

    # All variants (None) should be deterministic for same index
    variants_all = [val_dataset_all._choose_variant(has_reverb=True, idx=0) for _ in range(10)]
    assert len(set(variants_all)) == 1, "Should return same variant consistently"


def test_get_curriculum_variants():
    """Test _get_curriculum_variants method."""
    from training.trainer import Trainer

    config = SimpleNamespace()
    config.data = SimpleNamespace(task="ES", batch_size=2)
    config.training = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=5.0,
        lr_factor=0.5,
        lr_patience=2,
        use_amp=False,
        validation_variants=["SER"],
        curriculum_learning=[
            {"epoch": 1, "variants": ["C"]},
            {"epoch": 3, "variants": ["C", "R", "S"]},
            {"epoch": 5, "variants": ["C", "R", "S", "SR", "E"]},
        ],
    )

    import torch.nn as nn

    model = nn.Linear(10, 5)  # Simple model with actual parameters

    mock_train_loader = Mock()
    mock_train_loader.batch_size = 2
    mock_val_loader = Mock()

    trainer = Trainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device="cpu",
        logger=Mock(),
        wandb_logger=None,
    )

    # Test getting variants for different epochs
    assert trainer._get_curriculum_variants(1) == ["C"]
    assert trainer._get_curriculum_variants(2) == ["C"]  # Uses most recent (epoch 1)
    assert trainer._get_curriculum_variants(3) == ["C", "R", "S"]
    assert trainer._get_curriculum_variants(4) == ["C", "R", "S"]  # Uses epoch 3
    assert trainer._get_curriculum_variants(5) == ["C", "R", "S", "SR", "E"]
    assert trainer._get_curriculum_variants(10) == ["C", "R", "S", "SR", "E"]  # Uses epoch 5
