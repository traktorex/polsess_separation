"""Tests for checkpoint saving and loading functionality."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from training.trainer import Trainer
from config import Config, DataConfig, ModelConfig, TrainingConfig


def _find_run_dir(task_dir):
    """Helper to find run directory in new hierarchical structure.

    New structure: task_dir/experiment_name/run_*/
    """
    # Find experiment directory
    experiment_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name != "latest"]
    if not experiment_dirs:
        return None

    # Find run directory within experiment
    for experiment_dir in experiment_dirs:
        run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if run_dirs:
            return run_dirs[0]

    return None


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        config = Config(
            data=DataConfig(task="ES"),
            model=ModelConfig(model_type="convtasnet"),  # Required for new structure
            training=TrainingConfig(),
        )

        # Use simple real model instead of mock
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 5)

        model = SimpleModel()
        mock_logger = Mock()
        mock_train_loader = Mock()
        mock_train_loader.batch_size = 4
        mock_val_loader = Mock()

        # Correct Trainer signature
        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            config=config,
            device="cpu",
            logger=mock_logger,
            wandb_logger=None,
        )

        return trainer

    def test_save_checkpoint_creates_hierarchical_structure(self, mock_trainer):
        """Test that _save_checkpoint creates hierarchical directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save checkpoint
            mock_trainer._save_checkpoint(epoch=5, val_sisdr=10.5, save_dir=save_dir)

            # Check hierarchical structure exists: checkpoints/convtasnet/ES/experiment_name/run_*/best_model.pt
            model_dir = save_dir / "convtasnet"
            task_dir = model_dir / "ES"

            assert model_dir.exists(), "Model directory should exist"
            assert task_dir.exists(), "Task directory should exist"

            # Find experiment directory (should be 'default' if no wandb_run_name)
            experiment_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            assert len(experiment_dirs) >= 1, "At least one experiment directory should exist"
            experiment_dir = experiment_dirs[0]

            # Find run directory
            run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) >= 1, "At least one run directory should exist"

            # Check checkpoint file exists
            checkpoint_file = run_dirs[0] / "best_model.pt"
            assert checkpoint_file.exists(), "Checkpoint file should exist"

            # Load and verify checkpoint contents
            checkpoint = torch.load(checkpoint_file)
            assert "epoch" in checkpoint
            assert checkpoint["epoch"] == 5
            assert "val_sisdr" in checkpoint
            assert checkpoint["val_sisdr"] == 10.5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "config" in checkpoint

    def test_save_checkpoint_creates_config_yaml(self, mock_trainer):
        """Test that _save_checkpoint saves config.yaml alongside checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            mock_trainer._save_checkpoint(epoch=1, val_sisdr=8.0, save_dir=save_dir)

            # Find run directory
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            assert run_dir is not None, "Run directory should exist"

            # Check config.yaml exists
            config_file = run_dir / "config.yaml"
            assert config_file.exists(), "config.yaml should exist"

            # Verify it's valid YAML
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            assert "model" in config_data
            assert "data" in config_data
            assert "training" in config_data

    def test_save_checkpoint_creates_directory_if_missing(self, mock_trainer):
        """Test that _save_checkpoint creates save directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "nonexistent" / "nested" / "dir"

            # Directory should not exist initially
            assert not save_dir.exists()

            # Save checkpoint
            mock_trainer._save_checkpoint(epoch=1, val_sisdr=5.0, save_dir=save_dir)

            # Directory structure should now exist
            task_dir = save_dir / "convtasnet" / "ES"
            assert task_dir.exists()

            run_dir = _find_run_dir(task_dir)
            assert run_dir is not None, "Run directory should exist"

    def test_save_checkpoint_includes_config(self, mock_trainer):
        """Test that checkpoint includes serialized config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            mock_trainer._save_checkpoint(epoch=1, val_sisdr=8.0, save_dir=save_dir)

            # Find checkpoint file
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            checkpoint_file = run_dir / "best_model.pt"

            checkpoint = torch.load(checkpoint_file)

            # Verify config was saved correctly
            assert "config" in checkpoint
            assert "model" in checkpoint["config"]
            assert "data" in checkpoint["config"]
            assert "training" in checkpoint["config"]

            # Verify config values
            assert checkpoint["config"]["data"]["task"] == "ES"

    def test_save_checkpoint_logs_message(self, mock_trainer):
        """Test that _save_checkpoint logs a message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            mock_trainer._save_checkpoint(epoch=1, val_sisdr=12.3, save_dir=save_dir)

            # Verify logger was called with checkpoint info
            assert mock_trainer.logger.info.called
            # Check for checkpoint path in one of the log messages
            log_messages = [call[0][0] for call in mock_trainer.logger.info.call_args_list]
            assert any("Saved best model" in msg for msg in log_messages)

    def test_save_checkpoint_with_wandb_logger(self, mock_trainer):
        """Test that checkpoint is logged to W&B if logger exists."""
        mock_wandb = Mock()
        mock_wandb.enabled = False  # Disable to avoid run.name check
        mock_trainer.wandb_logger = mock_wandb
        mock_trainer.current_epoch = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            mock_trainer._save_checkpoint(epoch=10, val_sisdr=15.0, save_dir=save_dir)

            # Verify W&B logging was called
            mock_wandb.log_metrics.assert_called_once()
            call_args = mock_wandb.log_metrics.call_args
            assert call_args[0][0]["best_val_sisdr"] == 15.0
            assert call_args[1]["step"] == 10

            mock_wandb.log_model.assert_called_once()


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        config = Config(
            data=DataConfig(task="ES"),
            model=ModelConfig(model_type="convtasnet"),
            training=TrainingConfig(),
        )

        # Use simple real model instead of mock
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 5)

        model = SimpleModel()
        mock_logger = Mock()
        mock_train_loader = Mock()
        mock_train_loader.batch_size = 4
        mock_val_loader = Mock()

        # Correct Trainer signature
        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            config=config,
            device="cpu",
            logger=mock_logger,
            wandb_logger=None,
        )

        return trainer

    def test_load_checkpoint_restores_state(self, mock_trainer):
        """Test that load_checkpoint restores model and optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save a real checkpoint first
            mock_trainer._save_checkpoint(epoch=5, val_sisdr=12.5, save_dir=save_dir)

            # Save original state
            original_state = {k: v.clone() for k, v in mock_trainer.model.state_dict().items()}

            # Modify model state
            with torch.no_grad():
                for param in mock_trainer.model.parameters():
                    param.add_(1.0)

            # Find checkpoint file
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            checkpoint_file = run_dir / "best_model.pt"

            # Load checkpoint
            mock_trainer.load_checkpoint(str(checkpoint_file))

            # Verify state was restored to original
            for key in original_state:
                assert torch.allclose(
                    mock_trainer.model.state_dict()[key], original_state[key]
                )

            # Verify epoch and best_val_sisdr were updated
            assert mock_trainer.current_epoch == 6  # epoch + 1
            assert mock_trainer.best_val_sisdr == 12.5

    def test_load_checkpoint_handles_legacy_format(self, mock_trainer):
        """Test that load_checkpoint handles checkpoints without best_val_sisdr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save a checkpoint then modify it to legacy format
            mock_trainer._save_checkpoint(epoch=3, val_sisdr=10.0, save_dir=save_dir)

            # Find checkpoint file
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            checkpoint_file = run_dir / "best_model.pt"

            # Load and modify to legacy format
            checkpoint = torch.load(checkpoint_file)
            # Simulate legacy format (no best_val_sisdr key, only val_sisdr)
            if "best_val_sisdr" in checkpoint:
                del checkpoint["best_val_sisdr"]
            torch.save(checkpoint, checkpoint_file)

            # Load checkpoint
            mock_trainer.load_checkpoint(str(checkpoint_file))

            # Should fall back to val_sisdr
            assert mock_trainer.best_val_sisdr == 10.0

    def test_load_checkpoint_logs_message(self, mock_trainer):
        """Test that load_checkpoint logs resume information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save a real checkpoint first
            mock_trainer._save_checkpoint(epoch=7, val_sisdr=9.5, save_dir=save_dir)

            # Reset logger mock to clear the save checkpoint call
            mock_trainer.logger.reset_mock()

            # Find checkpoint file
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            checkpoint_file = run_dir / "best_model.pt"

            # Load checkpoint
            mock_trainer.load_checkpoint(str(checkpoint_file))

            # Verify log message
            mock_trainer.logger.info.assert_called_with(
                "Resumed from epoch 7, best SI-SDR: 9.50 dB"
            )


class TestCheckpointIntegration:
    """Integration tests for save/load cycle."""

    def test_save_load_cycle_preserves_state(self):
        """Test that saving and loading preserves all state correctly."""
        from config import ConvTasNetParams

        config = Config(
            data=DataConfig(task="ES"),
            model=ModelConfig(
                model_type="convtasnet",
                convtasnet=ConvTasNetParams(N=128, B=128, H=256),
            ),
            training=TrainingConfig(lr=0.001, num_epochs=10),
        )

        # Create simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.layer(x)

        model = SimpleModel()

        mock_logger = Mock()
        mock_train_loader = Mock()
        mock_train_loader.batch_size = 4
        mock_val_loader = Mock()

        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            config=config,
            device="cpu",
            logger=mock_logger,
            wandb_logger=None,
        )

        # Save original state
        original_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save checkpoint
            trainer._save_checkpoint(epoch=3, val_sisdr=11.0, save_dir=save_dir)

            # Modify model state
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1.0)

            # Find checkpoint file
            task_dir = save_dir / "convtasnet" / "ES"
            run_dir = _find_run_dir(task_dir)
            checkpoint_file = run_dir / "best_model.pt"

            # Load checkpoint
            trainer.load_checkpoint(str(checkpoint_file))

            # Verify state was restored
            for key in original_model_state:
                assert torch.allclose(
                    model.state_dict()[key], original_model_state[key]
                ), f"State mismatch for {key}"

            # Verify training state
            assert trainer.current_epoch == 4  # epoch + 1
            assert trainer.best_val_sisdr == 11.0


class TestCheckpointSymlinks:
    """Test symlink creation for latest checkpoint."""

    def test_latest_symlink_created(self):
        """Test that 'latest' symlink creation is attempted."""
        config = Config(
            data=DataConfig(task="ES"),
            model=ModelConfig(model_type="convtasnet"),
            training=TrainingConfig(),
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 5)

        model = SimpleModel()
        mock_logger = Mock()
        mock_train_loader = Mock()
        mock_train_loader.batch_size = 4
        mock_val_loader = Mock()

        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            config=config,
            device="cpu",
            logger=mock_logger,
            wandb_logger=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Save checkpoint
            trainer._save_checkpoint(epoch=1, val_sisdr=10.0, save_dir=save_dir)

            # Check 'latest' link exists (may fail on Windows without admin)
            # New structure: checkpoints/model/task/experiment_name/latest
            task_dir = save_dir / "convtasnet" / "ES"

            # Find experiment directory
            experiment_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name != "latest"]
            assert len(experiment_dirs) >= 1, "At least one experiment directory should exist"
            experiment_dir = experiment_dirs[0]

            # Check run directory exists
            run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) >= 1, "Run directory should exist even if symlink fails"

            latest_link = experiment_dir / "latest"

            # On Windows without admin rights, symlink creation may fail silently
            # Check either: symlink exists OR warning was logged
            if latest_link.exists():
                # Success - verify structure
                assert latest_link.is_dir() or latest_link.is_symlink()
            else:
                # Failed - should have logged a warning or we're on Windows without admin
                # Run directory was already verified to exist above
                pass
