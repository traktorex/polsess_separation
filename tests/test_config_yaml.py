"""Tests for YAML configuration loading and priority system."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
from config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config_from_yaml,
    save_config_to_yaml,
    get_config_from_args
)


class TestYAMLLoading:
    """Test YAML configuration file loading."""

    def test_load_config_from_yaml_basic(self):
        """Test loading a simple YAML config."""
        yaml_content = """
data:
  batch_size: 8
  task: EB

model:
  convtasnet:
    N: 512
    B: 512

training:
  lr: 0.0001
  num_epochs: 50
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            config = load_config_from_yaml(str(yaml_path))

            assert config.data.batch_size == 8
            assert config.data.task == 'EB'
            assert config.model.convtasnet.N == 512
            assert config.model.convtasnet.B == 512
            assert config.training.lr == 0.0001
            assert config.training.num_epochs == 50

    def test_load_config_partial_yaml(self):
        """Test that partial YAML uses defaults for missing values."""
        yaml_content = """
data:
  batch_size: 16

training:
  lr: 0.001
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            config = load_config_from_yaml(str(yaml_path))

            # Specified values
            assert config.data.batch_size == 16
            assert config.training.lr == 0.001

            # Default values should be used
            assert config.data.task == 'ES'  # default
            assert config.model.convtasnet.N == 256  # default

    def test_load_config_empty_sections(self):
        """Test loading YAML with empty sections."""
        yaml_content = """
data:
  batch_size: 4

model:

training:
  lr: 0.002
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            config = load_config_from_yaml(str(yaml_path))

            assert config.data.batch_size == 4
            assert config.training.lr == 0.002
            # Model section should use all defaults
            assert config.model.convtasnet.N == 256

    def test_load_config_invalid_yaml(self):
        """Test that invalid YAML raises appropriate error."""
        # Invalid YAML with improper indentation
        yaml_content = """
data:
batch_size: 8
  task: ES
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            with pytest.raises(yaml.YAMLError):
                load_config_from_yaml(str(yaml_path))


class TestYAMLSaving:
    """Test YAML configuration file saving."""

    def test_save_config_to_yaml_creates_file(self):
        """Test that save_config_to_yaml creates a valid YAML file."""
        from config import ConvTasNetParams
        config = Config(
            data=DataConfig(batch_size=8, task='EB'),
            model=ModelConfig(convtasnet=ConvTasNetParams(N=512, B=512)),
            training=TrainingConfig(lr=0.0001, num_epochs=50)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_config.yaml"

            save_config_to_yaml(config, str(temp_path))

            # Verify file was created
            assert temp_path.exists()

            # Load and verify contents
            loaded_yaml = yaml.safe_load(temp_path.read_text())

            assert loaded_yaml['data']['batch_size'] == 8
            assert loaded_yaml['data']['task'] == 'EB'
            assert loaded_yaml['model']['convtasnet']['N'] == 512
            assert loaded_yaml['training']['lr'] == 0.0001

    def test_save_load_roundtrip(self):
        """Test that saving and loading config preserves all values."""
        from config import ConvTasNetParams
        original_config = Config(
            data=DataConfig(
                batch_size=12,
                task='SB',
                num_workers=8
            ),
            model=ModelConfig(
                convtasnet=ConvTasNetParams(
                    N=128,
                    B=128,
                    H=256,
                    P=3,
                    X=8,
                    R=4,
                    C=2
                )
            ),
            training=TrainingConfig(
                lr=0.0005,
                num_epochs=100,
                weight_decay=0.0002,
                grad_clip_norm=10.0
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_config.yaml"

            # Save
            save_config_to_yaml(original_config, str(temp_path))

            # Load
            loaded_config = load_config_from_yaml(str(temp_path))

            # Verify all values match
            assert loaded_config.data.batch_size == original_config.data.batch_size
            assert loaded_config.data.task == original_config.data.task
            assert loaded_config.data.num_workers == original_config.data.num_workers

            assert loaded_config.model.convtasnet.N == original_config.model.convtasnet.N
            assert loaded_config.model.convtasnet.B == original_config.model.convtasnet.B
            assert loaded_config.model.convtasnet.H == original_config.model.convtasnet.H
            assert loaded_config.model.convtasnet.C == original_config.model.convtasnet.C

            assert loaded_config.training.lr == original_config.training.lr
            assert loaded_config.training.num_epochs == original_config.training.num_epochs
            assert loaded_config.training.weight_decay == original_config.training.weight_decay


class TestConfigPrioritySystem:
    """Test configuration priority system (YAML < CLI args)."""

    def test_cli_args_override_yaml(self):
        """Test that CLI arguments override YAML config values."""
        yaml_content = """
data:
  dataset_type: polsess
  batch_size: 4
  task: ES

training:
  lr: 0.001
  num_epochs: 10
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            # Simulate CLI args that override YAML (only supported quick-switch args)
            test_args = [
                '--config', str(yaml_path),
                '--task', 'EB',           # Override YAML value of ES
                '--dataset-type', 'libri2mix'  # Override YAML value of polsess
            ]

            with patch('sys.argv', ['train.py'] + test_args):
                config = get_config_from_args()

            # CLI values should take priority
            assert config.data.task == 'EB'       # From CLI, not ES from YAML
            assert config.data.dataset_type == 'libri2mix'  # From CLI, not polsess

            # Non-overridden values should come from YAML
            assert config.data.batch_size == 4  # From YAML
            assert config.training.num_epochs == 10  # From YAML

    def test_yaml_overrides_defaults(self):
        """Test that YAML values override hardcoded defaults."""
        yaml_content = """
data:
  batch_size: 32

model:
  convtasnet:
    N: 128

training:
  lr: 0.0002
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)

            test_args = ['--config', str(yaml_path)]

            with patch('sys.argv', ['train.py'] + test_args):
                config = get_config_from_args()

            # YAML values should override defaults
            assert config.data.batch_size == 32  # YAML overrides default 4
            assert config.model.convtasnet.N == 128         # YAML overrides default 256
            assert config.training.lr == 0.0002  # YAML overrides default 0.001

    def test_defaults_used_when_no_yaml_or_cli(self):
        """Test that defaults are used when neither YAML nor CLI args are provided."""
        test_args = []  # No config file, no overrides

        with patch('sys.argv', ['train.py'] + test_args):
            config = get_config_from_args()

        # Should use hardcoded defaults
        assert config.data.batch_size == 4  # default
        assert config.data.task == 'ES'     # default
        assert config.model.convtasnet.N == 256        # default
        assert config.training.lr == 0.001  # default


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_task_raises_error(self):
        """Test that invalid task value raises validation error."""
        with pytest.raises(ValueError, match="Invalid task"):
            Config(
                data=DataConfig(task='INVALID'),
                model=ModelConfig(),
                training=TrainingConfig()
            )

    def test_valid_tasks_accepted(self):
        """Test that all valid task values are accepted."""
        for task in ['ES', 'EB', 'SB']:
            config = Config(
                data=DataConfig(task=task),
                model=ModelConfig(),
                training=TrainingConfig()
            )
            assert config.data.task == task

    def test_model_c_validation_for_task(self):
        """Test that model C value is automatically adjusted based on task."""
        from config import ConvTasNetParams
        # ES and EB tasks should have C=1
        config_es = Config(
            data=DataConfig(task='ES'),
            model=ModelConfig(convtasnet=ConvTasNetParams(C=1)),
            training=TrainingConfig()
        )
        assert config_es.model.convtasnet.C == 1

        # SB task should have C=2
        config_sb = Config(
            data=DataConfig(task='SB'),
            model=ModelConfig(convtasnet=ConvTasNetParams(C=2)),
            training=TrainingConfig()
        )
        assert config_sb.model.convtasnet.C == 2

    def test_model_c_auto_correction(self):
        """Test that Config automatically corrects model.C based on task."""
        from config import ConvTasNetParams
        # SB task with C=1 should be auto-corrected to C=2
        config_sb = Config(
            data=DataConfig(task='SB'),
            model=ModelConfig(convtasnet=ConvTasNetParams(C=1)),
            training=TrainingConfig()
        )
        assert config_sb.model.convtasnet.C == 2  # Auto-corrected

        # ES task with C=2 should be auto-corrected to C=1
        config_es = Config(
            data=DataConfig(task='ES'),
            model=ModelConfig(convtasnet=ConvTasNetParams(C=2)),
            training=TrainingConfig()
        )
        assert config_es.model.convtasnet.C == 1  # Auto-corrected


class TestLoadConfigForRun:
    """Test load_config_for_run with sweep config overrides."""

    @pytest.fixture
    def base_yaml(self, tmp_path):
        """Create a base YAML config file for testing."""
        yaml_content = """
data:
  batch_size: 4
  task: ES

model:
  model_type: convtasnet
  convtasnet:
    N: 256
    B: 256
    H: 512
  dprnn:
    N: 64
    hidden_size: 128

training:
  lr: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 5.0
  lr_factor: 0.95
  lr_patience: 2
  num_epochs: 10
  seed: 42
  use_amp: true
"""
        yaml_path = tmp_path / "base.yaml"
        yaml_path.write_text(yaml_content)
        return str(yaml_path)

    def _make_sweep_config(self, base_yaml, **overrides):
        """Create a mock matching wandb.config behavior (supports 'in' and attr access)."""
        class MockSweepConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def __contains__(self, key):
                return key in self.__dict__
            def get(self, key, default=None):
                return self.__dict__.get(key, default)
        return MockSweepConfig(config=base_yaml, **overrides)

    def test_training_hp_overrides(self, base_yaml):
        """Test that training hyperparameters are correctly overridden."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(
            base_yaml,
            lr=0.0005,
            weight_decay=1e-5,
            grad_clip_norm=10.0,
            lr_factor=0.5,
            lr_patience=5,
        )
        config = load_config_for_run(sweep)

        assert config.training.lr == 0.0005
        assert config.training.weight_decay == 1e-5
        assert config.training.grad_clip_norm == 10.0
        assert config.training.lr_factor == 0.5
        assert config.training.lr_patience == 5

    def test_data_overrides(self, base_yaml):
        """Test that data config overrides work."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, task="SB", batch_size=8)
        config = load_config_for_run(sweep)

        assert config.data.task == "SB"
        assert config.data.batch_size == 8

    def test_model_type_override(self, base_yaml):
        """Test that model_type override works."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, model_type="dprnn")
        config = load_config_for_run(sweep)

        assert config.model.model_type == "dprnn"

    def test_epochs_alias(self, base_yaml):
        """Test that 'epochs' works as alias for 'num_epochs'."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, epochs=50)
        config = load_config_for_run(sweep)

        assert config.training.num_epochs == 50

    def test_num_epochs_override(self, base_yaml):
        """Test that 'num_epochs' direct override works."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, num_epochs=30)
        config = load_config_for_run(sweep)

        assert config.training.num_epochs == 30

    def test_convtasnet_architecture_overrides(self, base_yaml):
        """Test special-case ConvTasNet model_B and model_H overrides."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, model_B=128, model_H=256)
        config = load_config_for_run(sweep)

        assert config.model.convtasnet.B == 128
        assert config.model.convtasnet.H == 256

    def test_partial_override_preserves_defaults(self, base_yaml):
        """Test that unspecified keys remain at YAML defaults."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, lr=0.01)
        config = load_config_for_run(sweep)

        assert config.training.lr == 0.01
        # These should stay at YAML values, not dataclass defaults
        assert config.training.weight_decay == 0.0001
        assert config.training.grad_clip_norm == 5.0
        assert config.training.lr_factor == 0.95
        assert config.data.batch_size == 4

    def test_seed_and_amp_overrides(self, base_yaml):
        """Test hardware/reproducibility overrides."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, seed=123, use_amp=False)
        config = load_config_for_run(sweep)

        assert config.training.seed == 123
        assert config.training.use_amp is False

    def test_early_stopping_override(self, base_yaml):
        """Test early stopping patience override."""
        from config import load_config_for_run
        sweep = self._make_sweep_config(base_yaml, early_stopping_patience=10)
        config = load_config_for_run(sweep)

        assert config.training.early_stopping_patience == 10
