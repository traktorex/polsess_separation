"""Configuration management for PolSESS speech enhancement training."""

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from utils import ensure_dir


@dataclass
class PolSESSParams:
    """PolSESS dataset-specific parameters."""

    data_root: str = field(
        default_factory=lambda: os.getenv(
            "POLSESS_DATA_ROOT",
            "/home/user/datasets/PolSESS_C_both/PolSESS_C_both",
        )
    )


@dataclass
class ConvTasNetParams:
    """ConvTasNet model-specific parameters."""

    N: int = 256  # Encoder/decoder channels
    kernel_size: int = 16  # Encoder kernel size
    stride: int = 8  # Encoder stride
    B: int = 256  # Mask network: bottleneck channels
    H: int = 512  # Mask network: convolutional block channels
    P: int = 3  # Mask network: kernel size
    X: int = 8  # Mask network: blocks per repeat
    R: int = 4  # Mask network: number of repeats
    C: int = 1  # Output sources
    norm_type: str = "gLN"
    causal: bool = False
    mask_nonlinear: str = "relu"


@dataclass
class SepFormerParams:
    """SepFormer model-specific parameters."""

    N: int = 256  # Encoder/decoder channels
    kernel_size: int = 16  # Encoder kernel size
    stride: int = 8  # Encoder stride
    C: int = 2  # Output sources (number of speakers)
    causal: bool = False
    num_blocks: int = 2  # Number of times to repeat SepFormer block (IntraT→InterT)
    num_layers: int = 8  # Number of transformer layers in both IntraT and InterT
    d_model: int = 256  # Transformer dimension
    nhead: int = 8  # Number of attention heads
    d_ffn: int = 1024  # Feed-forward network dimension
    dropout: float = 0.0  # Dropout rate
    chunk_size: int = 250  # Chunk size for dual-path processing
    hop_size: int = 125  # Hop size between chunks


@dataclass
class DPRNNParams:
    """DPRNN (Dual-Path RNN) model-specific parameters."""

    N: int = 64  # Encoder/decoder channels (paper uses 64)
    kernel_size: int = 16  # Encoder kernel size
    stride: int = 8  # Encoder stride
    C: int = 1  # Output sources (1 for enhancement, 2 for separation)
    num_layers: int = 6  # Number of dual-path blocks (paper uses 6)
    chunk_size: int = 100  # Chunk length K (paper uses 100 for window=16)
    rnn_type: str = "LSTM"  # RNN type: LSTM or GRU
    hidden_size: int = 128  # Hidden units PER DIRECTION
    num_rnn_layers: int = 1  # RNN depth within each block
    dropout: float = 0.0  # Dropout probability
    bidirectional: bool = True  # Bidirectional RNNs
    norm_type: str = "ln"  # Normalization: ln, gln, cln, bn


@dataclass
class SPMambaParams:
    """SPMamba (Speech Processing Mamba) model-specific parameters."""

    input_dim: int = 64  # STFT input dimension (kept for compatibility)
    n_srcs: int = 1  # Number of output sources (1 for enhancement, 2 for separation)
    n_fft: int = 256  # FFT size (paper uses 256)
    stride: int = 64  # STFT hop length (paper uses 64)
    window: str = "hann"  # Window function
    n_layers: int = 6  # Number of GridNet blocks (paper uses 6)
    lstm_hidden_units: int = 256  # Hidden dimension (misleading name, for Mamba blocks)
    attn_n_head: int = 4  # Number of attention heads
    attn_approx_qk_dim: int = 512  # Approximate Q/K dimension for attention
    emb_dim: int = 16  # Embedding dimension
    emb_ks: int = 4  # Embedding kernel size
    emb_hs: int = 1  # Embedding hop size
    activation: str = "prelu"  # Activation function
    eps: float = 1.0e-5  # Epsilon for numerical stability
    sample_rate: int = 16000  # Audio sample rate


@dataclass
class DataConfig:
    """Common data configuration across all datasets."""

    dataset_type: str = "polsess"  # Dataset selector: polsess, libri2mix
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2
    task: str = "ES"  # ES=single speaker, EB=both speakers, SB=separate both
    train_max_samples: Optional[int] = None
    val_max_samples: Optional[int] = None
    polsess: Optional[PolSESSParams] = None

    def __post_init__(self):
        """Initialize dataset-specific params if not provided."""
        if self.dataset_type == "polsess" and self.polsess is None:
            self.polsess = PolSESSParams()


@dataclass
class ModelConfig:
    """Common model configuration across all architectures."""

    model_type: str = (
        "convtasnet"  # Model selector: convtasnet, sepformer, dprnn, spmamba
    )
    convtasnet: Optional[ConvTasNetParams] = None
    sepformer: Optional[SepFormerParams] = None
    dprnn: Optional[DPRNNParams] = None
    spmamba: Optional[SPMambaParams] = None

    def __post_init__(self):
        """Initialize model-specific params if not provided."""
        if self.model_type == "convtasnet" and self.convtasnet is None:
            self.convtasnet = ConvTasNetParams()
        elif self.model_type == "sepformer" and self.sepformer is None:
            self.sepformer = SepFormerParams()
        elif self.model_type == "dprnn" and self.dprnn is None:
            self.dprnn = DPRNNParams()
        elif self.model_type == "spmamba" and self.spmamba is None:
            self.spmamba = SPMambaParams()


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    lr_factor: float = 0.95
    lr_patience: int = 2
    num_epochs: int = 10
    use_amp: bool = True
    amp_eps: float = 1e-4
    save_dir: str = "checkpoints"
    use_wandb: bool = True
    wandb_project: str = "polsess-separation"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_file: Optional[str] = None
    log_level: str = "INFO"
    device: str = "cuda"
    seed: int = 42
    resume_from: Optional[str] = None
    validation_variants: Optional[List[str]] = None
    curriculum_learning: Optional[List[Dict[str, Any]]] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure dataset-specific params are initialized
        if self.data.dataset_type == "polsess" and self.data.polsess is None:
            self.data.polsess = PolSESSParams()

        # Ensure model-specific params are initialized
        if self.model.model_type == "convtasnet" and self.model.convtasnet is None:
            self.model.convtasnet = ConvTasNetParams()

        # Validate paths (dataset-specific)
        if self.data.dataset_type == "polsess":
            data_root = Path(self.data.polsess.data_root)
            if not data_root.exists():
                raise FileNotFoundError(
                    f"Data root does not exist: {data_root}\n"
                    f"Set via --data-root CLI arg or POLSESS_DATA_ROOT environment variable"
                )

        # Validate task
        if self.data.task not in ["ES", "EB", "SB"]:
            raise ValueError(
                f"Invalid task: {self.data.task}. Must be 'ES', 'EB' or 'SB'."
            )

        # Adjust model output sources based on task
        if self.model.model_type == "convtasnet":
            if self.data.task == "ES":
                self.model.convtasnet.C = 1
            elif self.data.task == "EB":
                self.model.convtasnet.C = 1
            elif self.data.task == "SB":
                self.model.convtasnet.C = 2
        elif self.model.model_type == "sepformer":
            if self.data.task == "ES":
                self.model.sepformer.C = 1
            elif self.data.task == "EB":
                self.model.sepformer.C = 1
            elif self.data.task == "SB":
                self.model.sepformer.C = 2
        elif self.model.model_type == "dprnn":
            if self.data.task == "ES":
                self.model.dprnn.C = 1
            elif self.data.task == "EB":
                self.model.dprnn.C = 1
            elif self.data.task == "SB":
                self.model.dprnn.C = 2
        elif self.model.model_type == "spmamba":
            if self.data.task == "ES":
                self.model.spmamba.n_srcs = 1
            elif self.data.task == "EB":
                self.model.spmamba.n_srcs = 1
            elif self.data.task == "SB":
                self.model.spmamba.n_srcs = 2

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "Configuration Summary",
            "=" * 80,
            "",
            "Data:",
            f"  Dataset: {self.data.dataset_type}",
        ]

        # Add dataset-specific info
        if self.data.dataset_type == "polsess":
            lines.extend(
                [
                    f"  Root: {self.data.polsess.data_root}",
                ]
            )

        lines.extend(
            [
                f"  Task: {self.data.task}",
                f"  Batch size: {self.data.batch_size}",
                "",
                "Model:",
                f"  Architecture: {self.model.model_type}",
            ]
        )

        # Add model-specific info
        if self.model.model_type == "convtasnet":
            lines.extend(
                [
                    f"  Parameters: N={self.model.convtasnet.N}, B={self.model.convtasnet.B}, H={self.model.convtasnet.H}",
                    f"  Temporal blocks: R={self.model.convtasnet.R}, X={self.model.convtasnet.X}",
                    f"  Output sources: C={self.model.convtasnet.C}",
                ]
            )

        lines.extend(
            [
                "",
                "Training:",
                f"  Epochs: {self.training.num_epochs}",
                f"  Learning rate: {self.training.lr}",
                f"  Weight decay: {self.training.weight_decay}",
                f"  AMP enabled: {self.training.use_amp}",
                f"  Device: {self.training.device}",
                "",
                "=" * 80,
            ]
        )
        return "\n".join(lines)


def load_config_from_yaml(yaml_path: str) -> Config:
    """Load configuration from YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    data_dict = config_dict.get("data", {}) or {}
    model_dict = config_dict.get("model", {}) or {}
    training_dict = config_dict.get("training", {}) or {}

    # Handle nested dataset-specific params
    polsess_dict = data_dict.pop("polsess", None)
    polsess_params = PolSESSParams(**polsess_dict) if polsess_dict else None

    # Handle nested model-specific params
    convtasnet_dict = model_dict.pop("convtasnet", None)
    convtasnet_params = ConvTasNetParams(**convtasnet_dict) if convtasnet_dict else None

    sepformer_dict = model_dict.pop("sepformer", None)
    sepformer_params = SepFormerParams(**sepformer_dict) if sepformer_dict else None

    dprnn_dict = model_dict.pop("dprnn", None)
    dprnn_params = DPRNNParams(**dprnn_dict) if dprnn_dict else None

    spmamba_dict = model_dict.pop("spmamba", None)
    spmamba_params = SPMambaParams(**spmamba_dict) if spmamba_dict else None

    data_config = DataConfig(**data_dict, polsess=polsess_params)
    model_config = ModelConfig(
        **model_dict,
        convtasnet=convtasnet_params,
        sepformer=sepformer_params,
        dprnn=dprnn_params,
        spmamba=spmamba_params,
    )
    training_config = TrainingConfig(**training_dict)

    return Config(data=data_config, model=model_config, training=training_config)


def save_config_to_yaml(config: Config, yaml_path: str):
    """Save configuration to YAML file."""
    # Convert to dict with nested structure
    data_dict = {}
    for key, value in asdict(config.data).items():
        if key == "polsess" and value is not None:
            data_dict["polsess"] = value
        elif key != "polsess":
            data_dict[key] = value

    model_dict = {}
    for key, value in asdict(config.model).items():
        if key == "convtasnet" and value is not None:
            model_dict["convtasnet"] = value
        elif key == "sepformer" and value is not None:
            model_dict["sepformer"] = value
        elif key == "dprnn" and value is not None:
            model_dict["dprnn"] = value
        elif key == "spmamba" and value is not None:
            model_dict["spmamba"] = value
        elif key not in ["convtasnet", "sepformer", "dprnn", "spmamba"]:
            model_dict[key] = value

    config_dict = {
        "data": data_dict,
        "model": model_dict,
        "training": asdict(config.training),
    }

    yaml_path = Path(yaml_path)
    ensure_dir(yaml_path.parent)

    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_config_parser() -> "argparse.ArgumentParser":
    """Create argument parser for config CLI.

    Returns:
        Configured ArgumentParser for training configuration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train speech separation models on various datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (primary configuration method - use this!)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (recommended)",
    )

    # Quick-switch arguments (common overrides)
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override model architecture (convtasnet, sepformer, dprnn)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Override dataset type (polsess, libri2mix)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Override data root directory"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["ES", "EB", "SB"],
        help="Override task (ES=single speaker, EB=both speakers, SB=separate both)",
    )

    # Checkpointing
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override checkpoint save directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    # Logging
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    return parser


def apply_cli_overrides(config: Config, args: "argparse.Namespace") -> Config:
    """Apply CLI argument overrides to config object.

    Args:
        config: Base configuration object.
        args: Parsed command-line arguments.

    Returns:
        Config object with CLI overrides applied.
    """
    # Apply CLI overrides (only if explicitly provided)
    if args.model_type:
        config.model.model_type = args.model_type
    if args.dataset_type:
        config.data.dataset_type = args.dataset_type
    if args.data_root:
        # Apply to dataset-specific params
        if config.data.dataset_type == "polsess":
            if config.data.polsess is None:
                config.data.polsess = PolSESSParams()
            config.data.polsess.data_root = args.data_root
    if args.task:
        config.data.task = args.task
    if args.save_dir:
        config.training.save_dir = args.save_dir
    if args.resume:
        config.training.resume_from = args.resume
    if args.no_wandb:
        config.training.use_wandb = False

    return config


def get_config_from_args() -> Config:
    """Parse command-line arguments and create configuration.

    Main entry point that orchestrates config loading from CLI.

    Note: Most parameters should be set in YAML config files.
    CLI args are for quick switches and overrides only.
    W&B sweeps use load_config_for_run() instead, not these CLI args.

    Returns:
        Validated Config object with CLI overrides applied.
    """
    parser = create_config_parser()
    args = parser.parse_args()

    # Load from YAML if provided, otherwise use defaults
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Validate and return
    config.__post_init__()
    return config


def load_config_for_run(sweep_config: Optional[dict] = None) -> Config:
    """Unified config loader for scripts and sweeps.

    If `sweep_config` is None the function parses CLI arguments via
    `get_config_from_args()` (so CLI and --config YAML are supported).

    If `sweep_config` is provided (for W&B sweeps), it will load a base
    YAML (path taken from sweep_config['config'] or a sensible default)
    and then apply common sweep overrides found in the sweep config.

    Supported sweep override keys: model_B, model_H, weight_decay,
    grad_clip_norm, batch_size, lr, epochs, device
    """
    if sweep_config is None:
        return get_config_from_args()

    # Load YAML base config path from sweep config or fallback
    base_config_path = sweep_config.get("config", "experiments/baseline.yaml")
    config = load_config_from_yaml(base_config_path)

    # Apply common overrides (if present in sweep config)
    if "model_B" in sweep_config:
        config.model.convtasnet.B = sweep_config.model_B
    if "model_H" in sweep_config:
        config.model.convtasnet.H = sweep_config.model_H
    if "weight_decay" in sweep_config:
        config.training.weight_decay = sweep_config.weight_decay
    if "grad_clip_norm" in sweep_config:
        config.training.grad_clip_norm = sweep_config.grad_clip_norm
    if "batch_size" in sweep_config:
        config.data.batch_size = sweep_config.batch_size
    if "lr" in sweep_config:
        config.training.lr = sweep_config.lr
    if "epochs" in sweep_config:
        config.training.num_epochs = sweep_config.epochs
    if "device" in sweep_config:
        config.training.device = sweep_config.device
    if "use_amp" in sweep_config:
        config.training.use_amp = sweep_config.use_amp

    # Validate and return
    config.__post_init__()
    return config
