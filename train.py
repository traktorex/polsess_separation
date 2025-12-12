"""Training script for speech separation using various model architectures."""

import torch
from torch.utils.data import DataLoader
from datasets import get_dataset, polsess_collate_fn
from config import get_config_from_args
from models import get_model
from training.trainer import Trainer
from utils import (
    set_seed,
    setup_warnings,
    setup_logger,
    setup_device_and_amp,
    WandbLogger,
)
import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def create_dataloaders(config, summary_info=None):
    """Create train and val dataloaders using registry."""
    dataset_class = get_dataset(config.data.dataset_type)

    # Get dataset-specific parameters
    if config.data.dataset_type == "polsess":
        data_root = config.data.polsess.data_root
    else:
        raise ValueError(
            f"Dataset {config.data.dataset_type} not yet configured with nested params"
        )

    # Determine initial training variants (curriculum learning or None)
    train_variants = None
    if config.training.curriculum_learning:
        # Use first epoch's variants
        train_variants = config.training.curriculum_learning[0].get("variants")

    train_dataset = dataset_class(
        data_root,
        subset="train",
        task=config.data.task,
        max_samples=config.data.train_max_samples,
        allowed_variants=train_variants,
    )

    # Use validation_variants from config, default to None (all variants)
    val_variants = config.training.validation_variants

    val_dataset = dataset_class(
        data_root,
        subset="val",
        task=config.data.task,
        allowed_variants=val_variants,
        max_samples=config.data.val_max_samples,
    )

    if summary_info is not None:
        summary_info["train_samples"] = len(train_dataset)
        summary_info["val_samples"] = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        prefetch_factor=(
            config.data.prefetch_factor if config.data.num_workers > 0 else None
        ),
        collate_fn=polsess_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=(
            config.data.prefetch_factor if config.data.num_workers > 0 else None
        ),
        collate_fn=polsess_collate_fn,
    )

    return train_loader, val_loader


def create_model(config, summary_info=None):
    """Create model using registry."""
    model_class = get_model(config.model.model_type)

    # Get model-specific parameters
    if config.model.model_type == "convtasnet":
        params = config.model.convtasnet
        model = model_class(
            N=params.N,
            B=params.B,
            H=params.H,
            P=params.P,
            X=params.X,
            R=params.R,
            C=params.C,
            norm_type=params.norm_type,
            causal=params.causal,
            mask_nonlinear=params.mask_nonlinear,
            kernel_size=params.kernel_size,
            stride=params.stride,
        )
    elif config.model.model_type == "sepformer":
        params = config.model.sepformer
        model = model_class(
            N=params.N,
            kernel_size=params.kernel_size,
            stride=params.stride,
            C=params.C,
            causal=params.causal,
            num_blocks=params.num_blocks,
            num_layers=params.num_layers,
            d_model=params.d_model,
            nhead=params.nhead,
            d_ffn=params.d_ffn,
            dropout=params.dropout,
            chunk_size=params.chunk_size,
            hop_size=params.hop_size,
        )
    elif config.model.model_type == "dprnn":
        params = config.model.dprnn
        model = model_class(
            N=params.N,
            kernel_size=params.kernel_size,
            stride=params.stride,
            C=params.C,
            num_layers=params.num_layers,
            chunk_size=params.chunk_size,
            rnn_type=params.rnn_type,
            hidden_size=params.hidden_size,
            num_rnn_layers=params.num_rnn_layers,
            dropout=params.dropout,
            bidirectional=params.bidirectional,
            norm_type=params.norm_type,
        )
    elif config.model.model_type == "spmamba":
        params = config.model.spmamba
        model = model_class(
            input_dim=params.input_dim,
            n_srcs=params.n_srcs,
            n_fft=params.n_fft,
            stride=params.stride,
            window=params.window,
            n_layers=params.n_layers,
            lstm_hidden_units=params.lstm_hidden_units,
            attn_n_head=params.attn_n_head,
            attn_approx_qk_dim=params.attn_approx_qk_dim,
            emb_dim=params.emb_dim,
            emb_ks=params.emb_ks,
            emb_hs=params.emb_hs,
            activation=params.activation,
            eps=params.eps,
            sample_rate=params.sample_rate,
        )
    else:
        raise ValueError(
            f"Model {config.model.model_type} not yet configured with nested params"
        )

    if summary_info is not None:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        summary_info["model_params_millions"] = num_params
        summary_info["model_type"] = config.model.model_type

    return model


def main():
    setup_warnings()
    config = get_config_from_args()
    set_seed(config.training.seed)

    # Setup logger
    logger = setup_logger(
        name="polsess",
        log_level=config.training.log_level,
        log_file=config.training.log_file,
    )

    # Setup device and AMP
    summary_info = {"seed": config.training.seed}
    device = setup_device_and_amp(config, summary_info)

    # Create dataloaders and model using helper functions
    train_loader, val_loader = create_dataloaders(config, summary_info)
    model = create_model(config, summary_info)

    # Use torch.compile for speedup (PyTorch 2.0+, Linux only)
    if hasattr(torch, "compile") and sys.platform == "linux":
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="default")
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    elif hasattr(torch, "compile") and sys.platform != "linux":
        logger.info(
            "Skipping torch.compile (requires Triton/Linux, detected: %s)", sys.platform
        )

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        run_name=config.training.wandb_run_name,
        config=config,
        enabled=config.training.use_wandb,
        logger=logger,
    )

    # Log configuration
    logger.info("\n" + config.summary())
    logger.info(f"Dataset: {config.data.dataset_type}")
    logger.info(f"Train samples: {summary_info['train_samples']}")
    logger.info(f"Val samples: {summary_info['val_samples']}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Model parameters: {summary_info['model_params_millions']:.2f}M")

    # Create trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config,
        device=device,
        logger=logger,
        wandb_logger=wandb_logger,
    )

    # Resume from checkpoint if specified
    if config.training.resume_from:
        trainer.load_checkpoint(config.training.resume_from)

    # Start training
    trainer.train(
        num_epochs=config.training.num_epochs, save_dir=config.training.save_dir
    )

    # Log completion
    logger.info("Training complete!")
    logger.info(f"Best validation SI-SDR: {trainer.best_val_sisdr:.2f} dB")

    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
