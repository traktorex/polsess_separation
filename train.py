"""Training script for speech separation using various model architectures."""

from utils import warning_filters  # noqa: F401  must precede speechbrain imports (registers filters)

import os

import torch

from config import get_config_from_args
from training.setup import build_dataloaders, build_trainer
from utils import (
    set_seed,
    setup_warnings,
    setup_logger,
    setup_device_and_amp,
    WandbLogger,
    collect_run_manifest,
    read_wandb_run_id,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main():
    setup_warnings()
    torch.set_float32_matmul_precision('high')
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

    # Capture run provenance (git SHA, env, GPU, argv) for checkpoints + W&B.
    manifest = collect_run_manifest(seed=config.training.seed)

    # Build dataloaders (also applies the determinism policy).
    train_loader, val_loader, per_variant_val_loaders = build_dataloaders(
        config, summary_info, logger=logger
    )

    # W&B resume: reconnect to the original run if the checkpoint carries its id
    # (survey gap 14); older checkpoints / wandb-disabled runs → fresh run.
    resume_wandb_id = (
        read_wandb_run_id(config.training.resume_from)
        if config.training.resume_from
        else None
    )

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        run_name=config.training.wandb_run_name,
        config=config,
        enabled=config.training.use_wandb,
        logger=logger,
        provenance=manifest,
        resume_id=resume_wandb_id,
    )

    # Build model + trainer (populates model param count in summary_info).
    trainer = build_trainer(
        config,
        train_loader,
        val_loader,
        per_variant_val_loaders,
        device,
        logger,
        wandb_logger,
        summary_info,
        provenance=manifest,
    )

    # Log configuration (now that the model param count is known)
    logger.info("\n" + config.summary(runtime_info=summary_info))

    # Resume from checkpoint if specified
    if config.training.resume_from:
        trainer.load_checkpoint(config.training.resume_from)

    # Start training
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_dir=config.training.save_dir,
        early_stopping_patience=config.training.early_stopping_patience,
    )

    # Log completion
    logger.info("Training complete!")
    metric_name = "avg SI-SDRi" if trainer.per_variant_mode else "SI-SDR"
    logger.info(f"Best validation {metric_name}: {trainer.best_val_sisdr:.2f} dB")

    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
