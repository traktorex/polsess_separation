"""Training script for W&B hyperparameter sweeps."""

from utils import warning_filters  # noqa: F401  must precede speechbrain imports (registers filters)

import torch
import wandb

from config import load_config_for_run
from training.setup import build_dataloaders, build_trainer
from utils import (
    set_seed,
    setup_warnings,
    setup_logger,
    setup_device_and_amp,
    WandbLogger,
    collect_run_manifest,
)


def main():
    setup_warnings()
    torch.set_float32_matmul_precision('high')
    run = wandb.init()
    sweep_config = wandb.config

    # Load config with sweep overrides
    config = load_config_for_run(sweep_config)
    set_seed(config.training.seed)

    # Setup logger
    logger = setup_logger(
        name="polsess-sweep",
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

    # Setup WandB logger (use existing sweep run)
    wandb_logger = WandbLogger(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        run_name=config.training.wandb_run_name,
        config=config,
        enabled=True,  # Always enabled for sweeps
        logger=logger,
        run=run,
        upload_checkpoints=False,  # Don't upload model artifacts during sweeps
        provenance=manifest,
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

    # Run training
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_dir=config.training.save_dir,
        early_stopping_patience=config.training.early_stopping_patience,
    )

    # Log completion
    logger.info("\n" + "=" * 80)
    logger.info(f"Training complete! Best validation SI-SDR: {trainer.best_val_sisdr:.2f} dB")
    logger.info("=" * 80)

    # Log best metric to W&B
    wandb.log({"best_val_sisdr": trainer.best_val_sisdr})
    wandb.finish()


if __name__ == "__main__":
    main()
