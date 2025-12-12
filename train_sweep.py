"""Training script for W&B hyperparameter sweeps."""

import wandb

from config import load_config_for_run
from training.trainer import Trainer
from utils import set_seed, setup_warnings, setup_logger, setup_device_and_amp, WandbLogger
from train import create_model, create_dataloaders


def main():
    setup_warnings()
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

    # Create dataloaders and model using helper functions from train.py
    train_loader, val_loader = create_dataloaders(config, summary_info)
    model = create_model(config, summary_info)

    # Setup WandB logger (use existing run)
    wandb_logger = WandbLogger(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        run_name=config.training.wandb_run_name,
        config=config,
        enabled=True,  # Always enabled for sweeps
        logger=logger,
        run=run,
    )

    # Log configuration
    logger.info("\n" + config.summary())
    logger.info(f"\nDataset: {config.data.dataset_type}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Train samples: {summary_info['train_samples']}")
    logger.info(f"Val samples: {summary_info['val_samples']}")

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

    # Run training
    trainer.train(num_epochs=config.training.num_epochs, save_dir=config.training.save_dir)

    # Log completion
    logger.info("\n" + "=" * 80)
    logger.info(f"Training complete! Best validation SI-SDR: {trainer.best_val_sisdr:.2f} dB")
    logger.info("=" * 80)

    # Log best metric to W&B
    wandb.log({"best_val_sisdr": trainer.best_val_sisdr})
    wandb.finish()


if __name__ == "__main__":
    main()
