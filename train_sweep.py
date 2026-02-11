"""Training script for W&B hyperparameter sweeps."""

import wandb
from torch.utils.data import DataLoader

from config import load_config_for_run
from models.factory import create_model_from_config
from datasets import get_dataset, polsess_collate_fn
from training.trainer import Trainer
from utils import (
    set_seed,
    setup_warnings,
    setup_logger,
    setup_device_and_amp,
    WandbLogger,
    apply_torch_compile,
)


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

    # Create dataloaders
    dataset_class = get_dataset(config.data.dataset_type)
    
    # Get dataset root
    if config.data.dataset_type == "polsess":
        data_root = config.data.polsess.data_root
    else:
        raise ValueError(f"Dataset {config.data.dataset_type} not configured")
    
    # Determine variants
    train_variants = None
    if config.training.curriculum_learning:
        train_variants = config.training.curriculum_learning[0].get("variants")
    
    # Create train dataset
    train_dataset = dataset_class(
        data_root,
        subset="train",
        task=config.data.task,
        max_samples=config.data.train_max_samples,
        allowed_variants=train_variants,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        collate_fn=polsess_collate_fn,
    )
    
    # Create val dataset
    val_dataset = dataset_class(
        data_root,
        subset="val",
        task=config.data.task,
        max_samples=config.data.val_max_samples,
        allowed_variants=config.training.validation_variants,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        collate_fn=polsess_collate_fn,
    )
    
    # Update summary info
    summary_info["train_samples"] = len(train_loader.dataset)
    summary_info["val_samples"] = len(val_loader.dataset)

    # Create model using factory
    model = create_model_from_config(config.model, summary_info)

    # Apply torch.compile (PyTorch 2.0+, Linux only)
    model = apply_torch_compile(model, logger=logger)

    # Setup WandB logger (use existing run)
    wandb_logger = WandbLogger(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        run_name=config.training.wandb_run_name,
        config=config,
        enabled=True,  # Always enabled for sweeps
        logger=logger,
        run=run,
        upload_checkpoints=False,  # Don't upload model artifacts during sweeps
    )

    # Log configuration
    logger.info("\n" + config.summary())
    logger.info(f"\nDataset: {config.data.dataset_type}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Train samples: {summary_info['train_samples']}")
    logger.info(f"Val samples: {summary_info['val_samples']}")
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

    # Run training
    early_stopping = getattr(config.training, 'early_stopping_patience', None)
    trainer.train(
        num_epochs=config.training.num_epochs, 
        save_dir=config.training.save_dir,
        early_stopping_patience=early_stopping
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
