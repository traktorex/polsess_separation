"""Training script for speech separation using various model architectures."""

from utils import warning_filters  # noqa: F401  must precede speechbrain imports (registers filters)

import torch
from torch.utils.data import DataLoader
from config import get_config_from_args
from models.factory import create_model_from_config
from datasets import get_dataset, polsess_collate_fn
from training.trainer import Trainer
from utils import (
    set_seed,
    setup_warnings,
    setup_logger,
    setup_device_and_amp,
    WandbLogger,
    compile_for_model_type,
)
import os

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

    # Create dataloaders
    dataset_class = get_dataset(config.data.dataset_type)
    
    # Get dataset root
    if config.data.dataset_type == "polsess":
        data_root = config.data.polsess.data_root
    else:
        raise ValueError(f"Dataset {config.data.dataset_type} not configured for training. Only PolSESS is supported for training")
    
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
    
    # Create val dataset(s)
    # Canonical MM-IPC variant order (indoor + outdoor), matching evaluate.py.
    ALL_VARIANTS = ["SER", "SR", "ER", "R", "SE", "S", "E", "C"]

    val_loader = None
    per_variant_val_loaders = None

    if config.training.per_variant_validation:
        # Build one dataloader per variant. Each forces a specific MM-IPC variant
        # via allowed_variants=[v], so every val sample is re-rendered 8 times.
        filter_set = config.training.validation_variants
        variants_to_use = [v for v in ALL_VARIANTS if filter_set is None or v in filter_set]
        per_variant_val_loaders = {}
        for variant in variants_to_use:
            v_dataset = dataset_class(
                data_root,
                subset="val",
                task=config.data.task,
                max_samples=config.data.val_max_samples,
                allowed_variants=[variant],
            )
            per_variant_val_loaders[variant] = DataLoader(
                v_dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
                collate_fn=polsess_collate_fn,
            )
        # All variant loaders share the same underlying val pool size; pick one for the summary.
        first_loader = next(iter(per_variant_val_loaders.values()))
        summary_info["val_samples"] = len(first_loader.dataset)
        summary_info["val_variants"] = list(per_variant_val_loaders.keys())
    else:
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
        summary_info["val_samples"] = len(val_loader.dataset)

    # Update summary info
    summary_info["train_samples"] = len(train_loader.dataset)

    # Create model using factory
    model = create_model_from_config(config.model, summary_info)

    # Apply torch.compile (PyTorch 2.0+, Linux only) with per-architecture
    # settings — see compile_for_model_type for the Mamba/MossFormer2 rationale.
    model = compile_for_model_type(model, config.model.model_type, logger=logger)

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
    logger.info("\n" + config.summary(runtime_info=summary_info))

    # Create trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config,
        device=device,
        logger=logger,
        wandb_logger=wandb_logger,
        per_variant_val_loaders=per_variant_val_loaders,
    )

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
