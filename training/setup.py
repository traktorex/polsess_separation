"""Shared setup for the train.py and train_sweep.py entry points.

Both entry points build identical dataloaders, model, and Trainer; only config
loading, W&B wiring, and completion logging differ between them. Those shared
blocks live here so the two mains can't drift (survey gap 11) — each main is now
config-loading + these builders + ``trainer.train()``.
"""

import random

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, polsess_collate_fn
from models.factory import create_model_from_config
from training.trainer import Trainer
from utils import compile_for_model_type, configure_determinism

# Canonical MM-IPC variant order (indoor + outdoor), matching evaluate.py.
ALL_VARIANTS = ["SER", "SR", "ER", "R", "SE", "S", "E", "C"]


def _resolve_data_root(config) -> str:
    """Return the training data root, rejecting non-PolSESS datasets."""
    if config.data.dataset_type == "polsess":
        return config.data.polsess.data_root
    raise ValueError(
        f"Dataset {config.data.dataset_type} not configured for training. "
        "Only PolSESS is supported for training"
    )


def seed_worker(worker_id):
    """Seed the ``random`` module in each DataLoader worker (survey gap 6).

    torch hands each worker a distinct base seed derived from the loader's
    generator; mirror it into ``random`` — the module ``PolSESSDataset`` uses to
    pick the training MM-IPC variant (``_choose_variant`` -> ``random.choice``) —
    so the augmentation stream is reproducible by contract instead of by
    incidental global-RNG ordering. (With ``num_workers=0`` no worker forks, and
    the main-process ``random`` seeded by ``set_seed`` governs instead.)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)


def build_dataloaders(config, summary_info, logger=None):
    """Build the train + validation dataloaders shared by both entry points.

    Populates ``summary_info`` with train/val sample counts (and the val variant
    list in per-variant mode), and returns
    ``(train_loader, val_loader, per_variant_val_loaders)`` where exactly one of
    the last two is None (the Trainer asserts this).
    """
    # Apply the run's determinism policy (global torch state) at the single
    # shared setup site both entry points funnel through, before any model runs.
    # Default (config.training.deterministic is None) is a no-op == today.
    configure_determinism(config.training.deterministic, logger=logger)

    dataset_class = get_dataset(config.data.dataset_type)
    data_root = _resolve_data_root(config)
    seed = config.training.seed

    # Seeded generator + worker_init_fn make the training MM-IPC variant stream
    # reproducible by contract (survey gap 6). NOTE: adding generator= changes the
    # shuffle RNG stream relative to pre-2026-07 runs — accepted and intended per
    # the plan; future runs become reproducible, not bit-identical to historical
    # ones (which relied on incidental global-RNG ordering).
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    # Determine the training variants (curriculum stage 0, or None for all).
    train_variants = None
    if config.training.curriculum_learning:
        train_variants = config.training.curriculum_learning[0].get("variants")

    train_dataset = dataset_class(
        data_root,
        subset="train",
        task=config.data.task,
        max_samples=config.data.train_max_samples,
        allowed_variants=train_variants,
    )

    # persistent_workers is intentionally left at its default (False). Curriculum
    # learning mutates train_dataset.allowed_variants *in place* between epochs
    # (Trainer._update_training_variants); that mutation only reaches the workers
    # because they are re-forked every epoch. Enabling persistent_workers would
    # pin every epoch to the stage-0 variant set (survey gap 18). If a
    # persistent_workers knob is ever added, gate it on curriculum_learning being
    # None.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        collate_fn=polsess_collate_fn,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )

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
        # All variant loaders share the same underlying val pool size; pick one.
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

    summary_info["train_samples"] = len(train_loader.dataset)
    return train_loader, val_loader, per_variant_val_loaders


def build_trainer(
    config,
    train_loader,
    val_loader,
    per_variant_val_loaders,
    device,
    logger,
    wandb_logger,
    summary_info,
    provenance=None,
):
    """Build the model (with per-arch torch.compile) and wrap it in a Trainer.

    Shared by both entry points; the caller supplies the already-constructed
    ``wandb_logger`` (enabled/attached differently for standalone vs sweep runs)
    and the run ``provenance`` manifest to embed in checkpoints. Populates
    ``summary_info["model_params_millions"]`` via the model factory.
    """
    model = create_model_from_config(config.model, summary_info)

    # Apply torch.compile (PyTorch 2.0+, Linux only) with per-architecture
    # settings — see compile_for_model_type for the Mamba/MossFormer2 rationale.
    model = compile_for_model_type(model, config.model.model_type, logger=logger)

    return Trainer(
        model,
        train_loader,
        val_loader,
        config,
        device=device,
        logger=logger,
        wandb_logger=wandb_logger,
        per_variant_val_loaders=per_variant_val_loaders,
        provenance=provenance,
    )
