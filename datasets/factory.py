"""Dataset factory for creating dataloaders from configuration.

This module provides factory functions for creating datasets and dataloaders,
eliminating duplication and centralizing dataloader configuration.
"""

from typing import Optional, List, Tuple
from torch.utils.data import DataLoader
from config import DataConfig
from datasets import get_dataset, polsess_collate_fn


def create_dataloader(
    dataset_class,
    data_root: str,
    subset: str,
    config: DataConfig,
    allowed_variants: Optional[List[str]] = None,
    shuffle: bool = False,
) -> DataLoader:
    """Create dataloader with standard configuration."""
    # Map subset to max_samples config
    max_samples = None
    if subset == "train":
        max_samples = config.train_max_samples
    elif subset == "val":
        max_samples = config.val_max_samples
    
    # Create dataset
    dataset = dataset_class(
        data_root,
        subset=subset,
        task=config.task,
        max_samples=max_samples,
        allowed_variants=allowed_variants,
    )
    
    # Create dataloader with standard settings
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        prefetch_factor=(
            config.prefetch_factor if config.num_workers > 0 else None
        ),
        collate_fn=polsess_collate_fn,
    )
    
    return dataloader