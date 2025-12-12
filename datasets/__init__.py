"""Dataset loaders for speech separation tasks."""

from .polsess_dataset import PolSESSDataset, polsess_collate_fn
from .libri2mix_dataset import Libri2MixDataset, libri2mix_collate_fn

# Dataset registry for easy switching between datasets
DATASETS = {
    "polsess": PolSESSDataset,
    "libri2mix": Libri2MixDataset,
}


def get_dataset(dataset_type: str):
    """Get dataset class by name."""
    if dataset_type not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Available datasets: {available}"
        )
    return DATASETS[dataset_type]


__all__ = [
    "PolSESSDataset",
    "polsess_collate_fn",
    "Libri2MixDataset",
    "libri2mix_collate_fn",
    "DATASETS",
    "get_dataset",
]
