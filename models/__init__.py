"""Models module for speech separation architectures."""

from .conv_tasnet import ConvTasNet
from .sepformer import SepFormer
from .dprnn import DPRNN
from .spmamba import SPMamba
from .mamba_tasnet import MambaTasNet
from .dpmamba import DPMamba
from .sepmamba import SepMamba

# Model registry for easy switching between architectures
MODELS = {
    'convtasnet': ConvTasNet,
    'sepformer': SepFormer,
    'dprnn': DPRNN,
    'spmamba': SPMamba,
    'mamba_tasnet': MambaTasNet,
    'dpmamba': DPMamba,
    'sepmamba': SepMamba,
}


def get_model(model_type: str):
    """Get model class by name."""
    if model_type not in MODELS:
        available = ', '.join(MODELS.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}"
        )
    return MODELS[model_type]


__all__ = [
    'ConvTasNet', 'SepFormer', 'DPRNN', 'SPMamba',
    'MambaTasNet', 'DPMamba', 'SepMamba',
    'MODELS', 'get_model',
]
