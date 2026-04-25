"""Models module for speech separation architectures."""

from .conv_tasnet import ConvTasNet
from .sepformer import SepFormer
from .dprnn import DPRNN

# Mamba models require mamba-ssm (Linux + CUDA only)
from .mamba import MAMBA_AVAILABLE

if MAMBA_AVAILABLE:
    from .spmamba import SPMamba
    from .mamba_tasnet import MambaTasNet
    from .dpmamba import DPMamba

# Model registry for easy switching between architectures
MODELS = {
    'convtasnet': ConvTasNet,
    'sepformer': SepFormer,
    'dprnn': DPRNN,
}

if MAMBA_AVAILABLE:
    MODELS.update({
        'spmamba': SPMamba,
        'mamba_tasnet': MambaTasNet,
        'dpmamba': DPMamba,
    })

# Models that wrap mamba-ssm CUDA kernels. These need special handling in two
# places: bf16 autocast without GradScaler (the kernels run float32 internally),
# and skipping torch.compile (Dynamo tracing into mamba_ssm.MambaInnerFn breaks
# the delta/conv1d_out dtype contract — observed on native Linux; WSL2 happens
# to graph-break before this fires).
MAMBA_MODELS = ('spmamba', 'mamba_tasnet', 'dpmamba')


def get_model(model_type: str):
    """Get model class by name."""
    if model_type not in MODELS:
        available = ', '.join(MODELS.keys())
        hint = ""
        if model_type in ('spmamba', 'mamba_tasnet', 'dpmamba') and not MAMBA_AVAILABLE:
            hint = " (requires mamba-ssm library — Linux + CUDA only)"
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}{hint}"
        )
    return MODELS[model_type]


__all__ = [
    'ConvTasNet', 'SepFormer', 'DPRNN', 'MAMBA_AVAILABLE',
    'MODELS', 'MAMBA_MODELS', 'get_model',
]

if MAMBA_AVAILABLE:
    __all__ += ['SPMamba', 'MambaTasNet', 'DPMamba']
