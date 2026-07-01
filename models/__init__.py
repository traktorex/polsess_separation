"""Models module for speech separation architectures."""

from .conv_tasnet import ConvTasNet
from .sepformer import SepFormer
from .mossformer2 import MossFormer2
from .dprnn import DPRNN

# Mamba models require mamba-ssm (Linux + CUDA only)
from .mamba import MAMBA_AVAILABLE

if MAMBA_AVAILABLE:
    from .spmamba import SPMamba
    from .mamba_tasnet import MambaTasNet
    from .dpmamba import DPMamba

# SPMamba3 needs Mamba-3, which lives in the dedicated venv_mamba3 (separate from
# the Mamba-1 mamba-ssm in the main venv) — so it has its own availability flag.
# The import is always safe: the module guards its Mamba-3 import internally and
# defers the hard error to construction.
from .spmamba3 import SPMamba3, MAMBA3_AVAILABLE

# Model registry for easy switching between architectures
MODELS = {
    'convtasnet': ConvTasNet,
    'sepformer': SepFormer,
    'mossformer2': MossFormer2,
    'dprnn': DPRNN,
}

if MAMBA_AVAILABLE:
    MODELS.update({
        'spmamba': SPMamba,
        'mamba_tasnet': MambaTasNet,
        'dpmamba': DPMamba,
    })

if MAMBA3_AVAILABLE:
    MODELS['spmamba3'] = SPMamba3

# Models that wrap mamba-ssm CUDA/Triton kernels. These need special handling in
# two places: bf16 autocast without GradScaler (the kernels run float32/bf16
# internally), and skipping torch.compile (Dynamo tracing into the mamba_ssm
# kernels breaks the delta/conv1d_out dtype contract — observed on native Linux;
# WSL2 happens to graph-break before this fires). spmamba3 is included so it
# inherits both behaviours; membership is independent of whether it's registered.
MAMBA_MODELS = ('spmamba', 'mamba_tasnet', 'dpmamba', 'spmamba3')


def get_model(model_type: str):
    """Get model class by name."""
    if model_type not in MODELS:
        available = ', '.join(MODELS.keys())
        hint = ""
        if model_type in ('spmamba', 'mamba_tasnet', 'dpmamba') and not MAMBA_AVAILABLE:
            hint = " (requires mamba-ssm library — Linux + CUDA only)"
        elif model_type == 'spmamba3' and not MAMBA3_AVAILABLE:
            hint = " (requires Mamba-3: build mamba-ssm from source in venv_mamba3 — see CLAUDE.md)"
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}{hint}"
        )
    return MODELS[model_type]


__all__ = [
    'ConvTasNet', 'SepFormer', 'MossFormer2', 'DPRNN', 'MAMBA_AVAILABLE',
    'MAMBA3_AVAILABLE', 'SPMamba3', 'MODELS', 'MAMBA_MODELS', 'get_model',
]

if MAMBA_AVAILABLE:
    __all__ += ['SPMamba', 'MambaTasNet', 'DPMamba']
