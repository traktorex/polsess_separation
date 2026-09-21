"""Models module for speech separation architectures.

Also published as the pip-installable ``polsess-models`` distribution (import
name ``polsess_models``): see ``models/README.md`` and the root ``pyproject.toml``.
"""

# Version of the polsess-models distribution (read by pyproject.toml). Bump it
# when models/ changes in a way a consumer can notice; models/README.md has the rule.
__version__ = "0.1.0"

from .conv_tasnet import ConvTasNet
from .sepformer import SepFormer
from .mossformer2 import MossFormer2
from .tf_mossformer import TFMossFormer
from .dprnn import DPRNN

# Mamba models require mamba-ssm (Linux + CUDA only). The polsess-models wheel
# ships without models/mamba/ (GPL-3.0 upstream), hence the guard: there the
# subpackage itself is absent. In this repository it always exists, and
# models/mamba/__init__.py reports a missing mamba-ssm through MAMBA_AVAILABLE.
try:
    from .mamba import MAMBA_AVAILABLE
except ModuleNotFoundError as err:
    if err.name != f"{__name__}.mamba":   # anything else missing is a real error
        raise
    MAMBA_AVAILABLE = False

if MAMBA_AVAILABLE:
    from .spmamba import SPMamba
    from .mamba_tasnet import MambaTasNet
    from .dpmamba import DPMamba

# Model registry for easy switching between architectures
MODELS = {
    'convtasnet': ConvTasNet,
    'sepformer': SepFormer,
    'mossformer2': MossFormer2,
    'tf_mossformer': TFMossFormer,
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
        if model_type in MAMBA_MODELS and not MAMBA_AVAILABLE:
            hint = (" (Mamba family: requires the mamba-ssm library, Linux + CUDA only,"
                    " and is not part of the polsess-models distribution)")
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}{hint}"
        )
    return MODELS[model_type]


# After get_model and __version__: inference.py imports both from this package.
from .inference import load_model_for_inference, load_separator

__all__ = [
    'ConvTasNet', 'SepFormer', 'MossFormer2', 'TFMossFormer', 'DPRNN', 'MAMBA_AVAILABLE',
    'MODELS', 'MAMBA_MODELS', 'get_model',
    'load_model_for_inference', 'load_separator',
]

if MAMBA_AVAILABLE:
    __all__ += ['SPMamba', 'MambaTasNet', 'DPMamba']
